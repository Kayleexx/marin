use crate::{TensorError, bf16_to_f32};

/// Group size used when none is asked for.
pub const DEFAULT_GROUP: usize = 128;
/// Group sizes are multiples of this, so a group never straddles the 16-value steps of the GPU kernels.
const GROUP_MULTIPLE: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum QuantBits {
    Int8,
    Int4,
}

impl QuantBits {
    /// Largest magnitude a code takes; codes are symmetric around zero.
    pub fn max_code(self) -> i32 {
        match self {
            Self::Int8 => 127,
            Self::Int4 => 7,
        }
    }

    pub fn name(self) -> &'static str {
        match self {
            Self::Int8 => "int8",
            Self::Int4 => "int4",
        }
    }
}

/// Weight-only quantization: every `group` consecutive values of a row share one f32 scale.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct QuantScheme {
    pub bits: QuantBits,
    pub group: usize,
}

impl QuantScheme {
    pub fn new(bits: QuantBits, group: usize) -> Result<Self, TensorError> {
        if group == 0 || group.checked_rem(GROUP_MULTIPLE) != Some(0) {
            return Err(TensorError::Quantization(format!(
                "group size {group} must be a positive multiple of {GROUP_MULTIPLE}"
            )));
        }
        Ok(Self { bits, group })
    }

    /// What the model is stored as, for reports. The embedding table and output head are always int8.
    pub fn label(&self) -> String {
        match self.bits {
            QuantBits::Int8 => format!("int8 weights, groups of {}", self.group),
            QuantBits::Int4 => format!(
                "int4 layers with int8 embedding and head, groups of {}",
                self.group
            ),
        }
    }

    /// The scheme the embedding table and output head use: int8 with the same groups.
    pub fn table(&self) -> Self {
        Self {
            bits: QuantBits::Int8,
            group: self.group,
        }
    }

    /// Bytes per weight once quantized, scales included.
    pub fn bytes_per_weight(&self) -> f64 {
        let codes = match self.bits {
            QuantBits::Int8 => 1.0,
            QuantBits::Int4 => 0.5,
        };
        codes + size_of::<f32>() as f64 / self.group as f64
    }

    /// Bytes of codes in one row of `cols` values.
    pub fn row_bytes(&self, cols: usize) -> usize {
        match self.bits {
            QuantBits::Int8 => cols,
            QuantBits::Int4 => cols.div_ceil(2),
        }
    }

    /// Bytes a `rows` by `cols` matrix takes once quantized: the codes plus one f32 scale per group.
    pub fn matrix_bytes(&self, rows: usize, cols: usize) -> Option<u64> {
        let codes = rows.checked_mul(self.row_bytes(cols))?;
        let scales = rows
            .checked_mul(cols.div_ceil(self.group))?
            .checked_mul(size_of::<f32>())?;
        u64::try_from(codes.checked_add(scales)?).ok()
    }
}

/// A quantized matrix: `codes` row by row (int4 packs two per byte, the even column in the low nibble, stored with
/// an offset of 8) and `scales[row * groups_per_row + column / group]`.
#[derive(Debug, PartialEq)]
pub struct Quantized {
    pub rows: usize,
    pub cols: usize,
    pub scheme: QuantScheme,
    pub codes: Vec<u8>,
    pub scales: Vec<f32>,
}

/// An int4 code, in -7..=7, as the unsigned nibble stored for it (offset by 8).
fn nibble(code: i32) -> u8 {
    u8::try_from(code.saturating_add(8)).unwrap_or(0)
}

fn not_finite(row: usize) -> TensorError {
    TensorError::Quantization(format!("row {row} has a NaN or infinite weight"))
}

/// Matrices smaller than this many values are quantized on the calling thread.
const PARALLEL_FROM: usize = 1 << 16;

/// Quantizes a bf16 matrix. Every group's scale is its largest magnitude over the largest code, so that value is
/// exact and the others are off by at most half a step. Large matrices are split by rows across the cores; the
/// result does not depend on how many.
pub fn quantize(
    bits: &[u16],
    rows: usize,
    cols: usize,
    scheme: QuantScheme,
) -> Result<Quantized, TensorError> {
    if rows.checked_mul(cols) != Some(bits.len()) || bits.is_empty() {
        return Err(TensorError::Shape {
            what: "matrix to quantize",
            expected: rows.saturating_mul(cols),
            actual: bits.len(),
        });
    }
    if cols.checked_rem(scheme.group) != Some(0) {
        return Err(TensorError::Quantization(format!(
            "{cols} columns are not a multiple of the group size {}",
            scheme.group
        )));
    }
    let workers = if bits.len() < PARALLEL_FROM {
        1
    } else {
        std::thread::available_parallelism()
            .map_or(1, usize::from)
            .min(rows)
    };
    let rows_each = rows.div_ceil(workers);
    let parts = std::thread::scope(|scope| {
        let handles: Vec<_> = bits
            .chunks(rows_each.saturating_mul(cols))
            .enumerate()
            .map(|(index, chunk)| {
                let first_row = index.saturating_mul(rows_each);
                scope.spawn(move || quantize_rows(chunk, cols, scheme, first_row))
            })
            .collect();
        handles
            .into_iter()
            .map(|handle| {
                handle.join().unwrap_or_else(|_| {
                    Err(TensorError::Quantization(
                        "a quantizing thread failed".into(),
                    ))
                })
            })
            .collect::<Result<Vec<_>, _>>()
    })?;
    let mut codes = Vec::with_capacity(rows.saturating_mul(scheme.row_bytes(cols)));
    let mut scales = Vec::new();
    for (part_codes, part_scales) in parts {
        codes.extend(part_codes);
        scales.extend(part_scales);
    }
    Ok(Quantized {
        rows,
        cols,
        scheme,
        codes,
        scales,
    })
}

/// The codes and scales of whole rows; `first_row` is only for error messages.
fn quantize_rows(
    bits: &[u16],
    cols: usize,
    scheme: QuantScheme,
    first_row: usize,
) -> Result<(Vec<u8>, Vec<f32>), TensorError> {
    let max = scheme.bits.max_code();
    let mut codes = Vec::with_capacity(bits.len());
    let mut scales = Vec::new();
    for (offset, row) in bits.chunks_exact(cols).enumerate() {
        for group in row.chunks_exact(scheme.group) {
            let mut largest = 0.0_f32;
            for value in group.iter().map(|b| bf16_to_f32(*b)) {
                if !value.is_finite() {
                    return Err(not_finite(first_row.saturating_add(offset)));
                }
                largest = largest.max(value.abs());
            }
            let scale = largest / max as f32;
            scales.push(scale);
            let code = |b: &u16| -> i32 {
                if scale > 0.0 {
                    ((bf16_to_f32(*b) / scale).round() as i32).clamp(max.saturating_neg(), max)
                } else {
                    0
                }
            };
            match scheme.bits {
                QuantBits::Int8 => codes.extend(group.iter().map(|b| code(b) as i8 as u8)),
                QuantBits::Int4 => codes.extend(group.chunks_exact(2).map(|pair| match pair {
                    [low, high] => nibble(code(low)) | nibble(code(high)).wrapping_shl(4),
                    _ => 0,
                })),
            }
        }
    }
    Ok((codes, scales))
}

impl Quantized {
    /// The values of row `row` as the GPU kernels read them: code times scale.
    pub fn dequantize_row(&self, row: usize, out: &mut [f32]) -> Result<(), TensorError> {
        if out.len() != self.cols || row >= self.rows {
            return Err(TensorError::Shape {
                what: "dequantized row",
                expected: self.cols,
                actual: out.len(),
            });
        }
        let row_bytes = self.scheme.row_bytes(self.cols);
        let groups = self.cols.div_ceil(self.scheme.group);
        let codes = self
            .codes
            .chunks_exact(row_bytes)
            .nth(row)
            .unwrap_or_default();
        let scales = self
            .scales
            .chunks_exact(groups)
            .nth(row)
            .unwrap_or_default();
        for (index, value) in out.iter_mut().enumerate() {
            let code = match self.scheme.bits {
                QuantBits::Int8 => codes.get(index).map_or(0, |b| i32::from(*b as i8)),
                QuantBits::Int4 => {
                    let byte = codes.get(index / 2).copied().unwrap_or(0);
                    let nibble = if index % 2 == 0 {
                        byte & 0xf
                    } else {
                        byte >> 4
                    };
                    i32::from(nibble).saturating_sub(8)
                }
            };
            let scale = scales
                .get(index.checked_div(self.scheme.group).unwrap_or(0))
                .copied()
                .unwrap_or(0.0);
            *value = code as f32 * scale;
        }
        Ok(())
    }
}
