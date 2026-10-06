use tensor::{QuantBits, QuantScheme, TensorError, bf16_to_f32, quantize};

fn bf16(values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}

/// Deterministic values in [-1, 1) that bf16 holds exactly.
fn weights(count: usize, seed: u64) -> Vec<u16> {
    let mut state = seed;
    let values: Vec<f32> = (0..count)
        .map(|_| {
            state = state
                .wrapping_mul(6_364_136_223_846_793_005)
                .wrapping_add(1_442_695_040_888_963_407);
            ((state >> 40) as f32 / 16_777_216.0) * 2.0 - 1.0
        })
        .collect();
    bf16(&values)
}

fn scheme(bits: QuantBits, group: usize) -> QuantScheme {
    QuantScheme { bits, group }
}

#[test]
fn group_sizes_must_be_positive_multiples_of_32() {
    for bad in [0, 1, 31, 33, 100] {
        assert!(QuantScheme::new(QuantBits::Int8, bad).is_err(), "{bad}");
    }
    for good in [32, 64, 128, 256] {
        assert!(QuantScheme::new(QuantBits::Int4, good).is_ok(), "{good}");
    }
}

#[test]
fn every_value_is_within_half_a_step_and_the_largest_is_exact() {
    for (bits, group) in [
        (QuantBits::Int8, 32),
        (QuantBits::Int8, 128),
        (QuantBits::Int4, 32),
        (QuantBits::Int4, 128),
    ] {
        let (rows, cols) = (5, 256);
        let original = weights(rows * cols, 3);
        let q = quantize(&original, rows, cols, scheme(bits, group)).unwrap();
        let mut row = vec![0.0; cols];
        for r in 0..rows {
            q.dequantize_row(r, &mut row).unwrap();
            for (g, chunk) in original[r * cols..(r + 1) * cols].chunks(group).enumerate() {
                let scale = q.scales[r * (cols / group) + g];
                let largest = chunk
                    .iter()
                    .map(|b| bf16_to_f32(*b).abs())
                    .fold(0.0, f32::max);
                assert!((scale * bits.max_code() as f32 - largest).abs() <= largest * 1e-6);
                for (i, b) in chunk.iter().enumerate() {
                    let error = (row[g * group + i] - bf16_to_f32(*b)).abs();
                    assert!(
                        error <= scale * 0.5 * 1.0001,
                        "{bits:?} group {group}: error {error} vs step {scale}"
                    );
                }
            }
        }
    }
}

#[test]
fn int4_packs_the_even_column_low_with_an_offset_of_eight() {
    let mut values = vec![0.0_f32; 32];
    values[0] = 7.0;
    values[1] = -7.0;
    values[2] = 3.0;
    values[3] = 0.0;
    let q = quantize(&bf16(&values), 1, 32, scheme(QuantBits::Int4, 32)).unwrap();
    assert_eq!(q.scales, [1.0]);
    assert_eq!(q.codes.len(), 16);
    assert_eq!(q.codes[0], 15 | (1 << 4), "7 and -7");
    assert_eq!(q.codes[1], 11 | (8 << 4), "3 and 0");
    assert!(
        q.codes[2..].iter().all(|b| *b == 0x88),
        "zeros are the offset"
    );
}

#[test]
fn int8_codes_are_twos_complement_and_clamped_to_127() {
    let mut values = vec![0.0_f32; 32];
    values[0] = 127.0;
    values[1] = -127.0;
    values[2] = 1.0;
    let q = quantize(&bf16(&values), 1, 32, scheme(QuantBits::Int8, 32)).unwrap();
    assert_eq!(q.scales, [1.0]);
    assert_eq!(&q.codes[..3], [127_u8, 129, 1]);
    assert!(q.codes.iter().all(|b| *b as i8 != i8::MIN));
}

#[test]
fn negating_the_weights_negates_the_codes() {
    let original = weights(64, 9);
    let negated: Vec<u16> = original.iter().map(|b| b ^ 0x8000).collect();
    let a = quantize(&original, 1, 64, scheme(QuantBits::Int8, 64)).unwrap();
    let b = quantize(&negated, 1, 64, scheme(QuantBits::Int8, 64)).unwrap();
    assert_eq!(a.scales, b.scales);
    for (x, y) in a.codes.iter().zip(&b.codes) {
        assert_eq!(i32::from(*x as i8), -i32::from(*y as i8));
    }
}

#[test]
fn a_group_of_zeros_gets_scale_zero_and_dequantizes_to_zero() {
    for bits in [QuantBits::Int8, QuantBits::Int4] {
        let mut values = vec![0.0_f32; 64];
        values[40] = 2.0;
        let q = quantize(&bf16(&values), 1, 64, scheme(bits, 32)).unwrap();
        assert_eq!(q.scales[0], 0.0);
        let mut row = vec![1.0; 64];
        q.dequantize_row(0, &mut row).unwrap();
        assert!(row[..32].iter().all(|v| *v == 0.0));
        assert_eq!(row[40], 2.0);
    }
}

#[test]
fn bad_inputs_are_refused_with_a_reason() {
    let s = scheme(QuantBits::Int8, 32);
    let ok = weights(64, 1);
    assert!(
        matches!(quantize(&ok, 3, 64, s), Err(TensorError::Shape { .. })),
        "length"
    );
    assert!(
        matches!(quantize(&[], 0, 0, s), Err(TensorError::Shape { .. })),
        "empty"
    );
    let err = quantize(&weights(48, 1), 1, 48, s).unwrap_err().to_string();
    assert!(err.contains("48 columns") && err.contains("32"), "{err}");
    for bad in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
        let mut values = bf16(&[0.5_f32; 64]);
        values[63] = (bad.to_bits() >> 16) as u16;
        let err = quantize(&values, 2, 32, s).unwrap_err().to_string();
        assert!(err.contains("row 1"), "{err}");
    }
}

#[test]
fn byte_counts_include_the_scales_and_refuse_overflow() {
    let int8 = scheme(QuantBits::Int8, 128);
    let int4 = scheme(QuantBits::Int4, 128);
    assert_eq!(
        int8.matrix_bytes(2048, 2048),
        Some(2048 * 2048 + 2048 * 16 * 4)
    );
    assert_eq!(
        int4.matrix_bytes(2048, 2048),
        Some(2048 * 1024 + 2048 * 16 * 4)
    );
    assert_eq!(int8.matrix_bytes(usize::MAX, 2), None);
}

#[test]
fn dequantizing_checks_the_row_and_output_length() {
    let q = quantize(&weights(64, 5), 2, 32, scheme(QuantBits::Int8, 32)).unwrap();
    assert!(q.dequantize_row(2, &mut [0.0; 32]).is_err());
    assert!(q.dequantize_row(0, &mut [0.0; 31]).is_err());
    assert!(q.dequantize_row(1, &mut [0.0; 32]).is_ok());
}

#[test]
fn quantizing_in_parallel_gives_exactly_the_rows_quantized_one_by_one() {
    // 300 x 512 values is past the size where the work is split across threads
    let (rows, cols) = (300, 512);
    let original = weights(rows * cols, 11);
    for bits in [QuantBits::Int8, QuantBits::Int4] {
        let whole = quantize(&original, rows, cols, scheme(bits, 128)).unwrap();
        let (mut codes, mut scales) = (Vec::new(), Vec::new());
        for row in original.chunks(cols) {
            let one = quantize(row, 1, cols, scheme(bits, 128)).unwrap();
            codes.extend(one.codes);
            scales.extend(one.scales);
        }
        assert_eq!((whole.codes, whole.scales), (codes, scales), "{bits:?}");
    }
}

#[test]
fn a_bad_weight_in_a_late_row_of_a_large_matrix_is_reported_with_its_row() {
    let (rows, cols) = (300, 512);
    let mut original = weights(rows * cols, 12);
    original[250 * cols + 7] = (f32::NAN.to_bits() >> 16) as u16;
    let err = quantize(&original, rows, cols, scheme(QuantBits::Int8, 128))
        .unwrap_err()
        .to_string();
    assert!(err.contains("row 250"), "{err}");
}
