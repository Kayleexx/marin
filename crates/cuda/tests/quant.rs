//! Compares the quantized matrix kernels with a plain f64 product of the dequantized weights.
//! Without a GPU these tests skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod common;

use common::{assert_close, gpu, values};
use cuda::{CudaError, Gpu};
use tensor::{QuantBits, QuantScheme, Quantized, quantize};

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}

fn scheme(bits: QuantBits, group: usize) -> QuantScheme {
    QuantScheme { bits, group }
}

/// What `matmul_quant` must compute: the dequantized rows dotted with every token, in f64.
fn reference(q: &Quantized, x: &[f32], tokens: usize) -> Result<Vec<f32>, tensor::TensorError> {
    let mut row = vec![0.0; q.cols];
    let mut out = vec![0.0; tokens.saturating_mul(q.rows)];
    for r in 0..q.rows {
        q.dequantize_row(r, &mut row)?;
        for (t, input) in x.chunks_exact(q.cols).enumerate() {
            let dot: f64 = row
                .iter()
                .zip(input)
                .map(|(w, v)| f64::from(*w) * f64::from(*v))
                .sum();
            if let Some(slot) = out.get_mut(t.saturating_mul(q.rows).saturating_add(r)) {
                *slot = dot as f32;
            }
        }
    }
    Ok(out)
}

fn case(
    gpu: &Gpu,
    scheme: QuantScheme,
    (rows, cols, tokens): (usize, usize, usize),
    weights: &[f32],
) -> Result<(), Box<dyn std::error::Error>> {
    let q = quantize(&bf16_bits(weights), rows, cols, scheme)?;
    let x = values(tokens.saturating_mul(cols), 77);
    let expected = reference(&q, &x, tokens)?;
    let device = gpu.upload_quantized(&q)?;
    let mut out = gpu.zeros(tokens.saturating_mul(rows))?;
    gpu.matmul_quant(&device, &gpu.upload(&x)?, tokens, &mut out)?;
    assert_close(&gpu.download(&out)?, &expected, 1e-4);
    Ok(())
}

#[test]
fn quantized_matmul_matches_the_dequantized_reference_for_every_format_group_and_token_count() {
    let Some(gpu) = gpu().unwrap() else { return };
    // tokens 1 to 15 take the one-warp-per-row kernel, 16 and up the tensor cores; the shapes cross the 4-row,
    // 8-token, 64-row and 64-token tile edges
    for bits in [QuantBits::Int8, QuantBits::Int4] {
        for group in [32, 64, 128, 256] {
            for (rows, cols) in [(5, 256), (70, 512), (130, 256)] {
                let weights = values(rows * cols, 5);
                for tokens in [1, 3, 8, 9, 15, 16, 40, 65] {
                    case(&gpu, scheme(bits, group), (rows, cols, tokens), &weights).unwrap_or_else(
                        |e| panic!("{bits:?} group {group} {rows}x{cols} tokens {tokens}: {e}"),
                    );
                }
            }
        }
    }
}

/// The precise Int8 product keeps the f32 activations whole (three bf16 parts) on the tensor-core path, so it is as close to the
/// f64 reference as the f32 kernel is; two parts, as `matmul_quant` uses, would be about ten times further off.
#[test]
fn precise_int8_matmul_is_float32_grade_on_the_tensor_core_path() {
    let Some(gpu) = gpu().unwrap() else { return };
    for group in [32, 128, 256] {
        for (rows, cols) in [(70, 512), (130, 1024)] {
            let q = quantize(
                &bf16_bits(&values(rows * cols, 5)),
                rows,
                cols,
                scheme(QuantBits::Int8, group),
            )
            .unwrap();
            let device = gpu.upload_quantized(&q).unwrap();
            for (tokens, magnitude) in [
                (1, 1.0),
                (15, 1.0),
                (16, 1.0),
                (17, 1.0),
                (40, 1.0e3),
                (64, 1.0),
                (65, 1.0e-3),
            ] {
                let x: Vec<f32> = values(tokens * cols, 77)
                    .iter()
                    .map(|v| v * magnitude)
                    .collect();
                let expected = reference(&q, &x, tokens).unwrap();
                let mut out = gpu.zeros(tokens * rows).unwrap();
                gpu.matmul_quant_precise(&device, &gpu.upload(&x).unwrap(), tokens, &mut out)
                    .unwrap();
                let got = gpu.download(&out).unwrap();
                let scale = expected.iter().fold(0.0_f32, |m, e| m.max(e.abs()));
                let worst = got
                    .iter()
                    .zip(&expected)
                    .fold(0.0_f32, |m, (a, e)| m.max((a - e).abs()));
                assert!(
                    worst <= 1.0e-6 * scale,
                    "group {group} {rows}x{cols} tokens {tokens}: worst {worst} of {scale}"
                );
            }
        }
    }
}

#[test]
fn quantized_matmul_handles_model_sized_rows_and_zero_groups_and_extreme_weights() {
    let Some(gpu) = gpu().unwrap() else { return };
    let mut weights = values(33 * 2048, 6);
    weights[..128].fill(0.0); // a whole group of zeros has scale 0
    weights[2048..2048 + 256]
        .iter_mut()
        .for_each(|w| *w *= 1.0e-12); // tiny scales
    weights[4096] = 1.0e6; // one huge weight sets its group's scale
    for bits in [QuantBits::Int8, QuantBits::Int4] {
        for tokens in [1, 20] {
            case(&gpu, scheme(bits, 128), (33, 2048, tokens), &weights).unwrap();
        }
    }
}

fn manual(rows: usize, cols: usize, scheme: QuantScheme, codes: usize, scales: usize) -> Quantized {
    Quantized {
        rows,
        cols,
        scheme,
        codes: vec![0; codes],
        scales: vec![0.0; scales],
    }
}

#[test]
fn uploads_reject_codes_and_scales_that_do_not_fit_the_matrix() {
    let Some(gpu) = gpu().unwrap() else { return };
    let (int8, int4) = (scheme(QuantBits::Int8, 32), scheme(QuantBits::Int4, 32));
    gpu.upload_quantized(&manual(2, 64, int8, 128, 4)).unwrap();
    gpu.upload_quantized(&manual(2, 64, int4, 64, 4)).unwrap();
    let rejected = |q: Quantized| matches!(gpu.upload_quantized(&q), Err(CudaError::Shape { .. }));
    assert!(rejected(manual(2, 64, int8, 127, 4)), "codes short");
    assert!(rejected(manual(2, 64, int8, 128, 5)), "scales long");
    assert!(
        rejected(manual(2, 64, int4, 128, 4)),
        "int4 packs two per byte"
    );
    assert!(
        rejected(manual(2, 48, int8, 96, 3)),
        "columns not a multiple of the group"
    );
    let mut odd_group = manual(2, 64, int8, 128, 4);
    odd_group.scheme.group = 48;
    assert!(rejected(odd_group), "group not a multiple of 32");
}

#[test]
fn matmul_quant_rejects_wrongly_sized_buffers_instead_of_launching() {
    let Some(gpu) = gpu().unwrap() else { return };
    let q = quantize(
        &bf16_bits(&values(2 * 64, 3)),
        2,
        64,
        scheme(QuantBits::Int8, 32),
    )
    .unwrap();
    let w = gpu.upload_quantized(&q).unwrap();
    let x = gpu.zeros(64).unwrap();
    assert!(
        gpu.matmul_quant(&w, &gpu.zeros(63).unwrap(), 1, &mut gpu.zeros(2).unwrap())
            .is_err()
    );
    assert!(
        gpu.matmul_quant(&w, &x, 1, &mut gpu.zeros(3).unwrap())
            .is_err()
    );
    assert!(
        gpu.matmul_quant(&w, &x, 2, &mut gpu.zeros(2).unwrap())
            .is_err()
    );
    gpu.matmul_quant(&w, &x, 1, &mut gpu.zeros(2).unwrap())
        .unwrap();
    gpu.matmul_quant(&w, &gpu.zeros(0).unwrap(), 0, &mut gpu.zeros(0).unwrap())
        .unwrap();
}

#[test]
fn int8_embedding_lookup_gives_the_dequantized_rows_and_refuses_what_it_cannot_do() {
    let Some(gpu) = gpu().unwrap() else { return };
    let (rows, cols) = (50, 256);
    let q = quantize(
        &bf16_bits(&values(rows * cols, 8)),
        rows,
        cols,
        scheme(QuantBits::Int8, 64),
    )
    .unwrap();
    let table = gpu.upload_quantized(&q).unwrap();
    let mut ids = gpu.tokens(4).unwrap();
    let mut out = gpu.zeros(3 * cols).unwrap();
    gpu.embed_quant(&table, &[7, 0, 49], &mut ids, &mut out)
        .unwrap();
    let mut expected = Vec::new();
    let mut row = vec![0.0; cols];
    for id in [7, 0, 49] {
        q.dequantize_row(id, &mut row).unwrap();
        expected.extend_from_slice(&row);
    }
    assert_eq!(gpu.download(&out).unwrap(), expected);
    // the same scratch serves a shorter pass, and bad ids, too many tokens and mismatched outputs are refused
    let mut one = gpu.zeros(cols).unwrap();
    gpu.embed_quant(&table, &[49], &mut ids, &mut one).unwrap();
    assert_eq!(gpu.download(&one).unwrap(), expected[2 * cols..]);
    assert!(
        gpu.embed_quant(&table, &[50], &mut ids, &mut one).is_err(),
        "id past the table"
    );
    assert!(
        gpu.embed_quant(&table, &[0; 5], &mut ids, &mut gpu.zeros(5 * cols).unwrap())
            .is_err(),
        "over capacity"
    );
    assert!(
        gpu.embed_quant(&table, &[0, 1], &mut ids, &mut one)
            .is_err(),
        "output too short"
    );
    gpu.embed_quant(&table, &[], &mut ids, &mut gpu.zeros(0).unwrap())
        .unwrap();
    let narrow = quantize(
        &bf16_bits(&values(rows * cols, 8)),
        rows,
        cols,
        scheme(QuantBits::Int4, 64),
    )
    .unwrap();
    let narrow = gpu.upload_quantized(&narrow).unwrap();
    assert!(
        gpu.embed_quant(&narrow, &[1], &mut ids, &mut one).is_err(),
        "int4 embeddings are not supported"
    );
}
