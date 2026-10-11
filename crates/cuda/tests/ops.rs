//! Compares the GPU kernels (other than attention) with the CPU implementation.
//! Without a GPU these tests skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod common;

use common::{assert_close, gpu, values};
use cuda::{Buffer, CudaError, Gpu};
use tensor::Matrix;

fn bf16_bits(values: &[f32]) -> Vec<u16> {
    values.iter().map(|v| (v.to_bits() >> 16) as u16).collect()
}

#[test]
fn buffers_round_trip_and_ranges_are_checked() {
    let Some(gpu) = gpu().unwrap() else { return };
    let data = values(10, 1);
    let buffer = gpu.upload(&data).unwrap();
    assert_eq!(gpu.download(&buffer).unwrap(), data);
    assert_eq!(gpu.download_range(&buffer, 2, 3).unwrap(), data[2..5]);
    assert!(gpu.download_range(&buffer, 8, 3).is_err());
    assert!(gpu.upload_matrix(2, 3, &[0; 5]).is_err());
}

#[test]
fn matmul_matches_the_cpu_for_aligned_and_odd_shapes_and_token_tiles() {
    let Some(gpu) = gpu().unwrap() else { return };
    for (rows, cols, tokens) in [
        (5, 64, 1),
        (70, 64, 8),
        (9, 13, 11),
        (33, 24, 17),
        (4, 256, 3),
        // from 16 tokens with cols a multiple of 16 the tensor-core kernel runs; these cross its 64-row and 64-token tiles
        (9, 16, 16),
        (70, 48, 65),
        (130, 256, 129),
        (64, 2048, 64),
        (33, 24, 40),
    ] {
        let weights = bf16_bits(&values(rows * cols, 7));
        let x = values(tokens * cols, 9);
        let mut expected = vec![0.0; tokens * rows];
        Matrix::new(rows, cols, weights.clone())
            .unwrap()
            .matmul(&x, &mut expected)
            .unwrap();
        let device = gpu.upload_matrix(rows, cols, &weights).unwrap();
        let input = gpu.upload(&x).unwrap();
        let mut out = gpu.zeros(tokens * rows).unwrap();
        gpu.matmul(&device, &input, tokens, &mut out).unwrap();
        assert_close(&gpu.download(&out).unwrap(), &expected, 1e-4);
    }
}

/// Matrices of the given row counts over the same input, run as separate matmuls and as one grouped launch.
fn grouped_case(gpu: &Gpu, rows: &[usize], cols: usize, tokens: usize) -> Result<(), CudaError> {
    let x = gpu.upload(&values(tokens.saturating_mul(cols), 51))?;
    let mut ws = Vec::new();
    for (i, r) in (60..).zip(rows) {
        let bits = bf16_bits(&values(r.saturating_mul(cols), i));
        ws.push(gpu.upload_matrix(*r, cols, &bits)?);
    }
    let zeros = |gpu: &Gpu| -> Result<Vec<Buffer>, CudaError> {
        rows.iter()
            .map(|r| gpu.zeros(tokens.saturating_mul(*r)))
            .collect()
    };
    let mut separate = zeros(gpu)?;
    for (w, out) in ws.iter().zip(&mut separate) {
        gpu.matmul(w, &x, tokens, out)?;
    }
    let mut grouped = zeros(gpu)?;
    match (&ws[..], &mut grouped[..]) {
        ([w0, w1], [y0, y1]) => gpu.matmul_pair((w0, w1), &x, tokens, (y0, y1))?,
        ([w0, w1, w2], [y0, y1, y2]) => {
            gpu.matmul_triple((w0, w1, w2), &x, tokens, (y0, y1, y2))?
        }
        _ => {
            return Err(CudaError::Shape {
                what: "grouped_case matrices",
                expected: 3,
                actual: rows.len(),
            });
        }
    }
    for (a, b) in grouped.iter().zip(&separate) {
        assert_eq!(
            gpu.download(a)?,
            gpu.download(b)?,
            "rows {rows:?}, cols {cols}, tokens {tokens}"
        );
    }
    Ok(())
}

#[test]
fn grouped_matmuls_give_the_same_bits_as_separate_ones() {
    let Some(gpu) = gpu().unwrap() else { return };
    // row counts that do not fill a 4-row block, so one block spans two matrices; odd columns take the scalar path;
    // 16 tokens and up leave the one-launch kernel
    for tokens in [1, 3, 8, 9, 15, 16, 40] {
        for cols in [64, 13, 2048] {
            grouped_case(&gpu, &[5, 7], cols, tokens).unwrap();
            grouped_case(&gpu, &[6, 1, 9], cols, tokens).unwrap();
        }
    }
    grouped_case(&gpu, &[128, 64, 64], 128, 1).unwrap();
    grouped_case(&gpu, &[96, 96], 64, 2).unwrap();
}

#[test]
fn grouped_matmuls_reject_mismatched_shapes_instead_of_launching() {
    let Some(gpu) = gpu().unwrap() else { return };
    let w = |rows, cols| {
        gpu.upload_matrix(rows, cols, &vec![0; rows * cols])
            .unwrap()
    };
    let x = gpu.zeros(8).unwrap();
    let (mut a, mut b) = (gpu.zeros(2).unwrap(), gpu.zeros(3).unwrap());
    assert!(
        gpu.matmul_pair((&w(2, 8), &w(3, 4)), &x, 1, (&mut a, &mut b))
            .is_err(),
        "columns differ"
    );
    assert!(
        gpu.matmul_pair((&w(2, 8), &w(4, 8)), &x, 1, (&mut a, &mut b))
            .is_err(),
        "output too short"
    );
    assert!(
        gpu.matmul_pair(
            (&w(2, 8), &w(3, 8)),
            &gpu.zeros(7).unwrap(),
            1,
            (&mut a, &mut b)
        )
        .is_err(),
        "input too short"
    );
    gpu.matmul_pair((&w(2, 8), &w(3, 8)), &x, 1, (&mut a, &mut b))
        .unwrap();
}

#[test]
fn matmul_rejects_wrongly_sized_buffers_instead_of_launching() {
    let Some(gpu) = gpu().unwrap() else { return };
    let w = gpu.upload_matrix(2, 8, &[0; 16]).unwrap();
    let x = gpu.upload(&[0.0; 8]).unwrap();
    let mut small = gpu.zeros(1).unwrap();
    assert!(gpu.matmul(&w, &x, 1, &mut small).is_err());
    let mut out = gpu.zeros(2).unwrap();
    assert!(gpu.matmul(&w, &x, 2, &mut out).is_err());
}

#[test]
fn rms_norm_matches_the_cpu_including_dims_that_are_not_a_warp_multiple() {
    let Some(gpu) = gpu().unwrap() else { return };
    for dim in [64, 17, 128, 2048] {
        let x = values(3 * dim, 3);
        let weight = values(dim, 4);
        let mut expected = x.clone();
        tensor::rms_norm_rows(&mut expected, &weight, 1e-6).unwrap();
        let mut device = gpu.upload(&x).unwrap();
        gpu.rms_norm(&mut device, &gpu.upload(&weight).unwrap(), 1e-6)
            .unwrap();
        assert_close(&gpu.download(&device).unwrap(), &expected, 1e-5);
    }
    let mut ragged = gpu.zeros(10).unwrap();
    assert!(
        gpu.rms_norm(&mut ragged, &gpu.zeros(4).unwrap(), 1e-6)
            .is_err()
    );
}

#[test]
fn rms_norm_into_matches_the_in_place_norm_and_leaves_its_input_alone() {
    let Some(gpu) = gpu().unwrap() else { return };
    for dim in [17, 128, 2048] {
        let x = values(3 * dim, 3);
        let weight = gpu.upload(&values(dim, 4)).unwrap();
        let mut in_place = gpu.upload(&x).unwrap();
        gpu.rms_norm(&mut in_place, &weight, 1e-6).unwrap();
        let input = gpu.upload(&x).unwrap();
        let mut out = gpu.zeros(x.len()).unwrap();
        gpu.rms_norm_into(&input, &weight, &mut out, 1e-6).unwrap();
        assert_eq!(
            gpu.download(&out).unwrap(),
            gpu.download(&in_place).unwrap()
        );
        assert_eq!(gpu.download(&input).unwrap(), x);
    }
    let weight = gpu.zeros(4).unwrap();
    let mut short = gpu.zeros(4).unwrap();
    assert!(
        gpu.rms_norm_into(&gpu.zeros(8).unwrap(), &weight, &mut short, 1e-6)
            .is_err()
    );
}

#[test]
fn rope_matches_the_cpu() {
    let Some(gpu) = gpu().unwrap() else { return };
    let (heads, head_dim, tokens, start) = (3, 16, 4, 5);
    let x = values(tokens * heads * head_dim, 5);
    let mut expected = x.clone();
    tensor::rope_tokens(&mut expected, heads * head_dim, head_dim, start, 10_000.0).unwrap();
    let mut device = gpu.upload(&x).unwrap();
    gpu.rope(&mut device, heads, head_dim, start, 10_000.0)
        .unwrap();
    assert_close(&gpu.download(&device).unwrap(), &expected, 1e-4);
    assert!(gpu.rope(&mut device, heads, 15, start, 10_000.0).is_err());
}

#[test]
fn silu_mul_add_and_embed_match_the_cpu() {
    let Some(gpu) = gpu().unwrap() else { return };
    let (a, b) = (values(1000, 11), values(1000, 12));

    let mut expected = a.clone();
    tensor::silu_mul(&mut expected, &b).unwrap();
    let mut gate = gpu.upload(&a).unwrap();
    gpu.silu_mul(&mut gate, &gpu.upload(&b).unwrap()).unwrap();
    assert_close(&gpu.download(&gate).unwrap(), &expected, 1e-5);

    let mut expected = a.clone();
    tensor::add_in_place(&mut expected, &b).unwrap();
    let mut sum = gpu.upload(&a).unwrap();
    gpu.add(&mut sum, &gpu.upload(&b).unwrap()).unwrap();
    assert_close(&gpu.download(&sum).unwrap(), &expected, 1e-6);
    assert!(gpu.add(&mut sum, &gpu.zeros(3).unwrap()).is_err());

    let table_values = values(10 * 16, 13);
    let bits = bf16_bits(&table_values);
    let table = gpu.upload_matrix(10, 16, &bits).unwrap();
    let mut out = gpu.zeros(3 * 16).unwrap();
    let mut ids = gpu.tokens(4).unwrap();
    gpu.embed(&table, &[4, 0, 9], &mut ids, &mut out).unwrap();
    let cpu = Matrix::new(10, 16, bits).unwrap();
    let mut row = vec![0.0; 16];
    let mut expected = Vec::new();
    for id in [4, 0, 9] {
        cpu.row_f32(id, &mut row).unwrap();
        expected.extend_from_slice(&row);
    }
    assert_eq!(gpu.download(&out).unwrap(), expected);
    // the same scratch serves a shorter pass next, and an id or a count it cannot hold is refused
    let mut one = gpu.zeros(16).unwrap();
    gpu.embed(&table, &[9], &mut ids, &mut one).unwrap();
    assert_eq!(gpu.download(&one).unwrap(), expected[32..]);
    assert!(gpu.embed(&table, &[10], &mut ids, &mut one).is_err());
    let mut five = gpu.zeros(5 * 16).unwrap();
    assert!(gpu.embed(&table, &[0; 5], &mut ids, &mut five).is_err());
}

#[test]
fn reports_the_device() {
    let Some(gpu) = gpu().unwrap() else { return };
    let info = gpu.info().unwrap();
    assert!(!info.name.is_empty() && info.total_bytes >= info.free_bytes);
    eprintln!(
        "{} (sm_{}{}), {} MB free",
        info.name,
        info.compute_capability.0,
        info.compute_capability.1,
        info.free_bytes / 1_000_000
    );
}
