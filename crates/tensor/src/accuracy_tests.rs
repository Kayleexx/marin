//! Every kernel result against an f64 reference, over shapes with row, column, token and k-block remainders.

use super::*;

struct Lcg(u64);

impl Lcg {
    fn next(&mut self) -> f32 {
        self.0 = self
            .0
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (self.0 >> 40) as f32 / 8_388_608.0 - 1.0
    }
}

const ROWS: [usize; 6] = [1, 3, 4, 5, 7, 13];
const COLS: [usize; 11] = [1, 2, 3, 15, 16, 17, 31, 33, 257, 513, 1001];
const TOKENS: [usize; 8] = [1, 2, 3, 5, 64, 65, 66, 129];

fn bf16_bits(value: f32) -> u16 {
    (value.to_bits() >> 16) as u16
}

/// Largest `|got - exact| / sum(|w * x|)` over the whole output.
fn worst_error(matrix: &Matrix, bits: &[u16], x: &[f32], out: &[f32]) -> f64 {
    let mut worst = 0.0_f64;
    let tokens = x
        .chunks_exact(matrix.cols())
        .zip(out.chunks_exact(matrix.rows()));
    for (token, token_out) in tokens {
        for (row, got) in bits.chunks_exact(matrix.cols()).zip(token_out) {
            let products = row
                .iter()
                .zip(token)
                .map(|(w, v)| f64::from(bf16_to_f32(*w)) * f64::from(*v));
            let exact: f64 = products.clone().sum();
            let scale: f64 = products.map(f64::abs).sum::<f64>().max(1e-30);
            worst = worst.max((f64::from(*got) - exact).abs() / scale);
        }
    }
    worst
}

#[test]
fn matmul_matches_an_f64_reference_on_every_path() {
    let mut rng = Lcg(7);
    for rows in ROWS {
        for cols in COLS {
            let bits: Vec<u16> = (0..rows * cols).map(|_| bf16_bits(rng.next())).collect();
            let matrix = Matrix::new(rows, cols, bits.clone()).unwrap();
            for tokens in TOKENS {
                let x: Vec<f32> = (0..tokens * cols).map(|_| rng.next()).collect();
                let mut dispatched = vec![f32::NAN; tokens * rows];
                matrix.matmul(&x, &mut dispatched).unwrap();
                let mut fallback = vec![f32::NAN; tokens * rows];
                let mut slices: Vec<&mut [f32]> = fallback.chunks_mut(matrix.rows()).collect();
                multiply_scalar(matrix.words().unwrap(), &x, &mut slices, matrix.cols());
                for (name, out) in [("dispatch", &dispatched), ("scalar", &fallback)] {
                    let error = worst_error(&matrix, &bits, &x, out);
                    assert!(
                        error < 1e-5,
                        "{name} {rows}x{cols}, {tokens} tokens: relative error {error:e}"
                    );
                }
            }
        }
    }
}

#[test]
fn row_f32_returns_every_element_of_an_odd_row() {
    let bits: Vec<u16> = [1.0, 2.0, 3.0, 4.0, 5.0, 6.0]
        .iter()
        .map(|v| bf16_bits(*v))
        .collect();
    let matrix = Matrix::new(2, 3, bits).unwrap();
    let mut row = [0.0; 3];
    matrix.row_f32(1, &mut row).unwrap();
    assert_eq!(row, [4.0, 5.0, 6.0]);
}

#[test]
fn empty_operands_give_zeros() {
    let matrix = Matrix::new(3, 0, Vec::new()).unwrap();
    let mut out = [1.0; 0];
    matrix.matmul(&[], &mut out).unwrap();
    let matrix = Matrix::new(2, 2, vec![0; 4]).unwrap();
    matrix.matmul(&[], &mut []).unwrap();
}

#[test]
fn packing_in_several_blocks_keeps_every_row_in_place() {
    let (rows, cols) = (5000, 1001);
    let bits: Vec<u16> = (0..rows * cols)
        .map(|i| bf16_bits((i % 251) as f32 + (i / cols) as f32))
        .collect();
    let matrix = Matrix::new(rows, cols, bits.clone()).unwrap();
    let mut row = vec![0.0; cols];
    for r in [0, 1, 2047, 2048, 4998, 4999] {
        matrix.row_f32(r, &mut row).unwrap();
        let expected: Vec<f32> = bits[r * cols..(r + 1) * cols]
            .iter()
            .map(|b| bf16_to_f32(*b))
            .collect();
        assert_eq!(row, expected, "row {r}");
    }
}

#[test]
fn even_rows_keep_the_loaders_buffer_and_odd_rows_are_repacked() {
    let even = Matrix::new(2, 4, vec![0; 8]).unwrap();
    assert!(matches!(even.data, Storage::Bits(_)));
    let odd = Matrix::new(2, 3, vec![0; 6]).unwrap();
    assert!(matches!(odd.data, Storage::Pairs(_)));
}

#[test]
fn matmul_is_identical_for_every_thread_count() {
    let mut rng = Lcg(21);
    for rows in [1, 3, 4, 5, 7, 13, 64, 100] {
        for cols in [1, 2, 3, 33, 257] {
            let bits: Vec<u16> = (0..rows * cols).map(|_| bf16_bits(rng.next())).collect();
            let matrix = Matrix::new(rows, cols, bits).unwrap();
            for tokens in [2, 3, 5, 66] {
                let x: Vec<f32> = (0..tokens * cols).map(|_| rng.next()).collect();
                let run = |threads| {
                    crate::set_threads(threads);
                    let mut out = vec![f32::NAN; tokens * rows];
                    matrix.matmul(&x, &mut out).unwrap();
                    out
                };
                let single = run(1);
                for threads in [2, 3, 4, 8, 16] {
                    assert_eq!(
                        run(threads),
                        single,
                        "{rows}x{cols}, {tokens} tokens, {threads} threads"
                    );
                }
                crate::set_threads(0);
            }
        }
    }
}
