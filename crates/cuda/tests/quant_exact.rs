//! The exact int8 prompt matmul against a plain f64 product of the dequantized weights, for every group size it
//! handles in one kernel, rows and tokens that are not whole tiles, empty groups, tiny values and large outliers.
//! Without a GPU this test skips; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod common;

use common::{assert_close, gpu, values};
use tensor::{QuantBits, QuantScheme, quantize};

#[test]
fn the_exact_int8_matmul_matches_f64_for_every_group_and_tile_edge()
-> Result<(), Box<dyn std::error::Error>> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for group in [32, 64, 128] {
        let scheme = QuantScheme {
            bits: QuantBits::Int8,
            group,
        };
        for (rows, cols, tokens) in [
            (300, 256, 17),
            (64, 128, 31),
            (65, 384, 33),
            (33, 256, 70),
            (1, 128, 16),
        ] {
            let weights = values(rows * cols, 11);
            let bits: Vec<u16> = weights.iter().map(|v| (v.to_bits() >> 16) as u16).collect();
            let q = quantize(&bits, rows, cols, scheme)?;
            let mut x = values(tokens * cols, 13);
            for (i, v) in x.iter_mut().enumerate() {
                match i % 211 {
                    0 => *v *= 10.0,
                    1 => *v *= 1e-20,
                    _ => {}
                }
            }
            x[..cols].fill(0.0);
            x[cols..cols + group].fill(0.0);
            let mut expected = vec![0.0_f32; tokens * rows];
            let mut row = vec![0.0; cols];
            for r in 0..rows {
                q.dequantize_row(r, &mut row)?;
                for t in 0..tokens {
                    let dot: f64 = row
                        .iter()
                        .zip(&x[t * cols..(t + 1) * cols])
                        .map(|(w, v)| f64::from(*w) * f64::from(*v))
                        .sum();
                    expected[t * rows + r] = dot as f32;
                }
            }
            let device = gpu.upload_quantized(&q)?;
            let mut out = gpu.zeros(tokens * rows)?;
            gpu.matmul_quant_precise(&device, &gpu.upload(&x)?, tokens, &mut out)?;
            assert_close(&gpu.download(&out)?, &expected, 2e-5);
        }
    }
    Ok(())
}
