use tensor::{Matrix, QuantBits, QuantScheme, bf16_to_f32};

#[test]
fn int8_products_match_independently_rounded_f64_reference() {
    for group in [32, 64, 96, 128, 256, 512] {
        for rows in [1, 3, 4, 7] {
            let cols = group * if group == 96 { 9 } else { 3 };
            let bits: Vec<_> = (0..rows * cols)
                .map(|i| {
                    let value = if i < group {
                        0.0
                    } else {
                        ((i * 17 % 255) as f32 - 127.0) / 31.0
                    };
                    (value.to_bits() >> 16) as u16
                })
                .collect();
            let scheme = QuantScheme::new(QuantBits::Int8, group).unwrap();
            let matrix = Matrix::quantized(rows, cols, bits.clone(), scheme).unwrap();
            let reference: Vec<f32> = bits
                .chunks_exact(group)
                .flat_map(|block| {
                    let max = block
                        .iter()
                        .map(|b| bf16_to_f32(*b).abs())
                        .fold(0.0, f32::max);
                    let scale = max / 127.0;
                    block.iter().map(move |b| {
                        if scale == 0.0 {
                            0.0
                        } else {
                            (bf16_to_f32(*b) / scale).round().clamp(-127.0, 127.0) * scale
                        }
                    })
                })
                .collect();
            for r in 0..rows {
                let mut got = vec![0.0; cols];
                matrix.row_f32(r, &mut got).unwrap();
                assert_eq!(got, reference[r * cols..(r + 1) * cols]);
            }
            for tokens in [1, 2, 3, 63, 64, 65] {
                let input: Vec<_> = (0..tokens * cols)
                    .map(|i| ((i * 13 % 101) as f32 - 50.0) / 17.0)
                    .collect();
                let mut got = vec![f32::NAN; tokens * rows];
                matrix.matmul(&input, &mut got).unwrap();
                for (t, x) in input.chunks_exact(cols).enumerate() {
                    for (r, w) in reference.chunks_exact(cols).enumerate() {
                        let sum: f64 = w
                            .iter()
                            .zip(x)
                            .map(|(w, x)| f64::from(*w) * f64::from(*x))
                            .sum();
                        let magnitude: f64 = w
                            .iter()
                            .zip(x)
                            .map(|(w, x)| (f64::from(*w) * f64::from(*x)).abs())
                            .sum();
                        assert!(
                            (f64::from(got[t * rows + r]) - sum).abs() <= 1e-5 * magnitude.max(1.0),
                            "g={group} r={r} t={t}: {} vs {sum}",
                            got[t * rows + r]
                        );
                    }
                }
            }
            assert!(matrix.row_f32(rows, &mut vec![0.0; cols]).is_err());
            assert!(
                matrix
                    .matmul(&vec![0.0; cols + 1], &mut vec![0.0; rows])
                    .is_err()
            );
            assert!(
                matrix
                    .matmul(&vec![0.0; cols], &mut vec![0.0; rows + 1])
                    .is_err()
            );
        }
    }
}

#[test]
fn cpu_quantization_rejects_invalid_storage_and_nonfinite_weights() {
    for (rows, cols, data, scheme) in [
        (
            1,
            32,
            vec![0; 31],
            QuantScheme {
                bits: QuantBits::Int8,
                group: 32,
            },
        ),
        (
            1,
            32,
            vec![0; 32],
            QuantScheme {
                bits: QuantBits::Int8,
                group: 0,
            },
        ),
        (
            1,
            32,
            vec![0; 32],
            QuantScheme {
                bits: QuantBits::Int8,
                group: 48,
            },
        ),
        (
            1,
            32,
            vec![0; 32],
            QuantScheme {
                bits: QuantBits::Int4,
                group: 32,
            },
        ),
        (
            1,
            32,
            vec![0x7f80; 32],
            QuantScheme {
                bits: QuantBits::Int8,
                group: 32,
            },
        ),
        (
            1,
            32,
            vec![0x7fc0; 32],
            QuantScheme {
                bits: QuantBits::Int8,
                group: 32,
            },
        ),
        (
            usize::MAX,
            32,
            vec![],
            QuantScheme {
                bits: QuantBits::Int8,
                group: 32,
            },
        ),
    ] {
        assert!(Matrix::quantized(rows, cols, data, scheme).is_err());
    }
}
