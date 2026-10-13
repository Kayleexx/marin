use tensor::{Matrix, QuantBits, QuantScheme, Quantized, quantize};

#[test]
fn a_consumed_packed_matrix_matches_on_load_quantization() {
    let scheme = QuantScheme::new(QuantBits::Int8, 32).unwrap();
    let bits: Vec<_> = (0..96 * 7)
        .map(|i| (((i % 31) as f32 - 15.0).to_bits() >> 16) as u16)
        .collect();
    let packed = Matrix::from_int8(quantize(&bits, 7, 96, scheme).unwrap()).unwrap();
    let reference = Matrix::quantized(7, 96, bits, scheme).unwrap();
    for n in [1, 3, 65] {
        let input = vec![0.75; n * 96];
        let mut a = vec![0.0; n * 7];
        let mut b = a.clone();
        reference.matmul(&input, &mut a).unwrap();
        packed.matmul(&input, &mut b).unwrap();
        assert_eq!(a, b);
    }
}

#[test]
fn packed_construction_validates_independently_of_the_exporter() {
    for case in 0..8 {
        let mut q = Quantized {
            rows: 1,
            cols: 32,
            scheme: QuantScheme {
                bits: QuantBits::Int8,
                group: 32,
            },
            codes: vec![0; 32],
            scales: vec![0.0],
        };
        match case {
            0 => q.scheme.group = 0,
            1 => q.cols = 31,
            2 => q.codes.pop().map(|_| ()).unwrap(),
            3 => q.scales.clear(),
            4 => q.codes[0] = 128,
            5 => q.scales[0] = f32::NAN,
            6 => q.scales[0] = -1.0,
            _ => q.rows = usize::MAX,
        }
        assert!(Matrix::from_int8(q).is_err(), "accepted case {case}");
    }
}
