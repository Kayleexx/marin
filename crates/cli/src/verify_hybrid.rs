/// Section 34/35 gates apply to every value, including low-ranked logits.
pub(crate) fn compare(actual: &[f32], expected: &[f32]) -> Result<(f64, f64, f64), String> {
    if actual.len() != expected.len()
        || expected.is_empty()
        || !actual.iter().chain(expected).all(|x| x.is_finite())
    {
        return Err("missing, nonfinite or incompatible full tensor reference".into());
    }
    let max_ref = expected
        .iter()
        .map(|v| f64::from(*v).abs())
        .fold(1.0, f64::max);
    let mut max_error = 0.0_f64;
    let mut error = 0.0_f64;
    let mut magnitude = 0.0_f64;
    for (a, b) in actual.iter().zip(expected) {
        let d = f64::from(*a) - f64::from(*b);
        max_error = max_error.max(d.abs());
        error += d * d;
        magnitude += f64::from(*b).powi(2);
    }
    let rms =
        (error / actual.len() as f64).sqrt() / (magnitude / actual.len() as f64).sqrt().max(1e-6);
    let normalized = max_error / max_ref;
    if normalized > 1e-4 || rms > 1e-5 {
        return Err(format!(
            "normalized maximum {normalized:.3e}, relative RMS {rms:.3e} exceed fixed 1e-4 / 1e-5 gates"
        ));
    }
    Ok((max_error, normalized, rms))
}
#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn full_tensor_gate_checks_rms_small_values_and_shapes() {
        assert!(compare(&[1.0, 2.0], &[1.0, 2.0]).is_ok());
        assert!(compare(&[1.0, 2.001], &[1.0, 2.0]).is_err());
        assert!(compare(&[1e-6], &[1.1e-6]).is_err());
        assert!(compare(&[], &[]).is_err());
        assert!(compare(&[f32::NAN], &[0.0]).is_err());
    }
}
