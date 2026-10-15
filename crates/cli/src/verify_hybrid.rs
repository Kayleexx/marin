pub(crate) fn l2(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt()
}

pub(crate) fn argmax(logits: &[f32]) -> Option<u32> {
    let (index, _) = logits
        .iter()
        .enumerate()
        .reduce(|best, next| if next.1 > best.1 { next } else { best })?;
    u32::try_from(index).ok()
}

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

type GraniteTensors<'a> = (&'a [Vec<f32>], &'a [Vec<usize>], &'a [Vec<f32>]);

pub(crate) fn compare_granite(
    actual: GraniteTensors<'_>,
    expected: GraniteTensors<'_>,
    layers: usize,
) -> Result<(f64, f64), String> {
    let (routers, tops, experts) = actual;
    let (reference_routers, reference_tops, reference_experts) = expected;
    if routers.len() != layers
        || tops.len() != layers
        || experts.len() != layers
        || reference_routers.len() != layers
        || reference_tops.len() != layers
        || reference_experts.len() != layers
    {
        return Err("missing full Granite router/expert references".into());
    }
    let mut worst = (0.0_f64, 0.0_f64);
    let pairs = routers
        .iter()
        .zip(reference_routers)
        .zip(experts.iter().zip(reference_experts))
        .zip(tops.iter().zip(reference_tops));
    for (layer, ((router, expert), top)) in pairs.enumerate() {
        for (label, left, right) in [
            ("router", router.0, router.1),
            ("experts", expert.0, expert.1),
        ] {
            let (_, normalized, rms) = compare(left, right)
                .map_err(|e| format!("first incorrect {label} layer {layer}: {e}"))?;
            worst.0 = worst.0.max(normalized);
            worst.1 = worst.1.max(rms);
        }
        if top.0 != top.1 {
            return Err(format!("Granite layer {layer} top-k routing differs"));
        }
    }
    Ok(worst)
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
