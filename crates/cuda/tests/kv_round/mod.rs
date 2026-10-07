//! What the cache hands back for a value after storing it: the same rounding the store kernel does, on the host.

use cuda::KvFormat;

fn f16(x: f32) -> f32 {
    let magnitude = x.abs();
    let rounded = if magnitude >= 65504.0 {
        65504.0
    } else if magnitude < 6.103_515_6e-5 {
        let step = 2.0_f32.powi(-24);
        (magnitude / step).round_ties_even() * step
    } else {
        let exponent = i32::try_from(magnitude.to_bits().wrapping_shr(23)).unwrap_or(0);
        let step = 2.0_f32.powi(exponent.wrapping_sub(137));
        (magnitude / step).round_ties_even() * step
    };
    rounded.copysign(x)
}

pub fn round_trip(format: KvFormat, values: &[f32]) -> Vec<f32> {
    match format {
        KvFormat::F32 => values.to_vec(),
        KvFormat::F16 => values.iter().map(|x| f16(*x)).collect(),
    }
}
