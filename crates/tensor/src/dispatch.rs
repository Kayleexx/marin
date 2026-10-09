use std::sync::OnceLock;

use pulp::WithSimd;

/// Runs `op` on the best SIMD level this CPU has (AVX2 with FMA), or hands it back when only the scalar
/// path is left. `MARIN_MATMUL_SCALAR=1` forces the scalar path, to test and time the fallback.
#[cfg(target_arch = "x86_64")]
pub(crate) fn accelerated<Op: WithSimd<Output = ()>>(op: Op) -> Result<(), Op> {
    use pulp::{Arch, Simd};

    match Arch::new() {
        Arch::V3(v3) if !forced_scalar() => {
            Simd::vectorize(v3, op);
            Ok(())
        }
        _ => Err(op),
    }
}

#[cfg(not(target_arch = "x86_64"))]
pub(crate) fn accelerated<Op: WithSimd<Output = ()>>(op: Op) -> Result<(), Op> {
    Err(op)
}

#[cfg(target_arch = "x86_64")]
fn forced_scalar() -> bool {
    static FORCED: OnceLock<bool> = OnceLock::new();
    *FORCED.get_or_init(|| {
        std::env::var("MARIN_MATMUL_SCALAR").is_ok_and(|v| !v.is_empty() && v != "0")
    })
}
