//! The Delta recurrence kernel against a plain CPU loop, for every key dimension class it is built for, and
//! against itself when the tokens are split across calls.
//! Without a GPU this test skips; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod common;

use common::{assert_close, gpu, values};
use cuda::{DeltaDims, DeltaInputs};

const EPS: f32 = 1e-6;

/// Deterministic inputs for `tokens` steps, plus a starting state.
struct Case {
    dims: DeltaDims,
    inputs: [Vec<f32>; 7],
    state: Vec<f32>,
}

#[test]
fn the_recurrence_kernel_matches_the_cpu_and_does_not_depend_on_how_tokens_are_split() {
    let Some(gpu) = gpu().unwrap() else { return };

    let case = |(key_heads, value_heads): (usize, usize),
                (key_dim, value_dim): (usize, usize),
                tokens: usize| {
        let dims = DeltaDims {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            kernel: 4,
        };
        let channels = 2 * key_heads * key_dim + value_heads * value_dim;
        let seed = (key_dim * 31 + value_dim) as u64;
        let scaled = |n: usize, s: u64, mul: f32, add: f32| -> Vec<f32> {
            values(n, seed + s).iter().map(|v| add + v * mul).collect()
        };
        Case {
            dims,
            inputs: [
                values(tokens * channels, seed),
                values(tokens * value_heads * value_dim, seed + 1),
                values(tokens * value_heads, seed + 2),
                values(tokens * value_heads, seed + 3),
                scaled(value_heads, 4, 0.5, 0.0),
                values(value_heads, seed + 5),
                scaled(value_dim, 6, 0.25, 1.0),
            ],
            state: scaled(value_heads * key_dim * value_dim, 7, 0.1, 0.0),
        }
    };

    // The kernel over the tokens in `range`, starting from `state`; returns (core, state).
    let run = |c: &Case, range: std::ops::Range<usize>, state: &[f32]| {
        let DeltaDims {
            key_heads,
            value_heads,
            key_dim,
            value_dim,
            ..
        } = c.dims;
        let widths = [
            2 * key_heads * key_dim + value_heads * value_dim,
            value_heads * value_dim,
            value_heads,
            value_heads,
        ];
        let [mixed, z, a, b] = [0, 1, 2, 3].map(|i| {
            gpu.upload(&c.inputs[i][range.start * widths[i]..range.end * widths[i]])
                .unwrap()
        });
        let [a_log, dt_bias, norm] = [4, 5, 6].map(|i| gpu.upload(&c.inputs[i]).unwrap());
        let mut state = gpu.upload(state).unwrap();
        let mut core = gpu.zeros(range.len() * widths[1]).unwrap();
        let input = DeltaInputs {
            mixed: &mixed,
            z: &z,
            a: &a,
            b: &b,
            a_log: &a_log,
            dt_bias: &dt_bias,
            norm: &norm,
        };
        gpu.q35_recurrent(input, &mut state, &mut core, c.dims, range.len(), EPS)
            .unwrap();
        (gpu.download(&core).unwrap(), gpu.download(&state).unwrap())
    };

    // The recurrence token by token in the order the kernel uses; returns (core, state).
    let reference = |c: &Case, tokens: usize| {
        let DeltaDims {
            key_heads,
            value_heads,
            key_dim: kd,
            value_dim: vd,
            ..
        } = c.dims;
        let [mixed, z, a, b, a_log, dt_bias, norm] = &c.inputs;
        let (kd_total, vd_total) = (key_heads * kd, value_heads * vd);
        let channels = 2 * kd_total + vd_total;
        let mut state = c.state.clone();
        let mut core = vec![0.0; tokens * vd_total];
        for t in 0..tokens {
            let src = &mixed[t * channels..(t + 1) * channels];
            for h in 0..value_heads {
                let kh = h / (value_heads / key_heads);
                let mut q = src[kh * kd..(kh + 1) * kd].to_vec();
                let mut k = src[kd_total + kh * kd..kd_total + (kh + 1) * kd].to_vec();
                let qi = (q.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip()
                    * (kd as f32).sqrt().recip();
                let ki = (k.iter().map(|x| x * x).sum::<f32>() + 1e-6).sqrt().recip();
                q.iter_mut().for_each(|x| *x *= qi);
                k.iter_mut().for_each(|x| *x *= ki);
                let av = a[t * value_heads + h] + dt_bias[h];
                let soft = if av > 20.0 { av } else { av.exp().ln_1p() };
                let decay = (-a_log[h].exp() * soft).exp();
                let beta = 1.0 / (1.0 + (-b[t * value_heads + h]).exp());
                let s = &mut state[h * kd * vd..(h + 1) * kd * vd];
                let mut out = vec![0.0_f32; vd];
                for (v, value) in out.iter_mut().enumerate() {
                    let mut pred = 0.0;
                    for i in 0..kd {
                        s[i * vd + v] *= decay;
                        pred += s[i * vd + v] * k[i];
                    }
                    let delta = (src[2 * kd_total + h * vd + v] - pred) * beta;
                    for i in 0..kd {
                        s[i * vd + v] += k[i] * delta;
                        *value += s[i * vd + v] * q[i];
                    }
                }
                let total: f32 = out.iter().map(|x| x * x).sum();
                let scale = (total / vd as f32 + EPS).sqrt().recip();
                for v in 0..vd {
                    let zv = z[t * vd_total + h * vd + v];
                    core[t * vd_total + h * vd + v] =
                        out[v] * scale * norm[v] * zv / (1.0 + (-zv).exp());
                }
            }
        }
        (core, state)
    };

    for (heads, dim, tokens) in [
        ((1, 1), (1, 1), 3),
        ((2, 4), (8, 16), 5),
        ((2, 4), (64, 32), 4),
        ((2, 4), (65, 48), 3),
        ((2, 4), (96, 64), 4),
        ((2, 4), (128, 128), 6),
        ((1, 2), (129, 100), 3),
        ((2, 4), (200, 128), 3),
        ((1, 2), (256, 128), 2),
        ((2, 4), (128, 33), 2),
    ] {
        let c = case(heads, dim, tokens);
        let (core, state) = run(&c, 0..tokens, &c.state);
        let (want_core, want_state) = reference(&c, tokens);
        assert_close(&core, &want_core, 2e-4);
        assert_close(&state, &want_state, 2e-4);
    }

    for (dim, tokens) in [((128, 128), 9), ((65, 40), 7), ((256, 64), 4)] {
        let c = case((2, 4), dim, tokens);
        let (whole_core, whole_state) = run(&c, 0..tokens, &c.state);
        let mut state = c.state.clone();
        let mut core = Vec::new();
        for t in 0..tokens {
            let (part, next) = run(&c, t..t + 1, &state);
            core.extend(part);
            state = next;
        }
        assert_eq!(core, whole_core, "core, key_dim {}", dim.0);
        assert_eq!(state, whole_state, "state, key_dim {}", dim.0);
    }
}
