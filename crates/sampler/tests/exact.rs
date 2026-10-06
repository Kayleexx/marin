//! The sampler must pick exactly the tokens the straightforward algorithm picks: rank every token, cut to top-k, keep
//! the smallest set reaching top-p, renormalize, then walk the cumulative probabilities with the seeded draw. The
//! reference below is that algorithm written the plain way, with a full sort.

use sampler::{Sampler, SamplerConfig};

fn reference_distribution(
    config: &SamplerConfig,
    logits: &[f32],
    usable: usize,
) -> Vec<(u32, f32)> {
    let candidates = logits.get(..usable).unwrap_or_default();
    let mut ranked: Vec<(u32, f32)> = (0..).zip(candidates.iter().copied()).collect();
    ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
    if config.top_k > 0 {
        ranked.truncate(config.top_k);
    }
    let top = ranked.first().map_or(0.0, |(_, l)| *l);
    let weights: Vec<(u32, f32)> = ranked
        .iter()
        .map(|(id, l)| (*id, ((l - top) / config.temperature).exp()))
        .collect();
    let total: f32 = weights.iter().map(|(_, w)| w).sum();
    let mut kept = Vec::new();
    let mut mass = 0.0;
    for (id, w) in weights {
        let p = w / total;
        kept.push((id, p));
        mass += p;
        if mass >= config.top_p {
            break;
        }
    }
    let kept_mass: f32 = kept.iter().map(|(_, p)| p).sum();
    kept.iter().map(|(id, p)| (*id, p / kept_mass)).collect()
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        self.0
    }

    fn unit(&mut self) -> f32 {
        (self.next() >> 40) as f32 / 16_777_216.0
    }
}

/// Logit vectors of different shapes: smooth, peaked like a language model's, coarse (many exact ties), flat, with
/// repeated maxima, and tiny.
fn vectors(rng: &mut Rng) -> Vec<Vec<f32>> {
    let mut out = Vec::new();
    for n in [1_usize, 2, 3, 7, 50, 400, 3000, 20_000] {
        out.push((0..n).map(|_| (rng.unit() - 0.5) * 12.0).collect());
        out.push(
            (0..n)
                .map(|i| {
                    if i % 97 == 3 {
                        15.0 + rng.unit()
                    } else {
                        rng.unit() * 6.0 - 9.0
                    }
                })
                .collect(),
        );
        out.push(
            (0..n)
                .map(|_| ((rng.unit() * 16.0).floor() - 8.0) * 0.5)
                .collect(),
        );
        out.push(vec![1.25; n]);
        let mut repeated: Vec<f32> = (0..n).map(|_| rng.unit() * 4.0 - 2.0).collect();
        for slot in repeated.iter_mut().step_by(5) {
            *slot = 3.0;
        }
        out.push(repeated);
        out.push(
            (0..n)
                .map(|i| if i % 2 == 0 { 0.0 } else { -0.0 })
                .collect(),
        );
    }
    out
}

const TEMPERATURES: [f32; 7] = [0.01, 0.3, 0.7, 1.0, 1.5, 5.0, 100.0];
const TOP_KS: [usize; 9] = [0, 1, 2, 20, 127, 128, 129, 1000, 1_000_000];
const TOP_PS: [f32; 8] = [0.05, 0.5, 0.8, 0.9, 0.99, 0.999_999, 0.999_999_94, 1.0];

#[test]
fn the_distribution_is_bit_for_bit_the_straightforward_one() {
    let mut rng = Rng(0x1234_5678_9abc_def1);
    for logits in vectors(&mut rng) {
        for usable in [logits.len(), logits.len().div_ceil(2).max(1)] {
            for temperature in TEMPERATURES {
                for top_k in TOP_KS {
                    for top_p in TOP_PS {
                        let config = SamplerConfig {
                            temperature,
                            top_k,
                            top_p,
                            seed: 1,
                        };
                        let expected = reference_distribution(&config, &logits, usable);
                        let actual = Sampler::new(config)
                            .unwrap()
                            .distribution(&logits, usable)
                            .unwrap();
                        assert_eq!(
                            actual,
                            expected,
                            "n {} usable {usable} t {temperature} k {top_k} p {top_p}",
                            logits.len()
                        );
                    }
                }
            }
        }
    }
}

#[test]
fn seeded_sampling_picks_the_same_tokens_in_the_same_order() {
    let mut rng = Rng(77);
    let mut vectors = vectors(&mut rng);
    vectors.retain(|v| v.len() >= 50);
    for (case, logits) in vectors.iter().enumerate() {
        for (temperature, top_k, top_p) in [
            (0.7_f32, 20_usize, 0.8_f32),
            (0.7, 0, 0.9),
            (1.0, 0, 0.95),
            (0.4, 128, 1.0),
        ] {
            let config = SamplerConfig {
                temperature,
                top_k,
                top_p,
                seed: 9 + case as u64,
            };
            let mut sampler = Sampler::new(config).unwrap();
            // The draws are splitmix64 over the seed, as documented in the sampler.
            let mut state = config.seed;
            for _ in 0..60 {
                state = state.wrapping_add(0x9E37_79B9_7F4A_7C15);
                let mut z = state;
                z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
                z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
                z ^= z >> 31;
                let draw = (z >> 40) as f32 / 16_777_216.0;
                let probs = reference_distribution(&config, logits, logits.len());
                let mut cumulative = 0.0;
                let mut expected = probs.last().map(|(id, _)| *id).unwrap();
                for (id, p) in &probs {
                    cumulative += p;
                    if draw < cumulative {
                        expected = *id;
                        break;
                    }
                }
                assert_eq!(
                    sampler.sample(logits, logits.len()).unwrap(),
                    expected,
                    "case {case} {config:?}"
                );
            }
        }
    }
}

#[test]
fn greedy_is_untouched_and_the_lowest_id_wins_ties() {
    let logits = [0.5_f32, 4.0, 4.0, 1.0, 4.0];
    for top_k in [0, 1, 3] {
        let mut sampler = Sampler::new(SamplerConfig {
            temperature: 0.0,
            top_k,
            top_p: 0.5,
            seed: 3,
        })
        .unwrap();
        assert_eq!(sampler.sample(&logits, 5).unwrap(), 1);
        assert_eq!(sampler.distribution(&logits, 5).unwrap(), [(1, 1.0)]);
    }
}

#[test]
fn bad_logits_and_empty_candidate_sets_are_still_errors() {
    let sampler = Sampler::new(SamplerConfig {
        temperature: 0.7,
        top_k: 20,
        top_p: 0.8,
        seed: 1,
    })
    .unwrap();
    assert!(sampler.distribution(&[1.0, f32::NAN], 2).is_err());
    assert!(sampler.distribution(&[1.0, f32::INFINITY], 2).is_err());
    assert!(sampler.distribution(&[1.0, 2.0], 0).is_err());
    assert!(sampler.distribution(&[1.0, 2.0], 3).is_err());
}
