#![forbid(unsafe_code)]

use thiserror::Error;

#[derive(Debug, Error, PartialEq)]
pub enum SamplerError {
    #[error("temperature must be finite and >= 0, got {0}")]
    Temperature(f32),
    #[error("top_p must be in (0, 1], got {0}")]
    TopP(f32),
    #[error("logits contain NaN or infinity")]
    NonFiniteLogits,
    #[error("cannot sample from {usable} candidates out of {available} logits")]
    NoCandidates { usable: usize, available: usize },
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub struct SamplerConfig {
    /// 0 selects the most likely token every time.
    pub temperature: f32,
    /// Keep only this many most likely tokens; 0 disables the filter.
    pub top_k: usize,
    pub top_p: f32,
    pub seed: u64,
}

impl SamplerConfig {
    pub fn validate(&self) -> Result<(), SamplerError> {
        if !self.temperature.is_finite() || self.temperature < 0.0 {
            return Err(SamplerError::Temperature(self.temperature));
        }
        if !(self.top_p > 0.0 && self.top_p <= 1.0) {
            return Err(SamplerError::TopP(self.top_p));
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct Sampler {
    config: SamplerConfig,
    state: u64,
}

impl Sampler {
    pub fn new(config: SamplerConfig) -> Result<Self, SamplerError> {
        config.validate()?;
        Ok(Self {
            state: config.seed,
            config,
        })
    }

    /// Picks a token among the first `usable` logits (ids beyond that cannot be decoded).
    pub fn sample(&mut self, logits: &[f32], usable: usize) -> Result<u32, SamplerError> {
        let probs = self.distribution(logits, usable)?;
        let draw = self.next_unit();
        let mut cumulative = 0.0;
        for (id, p) in &probs {
            cumulative += p;
            if draw < cumulative {
                return Ok(*id);
            }
        }
        probs
            .last()
            .map(|(id, _)| *id)
            .ok_or(SamplerError::NoCandidates {
                usable,
                available: logits.len(),
            })
    }

    /// The tokens a sample can come from, with their probabilities, most likely first:
    /// temperature, then top-k, then top-p, renormalized. Temperature 0 is the single best token.
    pub fn distribution(
        &self,
        logits: &[f32],
        usable: usize,
    ) -> Result<Vec<(u32, f32)>, SamplerError> {
        let no_candidates = SamplerError::NoCandidates {
            usable,
            available: logits.len(),
        };
        let candidates = match logits.get(..usable) {
            Some(c) if !c.is_empty() => c,
            _ => return Err(no_candidates),
        };
        if candidates.iter().any(|v| !v.is_finite()) {
            return Err(SamplerError::NonFiniteLogits);
        }
        let mut ranked: Vec<(u32, f32)> = (0..).zip(candidates.iter().copied()).collect();
        ranked.sort_by(|a, b| b.1.total_cmp(&a.1).then(a.0.cmp(&b.0)));
        let Some(&(best, _)) = ranked.first() else {
            return Err(no_candidates);
        };
        if self.config.temperature == 0.0 {
            return Ok(vec![(best, 1.0)]);
        }
        if self.config.top_k > 0 {
            ranked.truncate(self.config.top_k);
        }
        Ok(self.nucleus(&ranked))
    }

    /// Temperature-scaled probabilities of the top-p set, renormalized.
    fn nucleus(&self, ranked: &[(u32, f32)]) -> Vec<(u32, f32)> {
        let top = ranked.first().map_or(0.0, |(_, l)| *l);
        let weights: Vec<(u32, f32)> = ranked
            .iter()
            .map(|(id, l)| (*id, ((l - top) / self.config.temperature).exp()))
            .collect();
        let total: f32 = weights.iter().map(|(_, w)| w).sum();
        let mut kept = Vec::new();
        let mut mass = 0.0;
        for (id, w) in weights {
            let p = w / total;
            kept.push((id, p));
            mass += p;
            if mass >= self.config.top_p {
                break;
            }
        }
        let kept_mass: f32 = kept.iter().map(|(_, p)| p).sum();
        kept.iter().map(|(id, p)| (*id, p / kept_mass)).collect()
    }

    /// splitmix64, mapped to [0, 1) using the top 24 bits.
    fn next_unit(&mut self) -> f32 {
        self.state = self.state.wrapping_add(0x9E37_79B9_7F4A_7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58_476D_1CE4_E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D0_49BB_1331_11EB);
        z ^= z >> 31;
        (z >> 40) as f32 / 16_777_216.0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn config(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> SamplerConfig {
        SamplerConfig {
            temperature,
            top_k,
            top_p,
            seed,
        }
    }

    fn sampler(temperature: f32, top_k: usize, top_p: f32, seed: u64) -> Sampler {
        Sampler::new(config(temperature, top_k, top_p, seed)).unwrap()
    }

    #[test]
    fn zero_temperature_is_greedy_with_lowest_index_on_ties() {
        let mut s = sampler(0.0, 0, 1.0, 1);
        assert_eq!(s.sample(&[0.1, 3.0, 3.0, 2.0], 4).unwrap(), 1);
    }

    #[test]
    fn distribution_applies_temperature_then_top_k_then_top_p_and_renormalizes() {
        let logits = [3.0_f32, 2.0, 1.0, 0.0, -1.0];
        let all = sampler(1.0, 0, 1.0, 1).distribution(&logits, 5).unwrap();
        let total: f32 = all.iter().map(|(_, p)| p).sum();
        assert!((total - 1.0).abs() < 1e-6 && all.len() == 5);
        assert!(
            all.windows(2).all(|w| w[0].1 >= w[1].1),
            "most likely first"
        );
        let top_k = sampler(1.0, 2, 1.0, 1).distribution(&logits, 5).unwrap();
        assert_eq!(top_k.iter().map(|(id, _)| *id).collect::<Vec<_>>(), [0, 1]);
        let expected = 1.0 / (1.0 + (-1.0_f32).exp());
        assert!(
            (top_k[0].1 - expected).abs() < 1e-6,
            "renormalized over the kept tokens"
        );
        let nucleus = sampler(1.0, 0, 0.7, 1).distribution(&logits, 5).unwrap();
        assert_eq!(
            nucleus.iter().map(|(id, _)| *id).collect::<Vec<_>>(),
            [0, 1],
            "smallest set reaching top_p, crossing token included"
        );
        assert_eq!(
            sampler(0.0, 5, 0.1, 1).distribution(&logits, 5).unwrap(),
            [(0, 1.0)]
        );
        let sharp = sampler(0.1, 0, 1.0, 1).distribution(&logits, 5).unwrap();
        assert!(sharp[0].1 > 0.99, "low temperature concentrates the mass");
    }

    #[test]
    fn samples_come_from_the_distribution() {
        let logits = [2.0_f32, 1.9, 0.0, -5.0];
        let mut s = sampler(0.7, 2, 0.9, 5);
        let allowed: Vec<u32> = s
            .distribution(&logits, 4)
            .unwrap()
            .iter()
            .map(|(id, _)| *id)
            .collect();
        for _ in 0..200 {
            assert!(allowed.contains(&s.sample(&logits, 4).unwrap()));
        }
    }

    #[test]
    fn same_seed_gives_same_tokens() {
        let logits: Vec<f32> = (0..50).map(|i| (i % 7) as f32 * 0.3).collect();
        let draw = |seed| {
            let mut s = sampler(1.0, 0, 1.0, seed);
            (0..20)
                .map(|_| s.sample(&logits, 50).unwrap())
                .collect::<Vec<_>>()
        };
        assert_eq!(draw(7), draw(7));
        assert_ne!(draw(7), draw(8));
    }

    #[test]
    fn top_k_one_always_picks_the_best() {
        let mut s = sampler(5.0, 1, 1.0, 3);
        for _ in 0..20 {
            assert_eq!(s.sample(&[0.0, 1.0, 0.5], 3).unwrap(), 1);
        }
    }

    #[test]
    fn tiny_top_p_keeps_only_the_best() {
        let mut s = sampler(1.0, 0, 0.01, 3);
        for _ in 0..20 {
            assert_eq!(s.sample(&[0.0, 1.0, 0.5], 3).unwrap(), 1);
        }
    }

    #[test]
    fn ids_beyond_usable_are_never_sampled() {
        let mut s = sampler(1.0, 0, 1.0, 9);
        for _ in 0..50 {
            assert!(s.sample(&[0.0, 0.0, 100.0, 100.0], 2).unwrap() < 2);
        }
    }

    #[test]
    fn sampling_follows_the_distribution() {
        let mut s = sampler(1.0, 0, 1.0, 11);
        let hits = (0..2000)
            .filter(|_| s.sample(&[0.0, 2.0_f32.ln()], 2).unwrap() == 1)
            .count();
        assert!(
            (1250..1420).contains(&hits),
            "expected ~1333 hits, got {hits}"
        );
    }

    #[test]
    fn non_finite_logits_are_an_error() {
        let mut s = sampler(0.0, 0, 1.0, 1);
        assert_eq!(
            s.sample(&[1.0, f32::NAN], 2),
            Err(SamplerError::NonFiniteLogits)
        );
    }

    #[test]
    fn empty_or_oversized_candidate_range_is_an_error() {
        let mut s = sampler(0.0, 0, 1.0, 1);
        assert!(s.sample(&[1.0], 0).is_err());
        assert!(s.sample(&[1.0], 2).is_err());
    }

    #[test]
    fn invalid_settings_are_rejected() {
        assert!(Sampler::new(config(-1.0, 0, 1.0, 0)).is_err());
        assert!(Sampler::new(config(f32::NAN, 0, 1.0, 0)).is_err());
        assert!(Sampler::new(config(1.0, 0, 0.0, 0)).is_err());
        assert!(Sampler::new(config(1.0, 0, 1.5, 0)).is_err());
    }
}
