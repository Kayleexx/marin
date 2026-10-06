//! Temperature 0 must pick exactly the token the old sort-everything path picked: the largest logit under
//! `f32::total_cmp` (so +0.0 beats -0.0), the lowest id on ties, and the same errors for bad input.

use sampler::{Sampler, SamplerConfig, SamplerError};

/// The previous greedy path, kept as the reference: rank every candidate, take the first.
fn reference(logits: &[f32], usable: usize) -> Result<u32, SamplerError> {
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
    ranked.first().map(|(id, _)| *id).ok_or(no_candidates)
}

fn greedy(
    logits: &[f32],
    usable: usize,
    top_k: usize,
    top_p: f32,
    seed: u64,
) -> Result<u32, SamplerError> {
    let config = SamplerConfig {
        temperature: 0.0,
        top_k,
        top_p,
        seed,
    };
    Sampler::new(config)?.sample(logits, usable)
}

fn distribution(logits: &[f32], usable: usize) -> Result<Vec<(u32, f32)>, SamplerError> {
    let config = SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: 0,
    };
    Sampler::new(config)?.distribution(logits, usable)
}

struct Rng(u64);

impl Rng {
    fn next(&mut self) -> u64 {
        self.0 ^= self.0.wrapping_shl(13);
        self.0 ^= self.0.wrapping_shr(7);
        self.0 ^= self.0.wrapping_shl(17);
        self.0
    }

    fn below(&mut self, n: usize) -> usize {
        let drawn = self.next().checked_rem(n as u64).unwrap_or(0);
        usize::try_from(drawn).unwrap_or(0)
    }

    /// A logit from one of three alphabets: a few integers (many ties), signed zeros, or anything finite.
    fn logit(&mut self, kind: usize) -> f32 {
        const ZEROS: [f32; 4] = [-0.0, 0.0, 1.0, -1.0];
        match kind {
            0 => self.below(3) as f32,
            1 => ZEROS.get(self.below(ZEROS.len())).copied().unwrap_or(0.0),
            _ => {
                let value = f32::from_bits(self.next() as u32);
                if value.is_finite() { value } else { 0.0 }
            }
        }
    }
}

#[test]
fn hand_picked_cases_pick_the_expected_token_and_match_the_reference() {
    let tiny = f32::MIN_POSITIVE / 2.0;
    let cases: [(&[f32], usize, u32); 15] = [
        (&[0.1, 3.0, 3.0, 2.0], 4, 1),
        (&[3.0, 3.0, 1.0], 3, 0),
        (&[1.0, 2.0, 2.0, 2.0], 4, 1),
        (&[1.0, 2.0, 3.0, 3.0], 4, 2),
        (&[1.0; 5], 5, 0),
        (&[5.0], 1, 0),
        (&[2.0, 2.0], 2, 0),
        (&[0.0, 1.0, 9.0, 9.0], 2, 1),
        (&[-3.0, -1.0, -1.0, -2.0], 4, 1),
        (&[-0.0, 0.0], 2, 1),
        (&[0.0, -0.0], 2, 0),
        (&[-0.0, -0.0, 0.0, 0.0], 4, 2),
        (&[f32::MIN, f32::MAX, f32::MAX], 3, 1),
        (&[tiny, 0.0, -tiny], 3, 0),
        (&[1.0, 2.0, 3.0, 9.0], 3, 2),
    ];
    for (logits, usable, expected) in cases {
        assert_eq!(reference(logits, usable), Ok(expected), "{logits:?}");
        assert_eq!(
            greedy(logits, usable, 0, 1.0, 0),
            Ok(expected),
            "{logits:?}"
        );
        assert_eq!(distribution(logits, usable), Ok(vec![(expected, 1.0)]));
    }
}

#[test]
fn bad_input_gives_the_same_errors_as_the_reference() {
    let cases: [(&[f32], usize); 9] = [
        (&[1.0, f32::NAN], 2),
        (&[f32::NAN], 1),
        (&[f32::INFINITY, 1.0], 2),
        (&[1.0, f32::NEG_INFINITY], 2),
        (&[1.0], 0),
        (&[1.0], 2),
        (&[], 0),
        (&[], 3),
        (&[f32::NAN, f32::INFINITY, 1.0], 3),
    ];
    for (logits, usable) in cases {
        let expected = reference(logits, usable);
        assert!(expected.is_err(), "{logits:?} should be rejected");
        assert_eq!(greedy(logits, usable, 0, 1.0, 0), expected, "{logits:?}");
        assert_eq!(
            distribution(logits, usable),
            expected.map(|id| vec![(id, 1.0)])
        );
    }
}

#[test]
fn non_finite_logits_beyond_the_usable_range_are_ignored_as_before() {
    let logits = [1.0, 4.0, f32::NAN, f32::INFINITY];
    assert_eq!(reference(&logits, 2), Ok(1));
    assert_eq!(greedy(&logits, 2, 0, 1.0, 0), Ok(1));
}

#[test]
fn random_logits_match_the_reference_for_every_filter_setting() {
    let filters = [(0, 1.0), (1, 0.1), (5, 0.5), (0, 0.01), (50, 1.0)];
    let mut rng = Rng(0x9E37_79B9_7F4A_7C15);
    for case in 0..6000_usize {
        let len = 1 + rng.below(70);
        let kind = case % 3;
        let logits: Vec<f32> = (0..len).map(|_| rng.logit(kind)).collect();
        let usable = 1 + rng.below(len);
        let expected = reference(&logits, usable);
        assert!(expected.is_ok(), "finite input must be accepted");
        for (top_k, top_p) in filters {
            let seed = rng.next();
            assert_eq!(
                greedy(&logits, usable, top_k, top_p, seed),
                expected,
                "{logits:?} usable {usable}"
            );
        }
        assert_eq!(
            distribution(&logits, usable),
            expected.map(|id| vec![(id, 1.0)]),
            "{logits:?} usable {usable}"
        );
    }
}

#[test]
fn a_vocabulary_sized_input_with_many_ties_matches_the_reference() {
    const VOCAB: usize = 151_936;
    let mut rng = Rng(0xD1B5_4A32_D192_ED03);
    let mut logits: Vec<f32> = (0..VOCAB).map(|_| rng.below(50) as f32).collect();
    for peak in [10_u32, 1, 150_000] {
        let tail = logits.iter_mut().skip(usize::try_from(peak).unwrap());
        for slot in tail.take(1) {
            *slot = 1000.0;
        }
        assert_eq!(greedy(&logits, VOCAB, 0, 1.0, 0), reference(&logits, VOCAB));
        assert_eq!(
            greedy(&logits, 151_000, 0, 1.0, 0),
            reference(&logits, 151_000)
        );
    }
    let all_equal = vec![0.5_f32; VOCAB];
    assert_eq!(greedy(&all_equal, VOCAB, 0, 1.0, 0), Ok(0));
    let last_wins = {
        let mut v = vec![0.0_f32; VOCAB];
        v.pop();
        v.push(1.0);
        v
    };
    assert_eq!(
        greedy(&last_wins, VOCAB, 0, 1.0, 0),
        Ok(u32::try_from(VOCAB - 1).unwrap())
    );
}

#[test]
fn greedy_does_not_depend_on_the_seed_or_on_earlier_calls() {
    let logits = [0.5, 2.5, 2.5, -1.0];
    let mut sampler = Sampler::new(SamplerConfig {
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: 3,
    })
    .unwrap();
    for _ in 0..100 {
        assert_eq!(sampler.sample(&logits, 4), Ok(1));
    }
}
