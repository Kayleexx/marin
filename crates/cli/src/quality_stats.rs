use serde::{Deserialize, Serialize};

use crate::CliError;

/// The format of a saved scoring run; `--against` refuses anything else.
pub const FORMAT: u32 = 1;

/// How one model scored a text: for every scored position, which token it ranked first and how likely it thought
/// the token that really came next was. Saved so a later run can be compared with it.
#[derive(Serialize, Deserialize, Debug, PartialEq)]
pub struct Scores {
    pub format: u32,
    pub model: String,
    pub quantization: String,
    /// Tokens read as a single prompt before scoring starts (the prompt path of the engine).
    pub prompt_tokens: usize,
    pub text_tokens: usize,
    /// Fingerprint of the token ids, so a comparison only ever pairs runs over the same text.
    pub text_fingerprint: u64,
    pub top1: Vec<u32>,
    pub target_logprob: Vec<f32>,
}

#[derive(Debug, PartialEq)]
pub struct Summary {
    pub scored: usize,
    pub mean_loss: f64,
    pub perplexity: f64,
    pub top1_hit_rate: f64,
}

#[derive(Debug, PartialEq)]
pub struct Agreement {
    pub same_top1: f64,
    pub loss_change: f64,
    pub perplexity_change_percent: f64,
    pub mean_abs_logprob_diff: f64,
}

/// FNV-1a over the token ids.
pub fn fingerprint(tokens: &[u32]) -> u64 {
    tokens.iter().fold(0xcbf2_9ce4_8422_2325, |hash, token| {
        token.to_le_bytes().iter().fold(hash, |h, b| {
            (h ^ u64::from(*b)).wrapping_mul(0x0100_0000_01b3)
        })
    })
}

/// The first-ranked token among `logits[..known]`, and the log-probability of `target` among them.
pub fn score(logits: &[f32], known: usize, target: u32) -> Result<(u32, f32), CliError> {
    let seen = logits.get(..known).unwrap_or(logits);
    let target_logit = usize::try_from(target)
        .ok()
        .and_then(|i| seen.get(i))
        .copied()
        .ok_or_else(|| CliError::BadOption(format!("token {target} is outside the vocabulary")))?;
    let (mut best, mut best_logit) = (0_usize, f32::NEG_INFINITY);
    for (index, logit) in seen.iter().enumerate() {
        if *logit > best_logit {
            (best, best_logit) = (index, *logit);
        }
    }
    let total: f64 = seen.iter().map(|l| f64::from(l - best_logit).exp()).sum();
    let logprob = f64::from(target_logit - best_logit) - total.ln();
    Ok((u32::try_from(best).unwrap_or(u32::MAX), logprob as f32))
}

fn mean(values: impl Iterator<Item = f64>) -> f64 {
    let (sum, count) = values.fold((0.0, 0_u64), |(s, n), v| (s + v, n.saturating_add(1)));
    if count == 0 { 0.0 } else { sum / count as f64 }
}

/// How well the model predicted the real text. `targets` are the tokens that really followed each position.
pub fn summarize(scores: &Scores, targets: &[u32]) -> Summary {
    let mean_loss = mean(scores.target_logprob.iter().map(|p| -f64::from(*p)));
    let hits = scores
        .top1
        .iter()
        .zip(targets)
        .filter(|(a, b)| a == b)
        .count();
    Summary {
        scored: scores.target_logprob.len(),
        mean_loss,
        perplexity: mean_loss.exp(),
        top1_hit_rate: hits as f64 / scores.top1.len().max(1) as f64,
    }
}

/// How far `run` has drifted from `reference` on the same text.
pub fn agreement(reference: &Scores, run: &Scores) -> Result<Agreement, CliError> {
    if reference.format != FORMAT {
        return Err(CliError::BadOption(
            "the reference file is from another version; run `quality --save` again".into(),
        ));
    }
    if reference.text_fingerprint != run.text_fingerprint
        || reference.prompt_tokens != run.prompt_tokens
        || reference.top1.len() != run.top1.len()
        || reference.target_logprob.len() != run.target_logprob.len()
    {
        return Err(CliError::NotComparable(
            "the two runs did not score the same text".into(),
        ));
    }
    let same = reference
        .top1
        .iter()
        .zip(&run.top1)
        .filter(|(a, b)| a == b)
        .count();
    let diffs = reference
        .target_logprob
        .iter()
        .zip(&run.target_logprob)
        .map(|(a, b)| f64::from(*a) - f64::from(*b));
    let loss_change = mean(diffs.clone());
    let reference_loss = mean(reference.target_logprob.iter().map(|p| -f64::from(*p)));
    Ok(Agreement {
        same_top1: same as f64 / run.top1.len().max(1) as f64,
        loss_change,
        perplexity_change_percent: ((reference_loss + loss_change).exp() / reference_loss.exp()
            - 1.0)
            * 100.0,
        mean_abs_logprob_diff: mean(diffs.map(f64::abs)),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scores(top1: &[u32], logprob: &[f32]) -> Scores {
        Scores {
            format: FORMAT,
            model: "m".into(),
            quantization: "none".into(),
            prompt_tokens: 4,
            text_tokens: top1.len().saturating_add(4),
            text_fingerprint: 7,
            top1: top1.to_vec(),
            target_logprob: logprob.to_vec(),
        }
    }

    #[test]
    fn a_uniform_guess_over_four_tokens_costs_ln_four() {
        let (best, logprob) = score(&[1.0, 1.0, 1.0, 1.0], 4, 2).unwrap();
        assert_eq!(best, 0, "ties go to the first token");
        assert!((f64::from(logprob) + 4.0_f64.ln()).abs() < 1e-6);
    }

    #[test]
    fn a_dominant_token_is_nearly_certain_and_large_logits_do_not_overflow() {
        let (best, logprob) = score(&[0.0, 1000.0, 3.0], 3, 1).unwrap();
        assert_eq!(best, 1);
        assert!(logprob.abs() < 1e-6);
        let (_, unlikely) = score(&[0.0, 1000.0, 3.0], 3, 0).unwrap();
        assert!((unlikely + 1000.0).abs() < 1e-3);
    }

    #[test]
    fn only_the_known_vocabulary_counts_and_a_target_outside_it_is_refused() {
        let logits = [0.0, 0.0, 50.0];
        let (best, _) = score(&logits, 2, 0).unwrap();
        assert_eq!(best, 0, "the padding entry is ignored");
        assert!(score(&logits, 2, 2).is_err());
        assert!(score(&logits, 3, u32::MAX).is_err());
    }

    #[test]
    fn summary_reports_loss_perplexity_and_how_often_the_top_choice_was_right() {
        let s = scores(&[5, 6, 9], &[-1.0, -3.0, -2.0]);
        let summary = summarize(&s, &[5, 6, 7]);
        assert_eq!(summary.scored, 3);
        assert!((summary.mean_loss - 2.0).abs() < 1e-9);
        assert!((summary.perplexity - 2.0_f64.exp()).abs() < 1e-9);
        assert!((summary.top1_hit_rate - 2.0 / 3.0).abs() < 1e-9);
        assert_eq!(summarize(&scores(&[], &[]), &[]).scored, 0);
    }

    #[test]
    fn agreement_counts_matching_top_choices_and_the_drift_in_loss() {
        let reference = scores(&[1, 2, 3, 4], &[-1.0, -1.0, -1.0, -1.0]);
        let run = scores(&[1, 2, 9, 9], &[-1.5, -1.5, -1.5, -0.5]);
        let a = agreement(&reference, &run).unwrap();
        assert!((a.same_top1 - 0.5).abs() < 1e-9);
        assert!((a.loss_change - 0.25).abs() < 1e-9);
        assert!((a.mean_abs_logprob_diff - 0.5).abs() < 1e-9);
        assert!((a.perplexity_change_percent - ((0.25_f64).exp() - 1.0) * 100.0).abs() < 1e-9);
        let same = agreement(&reference, &reference).unwrap();
        assert_eq!((same.same_top1, same.loss_change), (1.0, 0.0));
    }

    #[test]
    fn runs_over_different_text_or_old_files_are_not_compared() {
        let reference = scores(&[1, 2], &[-1.0, -1.0]);
        let mut other_text = scores(&[1, 2], &[-1.0, -1.0]);
        other_text.text_fingerprint = 8;
        assert!(matches!(
            agreement(&reference, &other_text),
            Err(CliError::NotComparable(_))
        ));
        assert!(agreement(&reference, &scores(&[1], &[-1.0])).is_err());
        let mut old = scores(&[1, 2], &[-1.0, -1.0]);
        old.format = 0;
        assert!(agreement(&old, &reference).is_err());
    }

    #[test]
    fn the_fingerprint_depends_on_every_token_and_their_order() {
        assert_ne!(fingerprint(&[1, 2, 3]), fingerprint(&[1, 2, 4]));
        assert_ne!(fingerprint(&[1, 2, 3]), fingerprint(&[3, 2, 1]));
        assert_eq!(fingerprint(&[1, 2, 3]), fingerprint(&[1, 2, 3]));
        assert_ne!(fingerprint(&[]), fingerprint(&[0]));
    }
}
