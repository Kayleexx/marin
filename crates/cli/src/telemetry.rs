use std::time::Duration;

use kv::KvCache;
use metrics::{process_memory, spread};
use runtime::{Outcome, SequenceCache};

use crate::session::Session;
use crate::units;

/// Milliseconds below a second, seconds above.
pub fn time(duration: Duration) -> String {
    if duration >= Duration::from_secs(1) {
        format!("{:.2} s", duration.as_secs_f64())
    } else {
        format!("{:.1} ms", duration.as_secs_f64() * 1000.0)
    }
}

pub fn rate(tokens: usize, duration: Duration) -> Option<f64> {
    (!duration.is_zero() && tokens > 0).then(|| tokens as f64 / duration.as_secs_f64())
}

/// Where the time of one reply went: first token, prompt pass, decode, per-token latency, sampling.
pub fn request_lines(o: &Outcome) -> Vec<String> {
    let prompt = if o.reused > 0 {
        format!(
            "prompt          {} tokens   ({} kept from the last reply, {} read)",
            o.prompt_tokens,
            o.reused,
            o.prompt_tokens.saturating_sub(o.reused)
        )
    } else {
        format!("prompt          {} tokens", o.prompt_tokens)
    };
    let mut lines = vec![
        prompt,
        format!("generated       {} tokens", o.generated),
        format!(
            "first token     {}   (prompt pass {} + first sample)",
            time(o.prefill),
            time(o.prompt_forward)
        ),
    ];
    if let Some(tps) = rate(o.prompt_tokens.saturating_sub(o.reused), o.prompt_forward) {
        lines.push(format!("prefill         {tps:.1} tok/s"));
    }
    let decoded = o.generated.saturating_sub(1);
    if let Some(tps) = rate(decoded, o.decode) {
        lines.push(format!(
            "decode          {tps:.2} tok/s   ({} for {decoded} tokens)",
            time(o.decode)
        ));
    }
    if let Some(s) = spread(&o.gaps) {
        lines.push(format!(
            "token latency   p50 {} · p95 {} · max {}",
            time(s.p50),
            time(s.p95),
            time(s.max)
        ));
    }
    let guess = &o.speculation;
    if guess.guessed_passes > 0 {
        lines.push(format!(
            "guessing        {} of {} passes checked guesses, {} of {} guessed tokens right ({:.2} tokens per pass)",
            guess.guessed_passes,
            guess.passes,
            guess.accepted,
            guess.drafted,
            decoded as f64 / guess.passes.max(1) as f64
        ));
    }
    let total = o.prefill.saturating_add(o.decode);
    if !total.is_zero() {
        lines.push(format!(
            "sampling        {} in total ({:.1}% of the reply)",
            time(o.sampling),
            o.sampling.as_secs_f64() / total.as_secs_f64() * 100.0
        ));
    }
    lines.push(format!("total           {}", time(total)));
    lines
}

/// Memory the model and this process hold right now.
pub fn resource_lines(s: &Session) -> Vec<String> {
    let fp = &s.loaded.footprint;
    let used = KvCache::bytes_for(
        s.loaded.model.config().layers,
        s.loaded.model.config().kv_heads,
        s.loaded.model.config().head_dim,
        s.kv().used(),
    )
    .and_then(|b| u64::try_from(b).ok())
    .unwrap_or(0);
    let held = match s.kv().kv_held() {
        Some((pages, bytes)) => format!(
            "{} held for {pages} pages, up to {}",
            units::bytes(u64::try_from(bytes).unwrap_or(0)),
            units::bytes(fp.kv)
        ),
        None => format!("{} reserved", units::bytes(fp.kv)),
    };
    let mut lines = vec![
        format!("weights         {}", units::bytes(fp.weights)),
        format!(
            "kv cache        {held} · {} in use ({} of {} positions)",
            units::bytes(used),
            units::count(s.kv().used() as u64),
            units::count(SequenceCache::capacity(s.kv()) as u64)
        ),
    ];
    if let Some(m) = process_memory() {
        lines.push(format!(
            "process memory  {} now · {} peak",
            units::bytes(m.rss_bytes),
            units::bytes(m.peak_rss_bytes)
        ));
    }
    if let Some((free, total)) = s.loaded.model.gpu_memory() {
        lines.push(format!(
            "gpu memory      {} of {} in use (all programs)",
            units::bytes(total.saturating_sub(free)),
            units::bytes(total)
        ));
    }
    lines
}

#[cfg(test)]
mod tests {
    use super::*;
    use runtime::StopReason;

    fn outcome() -> Outcome {
        Outcome {
            prompt_tokens: 100,
            reused: 0,
            generated: 11,
            stop: StopReason::MaxTokens,
            prefill: Duration::from_millis(520),
            decode: Duration::from_millis(200),
            prompt_forward: Duration::from_millis(500),
            sampling: Duration::from_millis(36),
            gaps: vec![Duration::from_millis(20); 10],
            speculation: Default::default(),
        }
    }

    #[test]
    fn guessing_is_reported_only_when_it_happened() {
        assert!(!request_lines(&outcome()).join("\n").contains("guessing"));
        let mut guessed = outcome();
        guessed.speculation.passes = 4;
        guessed.speculation.guessed_passes = 2;
        guessed.speculation.drafted = 14;
        guessed.speculation.accepted = 6;
        let text = request_lines(&guessed).join("\n");
        assert!(
            text.contains(
                "2 of 4 passes checked guesses, 6 of 14 guessed tokens right (2.50 tokens per pass)"
            ),
            "{text}"
        );
    }

    #[test]
    fn a_reply_reports_first_token_prefill_decode_latency_and_sampling() {
        let text = request_lines(&outcome()).join("\n");
        for expected in [
            "100 tokens",
            "first token     520.0 ms   (prompt pass 500.0 ms + first sample)",
            "prefill         200.0 tok/s",
            "decode          50.00 tok/s   (200.0 ms for 10 tokens)",
            "p50 20.0 ms · p95 20.0 ms · max 20.0 ms",
            "36.0 ms in total (5.0% of the reply)",
            "total           720.0 ms",
        ] {
            assert!(text.contains(expected), "{expected:?} missing from\n{text}");
        }
    }

    #[test]
    fn a_reply_with_one_token_has_no_decode_or_latency_lines() {
        let mut o = outcome();
        o.generated = 1;
        o.decode = Duration::ZERO;
        o.gaps.clear();
        let text = request_lines(&o).join("\n");
        assert!(!text.contains("decode") && !text.contains("token latency"));
    }

    #[test]
    fn times_switch_to_seconds_at_one_second() {
        assert_eq!(time(Duration::from_millis(1500)), "1.50 s");
        assert_eq!(time(Duration::from_micros(300)), "0.3 ms");
    }
}
