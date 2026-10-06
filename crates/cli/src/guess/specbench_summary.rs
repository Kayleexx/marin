use super::specbench_report::RunRecord;

fn sum(runs: &[&RunRecord], f: impl Fn(&RunRecord) -> f64) -> f64 {
    runs.iter().map(|r| f(r)).sum()
}

fn div(a: f64, b: f64) -> f64 {
    if b > 0.0 { a / b } else { 0.0 }
}

fn mean(values: impl Iterator<Item = f64>) -> Option<f64> {
    let (n, total) = values.fold((0_usize, 0.0), |(n, t), v| (n.saturating_add(1), t + v));
    (n > 0).then(|| total / n as f64)
}

fn median(mut values: Vec<f64>) -> Option<f64> {
    values.sort_by(f64::total_cmp);
    values.get(values.len().checked_div(2)?).copied()
}

fn unique<'a>(runs: &'a [RunRecord], key: impl Fn(&'a RunRecord) -> &'a str) -> Vec<&'a str> {
    let mut seen: Vec<&str> = Vec::new();
    for run in runs {
        if !seen.contains(&key(run)) {
            seen.push(key(run));
        }
    }
    seen
}

fn tokens_per_second(runs: &[&RunRecord]) -> f64 {
    div(
        sum(runs, |r| r.tokens as f64),
        sum(runs, |r| r.decode_ms) / 1000.0,
    )
}

/// Ratios of this mode's speed to plain decoding's, run against run (same prompt, round and repeat).
fn paired_speedups(all: &[RunRecord], mode: &[&RunRecord]) -> Vec<f64> {
    mode.iter()
        .filter_map(|run| {
            let plain = all.iter().find(|p| {
                p.mode == plain_of(&run.mode)
                    && (p.class.as_str(), p.name.as_str(), p.round, p.rep)
                        == (run.class.as_str(), run.name.as_str(), run.round, run.rep)
            })?;
            let (a, b) = (
                div(run.tokens as f64, run.decode_ms),
                div(plain.tokens as f64, plain.decode_ms),
            );
            (b > 0.0).then(|| a / b)
        })
        .collect()
}

/// The mode a mode is compared with: `chat/spec` against `chat/plain`.
fn plain_of(mode: &str) -> String {
    mode.replace("spec", "plain")
}

fn row(cells: &[String], widths: &[usize]) -> String {
    let mut line = String::new();
    for (i, (text, width)) in cells.iter().zip(widths).enumerate() {
        if i == 0 || i == 1 {
            line.push_str(&format!("{text:<width$}  "));
        } else {
            line.push_str(&format!("{text:>width$}  "));
        }
    }
    line.trim_end().to_string()
}

fn cells(all: &[RunRecord], class: &str, mode: &str) -> Vec<String> {
    let runs: Vec<&RunRecord> = all
        .iter()
        .filter(|r| r.class == class && r.mode == mode)
        .collect();
    let baseline = plain_of(mode);
    let plain: Vec<&RunRecord> = all
        .iter()
        .filter(|r| r.class == class && r.mode == baseline)
        .collect();
    let speed = tokens_per_second(&runs);
    let tokens = sum(&runs, |r| r.tokens as f64);
    let passes = sum(&runs, |r| r.passes as f64);
    let drafted = sum(&runs, |r| r.drafted as f64);
    let paired = paired_speedups(all, &runs);
    let low = paired.iter().copied().fold(f64::INFINITY, f64::min);
    let high = paired.iter().copied().fold(0.0, f64::max);
    let same = runs.iter().filter(|r| r.same_as_plain).count();
    let joules = mean(runs.iter().filter_map(|r| r.joules_per_token));
    let watts = mean(runs.iter().filter_map(|r| r.power_w));
    let p95 = median(runs.iter().map(|r| r.gap_p95_ms).collect());
    let gain = if mode == baseline {
        "-".to_string()
    } else {
        format!("{:.2}x", div(speed, tokens_per_second(&plain)))
    };
    let spread = if mode == baseline || paired.is_empty() {
        "-".to_string()
    } else {
        format!("{low:.2}-{high:.2}x")
    };
    vec![
        class.to_string(),
        mode.to_string(),
        format!("{speed:.1}"),
        gain,
        spread,
        format!("{:.2}", div(tokens, passes)),
        format!(
            "{:.0}%",
            div(sum(&runs, |r| r.accepted as f64), drafted) * 100.0
        ),
        format!(
            "{:.0}%",
            div(sum(&runs, |r| r.guessed as f64), passes) * 100.0
        ),
        format!("{:.1}", p95.unwrap_or(0.0)),
        format!("{:.1}", div(sum(&runs, |r| r.guesser_ms) * 1000.0, tokens)),
        format!("{:.1}", div(sum(&runs, |r| r.rollback_ms) * 1000.0, tokens)),
        watts.map_or("-".into(), |w| format!("{w:.1}")),
        joules.map_or("-".into(), |j| format!("{j:.3}")),
        format!("{same}/{}", runs.len()),
    ]
}

/// A table of every class and mode: speed, gain over plain decoding, and what the guessing cost.
pub fn table(runs: &[RunRecord]) -> String {
    let head = [
        "scenario", "mode", "tok/s", "vs plain", "paired", "tok/pass", "guess ok", "guessed",
        "gap p95", "guesser", "rollback", "W", "J/tok", "same",
    ];
    let widths = [16, 7, 6, 8, 11, 8, 8, 7, 8, 7, 8, 5, 6, 6];
    let mut out = row(&head.map(String::from), &widths);
    out.push_str("\n                                                                                 (ms)  (us/tok) (us/tok)");
    let mut modes = unique(runs, |r| &r.mode);
    modes.sort_unstable();
    for class in unique(runs, |r| &r.class) {
        for &mode in &modes {
            if runs.iter().any(|r| r.class == class && r.mode == mode) {
                out.push('\n');
                out.push_str(&row(&cells(runs, class, mode), &widths));
            }
        }
    }
    out
}

pub fn prompts(runs: &[RunRecord]) -> String {
    let mut out = String::from("prompt tokens and time to first token (the same for every mode):");
    for class in unique(runs, |r| &r.class) {
        let mine: Vec<&RunRecord> = runs
            .iter()
            .filter(|r| r.class == class && r.mode.ends_with("plain"))
            .collect();
        let tokens = mean(mine.iter().map(|r| r.prompt_tokens as f64)).unwrap_or(0.0);
        let ttft = median(mine.iter().map(|r| r.ttft_ms).collect()).unwrap_or(0.0);
        out.push_str(&format!(
            "\n  {class:<16} {tokens:>6.0} tokens   first token {ttft:>7.0} ms"
        ));
    }
    out
}
