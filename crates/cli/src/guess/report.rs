use std::path::{Path, PathBuf};

use super::data::{CostModel, Costs, Curve, Transcripts, read_saved};
use super::replay::{Config, Replay, replay};
use crate::CliError;

/// Choosing a token on the CPU: measured at 0.15 ms (BENCHMARKS.md, "What choosing tokens costs").
const SAMPLE_MS: f64 = 0.15;
/// Every extra token in a checking pass needs its own logits: copying them takes about 0.1 ms (measured by
/// `marin-run probe`) and one more output-head row about 0.05 ms (an estimate, not measured).
const ROW_EXTRA_MS: f64 = 0.15;
const DEPTHS: [usize; 3] = [1, 2, 4];
const MIN_NS: [usize; 5] = [1, 2, 3, 4, 6];
const DRAFTS: [usize; 3] = [2, 4, 7];

pub struct Options {
    pub transcripts: PathBuf,
    pub costs: Option<PathBuf>,
    pub detail: Config,
}

#[derive(Default, Clone)]
struct Tally {
    tokens: usize,
    passes: usize,
    guessed: usize,
    eligible: [usize; 3],
    right: [usize; 3],
    plain_ms: f64,
    theory_base_ms: f64,
    theory_ms: f64,
    spec_ms: f64,
    guesser_ms: f64,
    memory: usize,
}

impl Tally {
    fn add(&mut self, run: &Replay, costs: Option<&CostModel>) {
        for pass in &run.passes {
            self.tokens = self.tokens.saturating_add(pass.produced);
            self.passes = self.passes.saturating_add(1);
            self.guessed = self.guessed.saturating_add(usize::from(pass.drafted > 0));
            for (slot, depth) in DEPTHS.iter().enumerate() {
                if pass.drafted >= *depth {
                    bump(&mut self.eligible, slot);
                    if pass.accepted >= *depth {
                        bump(&mut self.right, slot);
                    }
                }
            }
            let Some(costs) = costs else { continue };
            let produced = pass.produced as f64;
            let one = costs.pass_ms(pass.context, 1);
            let checked = costs.pass_ms(pass.context, pass.drafted.saturating_add(1));
            self.plain_ms += produced * (one + SAMPLE_MS);
            self.theory_base_ms += produced * one;
            self.theory_ms += checked;
            self.spec_ms += checked + pass.drafted as f64 * ROW_EXTRA_MS + produced * SAMPLE_MS;
        }
        self.guesser_ms += run.guesser.as_secs_f64() * 1000.0;
        self.memory = self.memory.max(run.memory);
    }

    fn merge(&mut self, other: &Self) {
        self.tokens = self.tokens.saturating_add(other.tokens);
        self.passes = self.passes.saturating_add(other.passes);
        self.guessed = self.guessed.saturating_add(other.guessed);
        for slot in 0..DEPTHS.len() {
            add_slot(&mut self.eligible, &other.eligible, slot);
            add_slot(&mut self.right, &other.right, slot);
        }
        self.plain_ms += other.plain_ms;
        self.theory_base_ms += other.theory_base_ms;
        self.theory_ms += other.theory_ms;
        self.spec_ms += other.spec_ms;
        self.guesser_ms += other.guesser_ms;
        self.memory = self.memory.max(other.memory);
    }

    fn tokens_per_pass(&self) -> f64 {
        ratio(self.tokens, self.passes)
    }

    fn theory(&self) -> f64 {
        div(self.theory_base_ms, self.theory_ms)
    }

    fn projected(&self) -> f64 {
        div(self.plain_ms, self.spec_ms + self.guesser_ms)
    }
}

fn bump(slots: &mut [usize; 3], slot: usize) {
    if let Some(n) = slots.get_mut(slot) {
        *n = n.saturating_add(1);
    }
}

fn add_slot(into: &mut [usize; 3], from: &[usize; 3], slot: usize) {
    if let (Some(a), Some(b)) = (into.get_mut(slot), from.get(slot)) {
        *a = a.saturating_add(*b);
    }
}

fn ratio(a: usize, b: usize) -> f64 {
    div(a as f64, b as f64)
}

fn div(a: f64, b: f64) -> f64 {
    if b > 0.0 { a / b } else { 0.0 }
}

fn percent(part: usize, whole: usize) -> String {
    if whole == 0 {
        "-".into()
    } else {
        format!("{:.0}%", ratio(part, whole) * 100.0)
    }
}

/// Every reply of one kind (`greedy` or `chat`) for one class of prompt: (prompt, reply) token lists.
struct Group<'a> {
    class: &'a str,
    mode: &'a str,
    items: Vec<(&'a [u32], &'a [u32])>,
}

fn groups(transcripts: &Transcripts) -> Vec<Group<'_>> {
    let mut out: Vec<Group<'_>> = Vec::new();
    for prompt in &transcripts.prompts {
        for reply in &prompt.replies {
            let item = (prompt.prompt.as_slice(), reply.tokens.as_slice());
            match out
                .iter_mut()
                .find(|g| g.class == prompt.class && g.mode == reply.mode)
            {
                Some(group) => group.items.push(item),
                None => out.push(Group {
                    class: &prompt.class,
                    mode: &reply.mode,
                    items: vec![item],
                }),
            }
        }
    }
    let mut modes: Vec<&str> = Vec::new();
    for group in &out {
        if !modes.contains(&group.mode) {
            modes.push(group.mode);
        }
    }
    for mode in modes {
        let items = out
            .iter()
            .filter(|g| g.mode == mode)
            .flat_map(|g| g.items.iter().copied())
            .collect();
        out.push(Group {
            class: "all",
            mode,
            items,
        });
    }
    out
}

fn tally(group: &Group<'_>, config: Config, costs: Option<&CostModel>) -> Tally {
    let mut total = Tally::default();
    for (prompt, reply) in &group.items {
        total.add(&replay(prompt, reply, config), costs);
    }
    total
}

fn cell(cells: Vec<String>, widths: &[usize]) -> String {
    let mut line = String::new();
    for (i, (text, width)) in cells.iter().zip(widths).enumerate() {
        if i == 0 {
            line.push_str(&format!("{text:<width$}"));
        } else {
            line.push_str(&format!("  {text:>width$}"));
        }
    }
    line.trim_end().to_string()
}

fn detail_table(
    groups: &[Group<'_>],
    mode: &str,
    detail: Config,
    costs: Option<&CostModel>,
) -> String {
    let widths = [14, 7, 7, 7, 7, 7, 7, 10, 8, 9, 15];
    let head = [
        "class",
        "prompts",
        "tokens",
        "guess",
        "right@1",
        "right@2",
        "right@4",
        "tokens/pass",
        "theory",
        "projected",
        "per prompt",
    ];
    let mut out = cell(head.iter().map(|s| s.to_string()).collect(), &widths);
    for group in groups.iter().filter(|g| g.mode == mode) {
        let mut total = Tally::default();
        let mut speeds = Vec::new();
        for (prompt, reply) in &group.items {
            let mut one = Tally::default();
            one.add(&replay(prompt, reply, detail), costs);
            speeds.push(one.projected());
            total.merge(&one);
        }
        let low = speeds.iter().copied().fold(f64::INFINITY, f64::min);
        let high = speeds.iter().copied().fold(0.0, f64::max);
        let times = |value: f64| match costs {
            Some(_) => format!("{value:.2}x"),
            None => "-".into(),
        };
        let spread = match costs {
            Some(_) => format!("{low:.2}-{high:.2}x"),
            None => "-".into(),
        };
        let row = vec![
            group.class.to_string(),
            group.items.len().to_string(),
            total.tokens.to_string(),
            percent(total.guessed, total.passes),
            percent(
                total.right.first().copied().unwrap_or(0),
                total.eligible.first().copied().unwrap_or(0),
            ),
            percent(
                total.right.get(1).copied().unwrap_or(0),
                total.eligible.get(1).copied().unwrap_or(0),
            ),
            percent(
                total.right.get(2).copied().unwrap_or(0),
                total.eligible.get(2).copied().unwrap_or(0),
            ),
            format!("{:.2}", total.tokens_per_pass()),
            times(total.theory()),
            times(total.projected()),
            spread,
        ];
        out.push('\n');
        out.push_str(&cell(row, &widths));
    }
    out
}

fn configs() -> impl Iterator<Item = Config> {
    MIN_NS
        .iter()
        .flat_map(|&min_n| DRAFTS.iter().map(move |&max_len| Config { min_n, max_len }))
}

fn grid_table(
    groups: &[Group<'_>],
    mode: &str,
    costs: Option<&CostModel>,
    value: impl Fn(&Tally) -> f64,
) -> String {
    let widths: Vec<usize> = std::iter::once(14).chain(configs().map(|_| 7)).collect();
    let head: Vec<String> = std::iter::once("class".to_string())
        .chain(configs().map(|c| format!("n{}/d{}", c.min_n, c.max_len)))
        .collect();
    let mut out = cell(head, &widths);
    for group in groups.iter().filter(|g| g.mode == mode) {
        let row = std::iter::once(group.class.to_string())
            .chain(configs().map(|c| format!("{:.2}", value(&tally(group, c, costs)))))
            .collect();
        out.push('\n');
        out.push_str(&cell(row, &widths));
    }
    out
}

fn priced_table(groups: &[Group<'_>], mode: &str, detail: Config, curves: &[Curve]) -> String {
    let widths: Vec<usize> = std::iter::once(14)
        .chain(curves.iter().map(|_| 10))
        .collect();
    let head = std::iter::once("class".to_string())
        .chain(curves.iter().map(|c| format!("ctx {}", c.context)))
        .collect();
    let mut out = cell(head, &widths);
    for group in groups.iter().filter(|g| g.mode == mode) {
        let row = std::iter::once(group.class.to_string())
            .chain(curves.iter().map(|curve| {
                let model = CostModel::new(vec![curve.clone()]);
                format!("{:.2}x", tally(group, detail, Some(&model)).projected())
            }))
            .collect();
        out.push('\n');
        out.push_str(&cell(row, &widths));
    }
    out
}

fn load_costs(path: &Path, transcripts: &Transcripts) -> Result<Vec<Curve>, CliError> {
    let costs: Costs = read_saved(path)?;
    if costs.quantization != transcripts.quantization {
        return Err(CliError::NotComparable(format!(
            "pass costs are for {} weights, the replies for {}",
            costs.quantization, transcripts.quantization
        )));
    }
    Ok(costs.curves)
}

pub fn run(options: &Options) -> Result<(), CliError> {
    let transcripts: Transcripts = read_saved(&options.transcripts)?;
    let curves = options
        .costs
        .as_deref()
        .map(|path| load_costs(path, &transcripts))
        .transpose()?;
    let model = curves.clone().map(CostModel::new);
    let costs = model.as_ref();
    let groups = groups(&transcripts);
    let mut modes: Vec<&str> = Vec::new();
    for group in &groups {
        if !modes.contains(&group.mode) {
            modes.push(group.mode);
        }
    }
    println!(
        "guessing report: {} · {} · {}",
        transcripts.model, transcripts.quantization, transcripts.backend
    );
    let d = options.detail;
    for mode in &modes {
        println!(
            "\n{mode}: guess after {}+ matching tokens, up to {} tokens at a time",
            d.min_n, d.max_len
        );
        println!("{}", detail_table(&groups, mode, d, costs));
        if let Some(curves) = &curves {
            println!("\n{mode}: the same guesses, every pass priced at one context length");
            println!("{}", priced_table(&groups, mode, d, curves));
        }
    }
    for mode in &modes {
        println!(
            "\n{mode}: tokens settled per model pass (n = matching tokens needed, d = tokens guessed at most)"
        );
        println!(
            "{}",
            grid_table(&groups, mode, costs, Tally::tokens_per_pass)
        );
        if costs.is_some() {
            println!("\n{mode}: projected speed-up over one token per pass");
            println!("{}", grid_table(&groups, mode, costs, Tally::projected));
        }
    }
    if let Some(pooled) = groups.iter().find(|g| g.class == "all") {
        let t = tally(pooled, d, costs);
        println!(
            "\nguesser cost ({} mode, all prompts): {:.1} µs per generated token, up to {:.0} KB of memory",
            pooled.mode,
            div(t.guesser_ms * 1000.0, t.tokens as f64),
            t.memory as f64 / 1024.0
        );
    }
    Ok(())
}
