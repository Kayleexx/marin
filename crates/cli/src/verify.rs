use std::path::Path;
use std::sync::atomic::{AtomicBool, Ordering};

use loader::Tokenizer;
use model::{BackendChoice, Engine, EngineCache, Precision};
use runtime::CausalLm;

use crate::CliError;
use crate::load::{Loaded, load};
use crate::memory::DEFAULT_CTX;
use crate::style::Style;
use crate::verify_hybrid::{argmax, l2};
pub(crate) use crate::verify_reference::{Case, Reference, read_reference, stored_weights};

/// Largest accepted gap between our logits and the float32 reference.
const LOGIT_ABS: f64 = 5e-4;
const LOGIT_REL: f64 = 1e-4;
/// Largest accepted relative gap in a layer's hidden-state size.
const NORM_REL: f64 = 1e-4;

#[derive(Debug)]
pub struct CaseReport {
    pub prompt: String,
    pub steps: usize,
    pub max_logit_diff: f64,
    pub max_norm_diff: f64,
    pub max_tensor_normalized: f64,
    pub max_tensor_rms: f64,
    pub failure: Option<String>,
}

fn check_case(
    model: &Engine,
    tokenizer: &Tokenizer,
    case: &Case,
    cancel: &AtomicBool,
) -> Result<CaseReport, CliError> {
    let mut report = CaseReport {
        prompt: case.prompt.clone(),
        steps: 0,
        max_logit_diff: 0.0,
        max_norm_diff: 0.0,
        max_tensor_normalized: 0.0,
        max_tensor_rms: 0.0,
        failure: None,
    };
    let ids = tokenizer.encode(&case.prompt)?;
    if ids != case.prompt_ids {
        report.failure = Some(format!(
            "tokenizer disagrees with the reference: {ids:?} vs {:?}",
            case.prompt_ids
        ));
        return Ok(report);
    }
    if cancel.load(Ordering::SeqCst) {
        return Err(CliError::Cancelled);
    }
    let capacity = ids.len().saturating_add(case.steps.len()).saturating_add(1);
    let mut kv: EngineCache = model.new_cache(capacity).map_err(CliError::Model)?;
    let layers = case.layer_norms.len();
    let mut norms = vec![0.0_f64; layers];
    let hybrid = matches!(
        model,
        Engine::Qwen35(_) | Engine::Qwen35Cuda(_) | Engine::Granite(_) | Engine::GraniteCuda(_)
    );
    let mut actual_layers = Vec::new();
    let mut routers = Vec::new();
    let mut tops = Vec::new();
    let mut experts = Vec::new();
    let mut logits = {
        let mut observer = |index: usize, hidden: &[f32]| {
            if let Some(slot) = norms.get_mut(index) {
                *slot = l2(hidden);
            }
            if hybrid {
                actual_layers.push(hidden.to_vec());
            }
        };
        match (model, &mut kv) {
            (Engine::Granite(m), EngineCache::Cpu(c)) => {
                let last = ids.len().saturating_sub(1);
                let mut route =
                    |_: usize, position: usize, values: &[f32], ids: &[usize], output: &[f32]| {
                        if position == last {
                            routers.push(values.to_vec());
                            tops.push(ids.to_vec());
                            experts.push(output.to_vec());
                        }
                    };
                m.forward_detailed(&ids, c, Some(&mut observer), Some(&mut route))?
            }
            (Engine::GraniteCuda(m), EngineCache::GraniteCuda(c)) => {
                let last = ids.len().saturating_sub(1);
                let mut route =
                    |_: usize, position: usize, values: &[f32], ids: &[usize], output: &[f32]| {
                        if position == last {
                            routers.push(values.to_vec());
                            tops.push(ids.to_vec());
                            experts.push(output.to_vec());
                        }
                    };
                m.forward_detailed(&ids, c, Some(&mut observer), Some(&mut route))?
            }
            (Engine::Qwen35(m), EngineCache::Qwen35(c)) => {
                m.forward_full_observed(&ids, c, Some(&mut observer))?
            }
            _ => model.forward_observed(&ids, &mut kv, Some(&mut observer))?,
        }
    };
    if hybrid {
        if actual_layers.len() != case.layer_values.len() {
            report.failure = Some("missing full hybrid layer references".into());
            return Ok(report);
        }
        for (i, (actual, expected)) in actual_layers.iter().zip(&case.layer_values).enumerate() {
            match crate::verify_hybrid::compare(actual, expected) {
                Ok((_, normalized, rms)) => {
                    report.max_tensor_normalized = report.max_tensor_normalized.max(normalized);
                    report.max_tensor_rms = report.max_tensor_rms.max(rms);
                }
                Err(e) => {
                    report.failure = Some(format!("first incorrect layer {i}: {e}"));
                    return Ok(report);
                }
            }
        }
    }
    if matches!(model, Engine::Granite(_) | Engine::GraniteCuda(_)) {
        match crate::verify_hybrid::compare_granite(
            (&routers, &tops, &experts),
            (&case.router_values, &case.router_top, &case.expert_values),
            model.config().layers,
        ) {
            Ok((normalized, rms)) => {
                report.max_tensor_normalized = report.max_tensor_normalized.max(normalized);
                report.max_tensor_rms = report.max_tensor_rms.max(rms);
            }
            Err(e) => {
                report.failure = Some(e);
                return Ok(report);
            }
        }
    }
    for (layer, (ours, theirs)) in norms.iter().zip(&case.layer_norms).enumerate() {
        let diff = (ours - theirs).abs() / theirs.abs().max(f64::EPSILON);
        report.max_norm_diff = report.max_norm_diff.max(diff);
        if diff > NORM_REL && report.failure.is_none() {
            report.failure = Some(format!(
                "layer {layer} output differs from the reference by {diff:.2e} (size {ours:.5} vs {theirs:.5})"
            ));
        }
    }
    if report.failure.is_some() {
        return Ok(report);
    }
    for (index, step) in case.steps.iter().enumerate() {
        if report.failure.is_some() {
            break;
        }
        if cancel.load(Ordering::SeqCst) {
            return Err(CliError::Cancelled);
        }
        if index > 0 {
            let previous = case
                .steps
                .get(index.saturating_sub(1))
                .map_or(0, |s| s.token);
            logits = match (model, &mut kv) {
                (Engine::Granite(m), EngineCache::Cpu(c)) if !step.router_top.is_empty() => {
                    tops.clear();
                    experts.clear();
                    actual_layers.clear();
                    let mut route =
                        |_: usize, _: usize, _: &[f32], ids: &[usize], output: &[f32]| {
                            tops.push(ids.to_vec());
                            experts.push(output.to_vec());
                        };
                    let mut watch = |_: usize, hidden: &[f32]| actual_layers.push(hidden.to_vec());
                    m.forward_detailed(&[previous], c, Some(&mut watch), Some(&mut route))?
                }
                (Engine::GraniteCuda(m), EngineCache::GraniteCuda(c))
                    if !step.router_top.is_empty() =>
                {
                    tops.clear();
                    experts.clear();
                    actual_layers.clear();
                    let mut route =
                        |_: usize, _: usize, _: &[f32], ids: &[usize], output: &[f32]| {
                            tops.push(ids.to_vec());
                            experts.push(output.to_vec());
                        };
                    let mut watch = |_: usize, hidden: &[f32]| actual_layers.push(hidden.to_vec());
                    m.forward_detailed(&[previous], c, Some(&mut watch), Some(&mut route))?
                }
                _ => model
                    .forward(&[previous], &mut kv)
                    .map_err(CliError::Model)?,
            };
        }
        if matches!(model, Engine::Granite(_) | Engine::GraniteCuda(_))
            && !step.router_top.is_empty()
            && tops != step.router_top
        {
            report.failure = Some(format!("step {index}: Granite top-k routing differs"));
            break;
        }
        if !step.layer_values.is_empty() {
            for (layer, (actual, expected)) in
                actual_layers.iter().zip(&step.layer_values).enumerate()
            {
                if let Err(e) = crate::verify_hybrid::compare(actual, expected) {
                    report.failure =
                        Some(format!("step {index}: first incorrect layer {layer}: {e}"));
                    break;
                }
            }
        }
        if !step.expert_values.is_empty() && report.failure.is_none() {
            for (layer, (actual, expected)) in experts.iter().zip(&step.expert_values).enumerate() {
                if let Err(e) = crate::verify_hybrid::compare(actual, expected) {
                    report.failure = Some(format!(
                        "step {index}: first incorrect expert layer {layer}: {e}"
                    ));
                    break;
                }
            }
        }
        if report.failure.is_some() {
            break;
        }
        if hybrid {
            match crate::verify_hybrid::compare(&logits, &step.logits) {
                Ok((max, normalized, rms)) => {
                    report.max_logit_diff = report.max_logit_diff.max(max);
                    report.max_tensor_normalized = report.max_tensor_normalized.max(normalized);
                    report.max_tensor_rms = report.max_tensor_rms.max(rms);
                }
                Err(e) => {
                    report.failure = Some(format!("step {index}: full logits: {e}"));
                    break;
                }
            }
        }
        for (id, expected) in &step.top {
            let ours = logits.get(*id as usize).map_or(f64::NAN, |v| f64::from(*v));
            let diff = (ours - expected).abs();
            report.max_logit_diff = report.max_logit_diff.max(diff);
            let tolerance = LOGIT_ABS + LOGIT_REL * expected.abs();
            if (diff.is_nan() || diff > tolerance) && report.failure.is_none() {
                report.failure = Some(format!(
                    "step {index}: logit of token {id} is {ours:.5}, reference says {expected:.5}"
                ));
            }
        }
        if argmax(&logits) != Some(step.token) && report.failure.is_none() {
            report.failure = Some(format!(
                "step {index}: we would pick token {:?}, the reference picks {}",
                argmax(&logits),
                step.token
            ));
        }
        report.steps = report.steps.saturating_add(1);
    }
    Ok(report)
}

/// Whether `check_all` also compares whole tensors for this model.
pub(crate) fn compares_tensors(model: &Engine) -> bool {
    matches!(
        model,
        Engine::Qwen35(_) | Engine::Qwen35Cuda(_) | Engine::Granite(_) | Engine::GraniteCuda(_)
    )
}

/// Runs every case of `reference` on the loaded model, handing each report to `shown` as it is done. True if any
/// case failed.
pub(crate) fn check_all(
    loaded: &Loaded,
    reference: &Reference,
    cancel: &AtomicBool,
    mut shown: impl FnMut(usize, &CaseReport),
) -> Result<bool, CliError> {
    let mut failed = false;
    for (n, case) in reference.cases.iter().enumerate() {
        let report = check_case(&loaded.model, &loaded.dir.tokenizer, case, cancel)?;
        failed |= report.failure.is_some();
        shown(n, &report);
    }
    Ok(failed)
}

pub fn run(
    model_dir: &Path,
    reference_path: &Path,
    backend: BackendChoice,
    precision: Precision,
    style: &Style,
) -> Result<(), CliError> {
    let reference = read_reference(reference_path)?;
    let ctx = reference.needed_context();
    let loaded = load(
        model_dir,
        Some(ctx),
        DEFAULT_CTX,
        backend,
        precision,
        style,
        true,
    )?;
    println!(
        "verifying {} on {} against float32 reference output ({}, float32 math)\n",
        style.bold(&loaded.name),
        loaded.model.backend_name(),
        precision
            .weights
            .map_or_else(|| "bf16 weights".into(), |q| q.label())
    );
    let total = reference.cases();
    let failed = check_all(&loaded, &reference, &AtomicBool::new(false), |n, report| {
        let label = format!("case {}/{total}  {:?}", n.saturating_add(1), report.prompt);
        match &report.failure {
            None => println!(
                "  {} {label}\n      {} steps, token choices match, max logit diff {:.2e}, max layer diff {:.2e}",
                style.green("pass"),
                report.steps,
                report.max_logit_diff,
                report.max_norm_diff
            ),
            Some(why) => println!("  {} {label}\n      {why}", style.red("FAIL")),
        }
        if compares_tensors(&loaded.model) {
            println!(
                "      full tensors: max normalized {:.3e}, relative RMS {:.3e}",
                report.max_tensor_normalized, report.max_tensor_rms
            );
        }
    })?;
    if failed {
        println!("\n{}", style.red("result: FAIL"));
        return Err(CliError::VerifyFailed);
    }
    println!("\n{}", style.green("result: PASS"));
    Ok(())
}

#[cfg(test)]
#[path = "verify_tests.rs"]
mod tests;
