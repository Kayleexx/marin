use std::fs;
use std::path::Path;

use loader::Tokenizer;
use model::{BackendChoice, Engine, EngineCache, Precision};
use runtime::CausalLm;
use serde::Deserialize;

use crate::CliError;
use crate::load::load;
use crate::memory::DEFAULT_CTX;
use crate::style::Style;

/// Largest accepted gap between our logits and the float32 reference.
const LOGIT_ABS: f64 = 5e-4;
const LOGIT_REL: f64 = 1e-4;
/// Largest accepted relative gap in a layer's hidden-state size.
const NORM_REL: f64 = 1e-4;

#[derive(Deserialize)]
struct Reference {
    format: u32,
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    prompt: String,
    prompt_ids: Vec<u32>,
    layer_norms: Vec<f64>,
    #[serde(default)]
    layer_values: Vec<Vec<f32>>,
    steps: Vec<Step>,
}

#[derive(Deserialize)]
struct Step {
    token: u32,
    top: Vec<(u32, f64)>,
    #[serde(default)]
    logits: Vec<f32>,
}

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

fn read_reference(path: &Path) -> Result<Reference, CliError> {
    let fail = |reason: String| CliError::Reference {
        path: path.display().to_string(),
        reason,
    };
    let text = fs::read_to_string(path).map_err(|e| fail(e.to_string()))?;
    let reference: Reference = serde_json::from_str(&text).map_err(|e| fail(e.to_string()))?;
    if reference.format != 1 {
        return Err(fail(format!("unknown format {}", reference.format)));
    }
    Ok(reference)
}

fn l2(values: &[f32]) -> f64 {
    values
        .iter()
        .map(|v| f64::from(*v).powi(2))
        .sum::<f64>()
        .sqrt()
}

fn argmax(logits: &[f32]) -> Option<u32> {
    let (index, _) = logits
        .iter()
        .enumerate()
        .reduce(|best, next| if next.1 > best.1 { next } else { best })?;
    u32::try_from(index).ok()
}

fn check_case(model: &Engine, tokenizer: &Tokenizer, case: &Case) -> Result<CaseReport, CliError> {
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
    let capacity = ids.len().saturating_add(case.steps.len()).saturating_add(1);
    let mut kv: EngineCache = model.new_cache(capacity).map_err(CliError::Model)?;
    let layers = case.layer_norms.len();
    let mut norms = vec![0.0_f64; layers];
    let hybrid = matches!(model, Engine::Qwen35(_) | Engine::Qwen35Cuda(_));
    let mut actual_layers = Vec::new();
    let mut observer = |index: usize, hidden: &[f32]| {
        if let Some(slot) = norms.get_mut(index) {
            *slot = l2(hidden);
        }
        if hybrid {
            actual_layers.push(hidden.to_vec());
        }
    };
    let mut logits = match (model, &mut kv) {
        (Engine::Qwen35(m), EngineCache::Qwen35(c)) => {
            m.forward_full_observed(&ids, c, Some(&mut observer))?
        }
        _ => model.forward_observed(&ids, &mut kv, Some(&mut observer))?,
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
        if index > 0 {
            let previous = case
                .steps
                .get(index.saturating_sub(1))
                .map_or(0, |s| s.token);
            logits = model
                .forward(&[previous], &mut kv)
                .map_err(CliError::Model)?;
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

pub fn run(
    model_dir: &Path,
    reference_path: &Path,
    backend: BackendChoice,
    precision: Precision,
    style: &Style,
) -> Result<(), CliError> {
    let reference = read_reference(reference_path)?;
    let ctx = reference
        .cases
        .iter()
        .map(|c| {
            c.prompt_ids
                .len()
                .saturating_add(c.steps.len())
                .saturating_add(1)
        })
        .max()
        .unwrap_or(1);
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
    let mut failed = false;
    for (n, case) in reference.cases.iter().enumerate() {
        let report = check_case(&loaded.model, &loaded.dir.tokenizer, case)?;
        let label = format!(
            "case {}/{}  {:?}",
            n.saturating_add(1),
            reference.cases.len(),
            report.prompt
        );
        match &report.failure {
            None => println!(
                "  {} {label}\n      {} steps, token choices match, max logit diff {:.2e}, max layer diff {:.2e}",
                style.green("pass"),
                report.steps,
                report.max_logit_diff,
                report.max_norm_diff
            ),
            Some(why) => {
                failed = true;
                println!("  {} {label}\n      {why}", style.red("FAIL"));
            }
        }
        if matches!(loaded.model, Engine::Qwen35(_) | Engine::Qwen35Cuda(_)) {
            println!(
                "      full tensors: max normalized {:.3e}, relative RMS {:.3e}",
                report.max_tensor_normalized, report.max_tensor_rms
            );
        }
    }
    if failed {
        println!("\n{}", style.red("result: FAIL"));
        return Err(CliError::VerifyFailed);
    }
    println!("\n{}", style.green("result: PASS"));
    Ok(())
}

#[cfg(test)]
mod tests {
    use std::path::PathBuf;

    use loader::ModelDir;
    use model::Backend;

    use super::*;

    fn fixture() -> PathBuf {
        Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen3")
    }

    #[test]
    fn tiny_model_matches_the_huggingface_reference() {
        let dir = ModelDir::open(&fixture()).unwrap();
        let model = Engine::load(&dir, Backend::Cpu, None, |_, _| {}).unwrap();
        let reference = read_reference(&fixture().join("reference.json")).unwrap();
        assert!(!reference.cases.is_empty());
        for case in &reference.cases {
            let report = check_case(&model, &dir.tokenizer, case).unwrap();
            assert_eq!(report.failure, None, "{report:?}");
            assert!(report.steps > 0);
            eprintln!(
                "{:?}: max logit diff {:.3e}, max layer diff {:.3e}",
                report.prompt, report.max_logit_diff, report.max_norm_diff
            );
        }
    }

    #[test]
    fn tiny_model_matches_the_huggingface_reference_on_the_gpu() {
        let selected = match Backend::select(BackendChoice::Cuda, 0, 0) {
            Ok(selected) => selected,
            Err(e) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
                eprintln!("skipping GPU test: {e}");
                return;
            }
            Err(e) => panic!("{e}"),
        };
        let dir = ModelDir::open(&fixture()).unwrap();
        let model = Engine::load(&dir, selected.backend, None, |_, _| {}).unwrap();
        let reference = read_reference(&fixture().join("reference.json")).unwrap();
        for case in &reference.cases {
            let report = check_case(&model, &dir.tokenizer, case).unwrap();
            assert_eq!(report.failure, None, "{report:?}");
            eprintln!(
                "gpu {:?}: max logit diff {:.3e}, max layer diff {:.3e}",
                report.prompt, report.max_logit_diff, report.max_norm_diff
            );
        }
    }

    fn load_fixture() -> (ModelDir, Engine, Reference) {
        let dir = ModelDir::open(&fixture()).unwrap();
        let model = Engine::load(&dir, Backend::Cpu, None, |_, _| {}).unwrap();
        let reference = read_reference(&fixture().join("reference.json")).unwrap();
        (dir, model, reference)
    }

    fn failure(case: &Case, dir: &ModelDir, model: &Engine) -> String {
        check_case(model, &dir.tokenizer, case)
            .unwrap()
            .failure
            .unwrap_or_default()
    }

    #[test]
    fn a_wrong_logit_in_the_reference_is_caught() {
        let (dir, model, mut reference) = load_fixture();
        let case = reference.cases.first_mut().unwrap();
        case.steps.get_mut(1).unwrap().top.get_mut(0).unwrap().1 += 0.5;
        assert!(failure(case, &dir, &model).contains("step 1: logit"));
    }

    #[test]
    fn a_wrong_greedy_token_in_the_reference_is_caught() {
        let (dir, model, mut reference) = load_fixture();
        let case = reference.cases.first_mut().unwrap();
        let step = case.steps.first_mut().unwrap();
        step.token = step.token.wrapping_add(1);
        assert!(failure(case, &dir, &model).contains("step 0"));
    }

    #[test]
    fn a_wrong_layer_size_in_the_reference_is_caught() {
        let (dir, model, mut reference) = load_fixture();
        let case = reference.cases.first_mut().unwrap();
        *case.layer_norms.get_mut(1).unwrap() *= 1.01;
        assert!(failure(case, &dir, &model).contains("layer 1"));
    }

    #[test]
    fn a_tokenizer_disagreement_is_caught() {
        let (dir, model, mut reference) = load_fixture();
        let case = reference.cases.first_mut().unwrap();
        case.prompt_ids.push(1);
        assert!(failure(case, &dir, &model).contains("tokenizer disagrees"));
    }

    /// `verify` is a greedy reference comparison: it is not a sampler, so no sampler setting or seed can matter,
    /// and it agrees with the sampler's own temperature-0 path.
    #[test]
    fn verification_is_greedy_and_ignores_every_sampling_setting() {
        use sampler::{Sampler, SamplerConfig};
        let logits = [0.5_f32, 3.0, 1.0, 3.0, -2.0];
        assert_eq!(argmax(&logits), Some(1), "ties go to the lowest id");
        for (top_k, top_p, seed) in [(0, 1.0, 1), (20, 0.8, 2), (2, 0.1, 99)] {
            let config = SamplerConfig {
                temperature: 0.0,
                top_k,
                top_p,
                seed,
            };
            let picked = Sampler::new(config).unwrap().sample(&logits, 5).unwrap();
            assert_eq!(Some(picked), argmax(&logits), "{top_k} {top_p} {seed}");
        }
    }
}
