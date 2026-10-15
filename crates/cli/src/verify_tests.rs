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
