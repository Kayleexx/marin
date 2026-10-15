//! Run with MARIN_GRANITE_MODEL pointing at the official checkpoint.

use std::error::Error;
use std::path::Path;

use kv::KvCache;
use loader::ModelDir;
use model::granite::{Granite, GraniteGpu};
use model::{QuantBits, QuantScheme};
use runtime::{CausalLm, SequenceCache};

type Result<T> = std::result::Result<T, Box<dyn Error>>;

#[test]
#[ignore = "requires official Granite BF16 and exported Q8 checkpoints"]
fn official_q8_source_and_direct_load_match() -> Result<()> {
    let source = ModelDir::open(Path::new(&std::env::var("MARIN_GRANITE_MODEL")?))?;
    let packed = ModelDir::open(Path::new(&std::env::var("MARIN_GRANITE_PACKED")?))?;
    let scheme = QuantScheme::new(QuantBits::Int8, 128)?;
    let converted = Granite::load_quantized(&source, Some(scheme), |_, _| {})?;
    let direct = Granite::load_quantized(&packed, Some(scheme), |_, _| {})?;
    let prompt = "<|start_of_role|>user<|end_of_role|>Reply with a brief greeting.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
    let ids = source.tokenizer.encode(prompt)?;
    let mut left = converted.new_cache(64)?;
    let mut right = direct.new_cache(64)?;
    let a = converted.forward(&ids, &mut left)?;
    let b = direct.forward(&ids, &mut right)?;
    assert_eq!(a, b);
    let token = a
        .iter()
        .enumerate()
        .max_by(|x, y| x.1.total_cmp(y.1))
        .ok_or("empty logits")?
        .0 as u32;
    assert_eq!(
        converted.forward(&[token], &mut left)?,
        direct.forward(&[token], &mut right)?
    );
    left.clear();
    right.clear();
    assert_eq!(
        converted.forward(&ids, &mut left)?,
        direct.forward(&ids, &mut right)?
    );
    assert!(Granite::load_quantized(&packed, None, |_, _| {}).is_err());
    Ok(())
}

fn close(a: &[f32], b: &[f32]) {
    assert_eq!(a.len(), b.len());
    let max_ref = b.iter().map(|x| x.abs()).fold(1.0_f32, f32::max);
    let max_error = a
        .iter()
        .zip(b)
        .map(|(x, y)| (x - y).abs())
        .fold(0.0_f32, f32::max);
    let error = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(x - y).powi(2))
        .sum::<f64>();
    let magnitude = b.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>();
    assert!(max_error / max_ref <= 1e-4, "{max_error} / {max_ref}");
    assert!((error / magnitude.max(1e-12)).sqrt() <= 1e-5);
}

#[test]
#[ignore = "requires the official Granite checkpoint"]
fn official_state_and_routing() -> Result<()> {
    let path = std::env::var("MARIN_GRANITE_MODEL")?;
    let dir = ModelDir::open(Path::new(&path))?;
    let model = Granite::load(&dir, |_, _| {})?;
    let prompts = [
        ("prose-short", "<|start_of_role|>user<|end_of_role|>Reply with a brief greeting.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>".to_string()),
        ("code-short", "<|start_of_role|>user<|end_of_role|>Write a Python function that adds two integers.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>".to_string()),
        ("prose-long", format!("<|start_of_role|>user<|end_of_role|>{}Summarize the workshop in one sentence.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>", "The workshop builds small wooden clocks. ".repeat(30))),
        ("code-long", format!("<|start_of_role|>user<|end_of_role|>{}Write a Python function that adds two integers.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>", "Please explain the inputs, output and edge cases. ".repeat(20))),
    ];
    for (label, prompt) in prompts {
        let ids = dir.tokenizer.encode(&prompt)?;
        let mut whole = model.new_cache(ids.len() + 16)?;
        let mut per_layer = vec![Vec::<Vec<usize>>::new(); model.config().layers];
        let mut route = |layer: usize, _: usize, _: &[f32], selected: &[usize], _: &[f32]| {
            if let Some(rows) = per_layer.get_mut(layer) {
                rows.push(selected.to_vec());
            }
        };
        let original = model.forward_detailed(&ids, &mut whole, None, Some(&mut route))?;
        if label == "prose-long" {
            let mut batched = model.new_cache(ids.len() + 1)?;
            let mut tokenwise = model.new_cache(ids.len() + 1)?;
            let packed = model.forward(&ids[..32], &mut batched)?;
            let mut serial = Vec::new();
            for &id in &ids[..32] {
                serial = model.forward(&[id], &mut tokenwise)?;
            }
            close(&packed, &serial);
            assert_eq!(batched.len(), tokenwise.len());
            let mut malformed =
                KvCache::new(1, model.config().kv_heads, model.config().head_dim, 32)?;
            assert!(model.forward(&ids[..32], &mut malformed).is_err());
            assert_eq!(malformed.len(), 0);
        }
        if label == "prose-short" || label == "prose-long" {
            let mut split = model.new_cache(ids.len() + 16)?;
            let midpoint = ids.len() / 2;
            model.forward(&ids[..midpoint], &mut split)?;
            let replay = model.forward(&ids[midpoint..], &mut split)?;
            close(&original, &replay);
            assert_eq!(whole.len(), ids.len());
            assert_eq!(split.len(), ids.len());
            let next = original
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| u32::try_from(i))
                .transpose()?
                .ok_or("empty logits")?;
            close(
                &model.forward(&[next], &mut whole)?,
                &model.forward(&[next], &mut split)?,
            );
            whole.clear();
            close(&model.forward(&ids, &mut whole)?, &original);
            let mut independent = model.new_cache(ids.len() + 16)?;
            close(&model.forward(&ids, &mut independent)?, &original);
            let mut malformed =
                KvCache::new(1, model.config().kv_heads, model.config().head_dim, 16)?;
            assert!(model.forward(&ids[..1], &mut malformed).is_err());
            assert_eq!(malformed.len(), 0);
        }

        let mut logits = original;
        for _ in 0..8 {
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(i, _)| u32::try_from(i))
                .transpose()?
                .ok_or("empty logits")?;
            let mut decode_route =
                |layer: usize, _: usize, _: &[f32], selected: &[usize], _: &[f32]| {
                    if let Some(rows) = per_layer.get_mut(layer) {
                        rows.push(selected.to_vec());
                    }
                };
            logits = model.forward_detailed(&[next], &mut whole, None, Some(&mut decode_route))?;
        }

        let mut uniques = Vec::new();
        let mut hits = [0_usize; 3];
        let mut decode_hits = [0_usize; 3];
        let mut frequencies = vec![0_usize; 32];
        let mut total = 0_usize;
        for rows in &per_layer {
            let mut seen = std::collections::BTreeSet::new();
            let mut caches = [Vec::<usize>::new(), Vec::new(), Vec::new()];
            for (token_index, selected) in rows.iter().enumerate() {
                for id in selected {
                    seen.insert(*id);
                    if let Some(count) = frequencies.get_mut(*id) {
                        *count += 1;
                    }
                    for (index, cache) in caches.iter_mut().enumerate() {
                        let size = [8, 16, 22][index];
                        if let Some(found) = cache.iter().position(|item| item == id) {
                            cache.remove(found);
                            hits[index] += 1;
                            if token_index >= ids.len() {
                                decode_hits[index] += 1;
                            }
                        }
                        cache.push(*id);
                        if cache.len() > size {
                            cache.remove(0);
                        }
                    }
                    total += 1;
                }
            }
            uniques.push(seen.len());
        }
        let min = uniques.iter().copied().min().unwrap_or(0);
        let max = uniques.iter().copied().max().unwrap_or(0);
        let mean = uniques.iter().sum::<usize>() as f64 / uniques.len() as f64;
        let low = frequencies.iter().copied().min().unwrap_or(0);
        let high = frequencies.iter().copied().max().unwrap_or(0);
        println!(
            "{label}: prompt={} unique_experts/layer={min}..{max} mean={mean:.2} expert_frequency={low}..{high} selected={total} lru_hits_8_16_22={hits:?} decode_hits={decode_hits:?}/1536",
            ids.len()
        );
        println!("{label}: per_layer_unique={uniques:?} expert_frequencies={frequencies:?}");
    }
    Ok(())
}

#[test]
#[ignore = "requires the official Granite checkpoint and CUDA GPU"]
fn official_cuda_state_and_routing() -> Result<()> {
    let dir = ModelDir::open(Path::new(&std::env::var("MARIN_GRANITE_MODEL")?))?;
    let gpu = GraniteGpu::load(&dir, cuda::Gpu::new()?, |_, _| {})?;
    let prompt = format!(
        "<|start_of_role|>user<|end_of_role|>{}Summarize the workshop in one sentence.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>",
        "The workshop builds small wooden clocks. ".repeat(13)
    );
    let ids = dir.tokenizer.encode(&prompt)?;
    let mut whole = gpu.new_cache(ids.len() + 8)?;
    let mut routes = Vec::new();
    let mut callback = |layer: usize, _: usize, _: &[f32], selected: &[usize], _: &[f32]| {
        routes.push((layer, selected.to_vec()));
    };
    let original = gpu.forward_detailed(&ids, &mut whole, None, Some(&mut callback))?;
    assert_eq!(routes.len(), gpu.config().layers);
    assert!(routes.iter().all(|(_, selected)| selected.len() == 8));
    let mut split = gpu.new_cache(ids.len() + 8)?;
    let middle = ids.len() / 2;
    gpu.forward(&ids[..middle], &mut split)?;
    close(&gpu.forward(&ids[middle..], &mut split)?, &original);
    assert_eq!(whole.used(), ids.len());
    assert_eq!(split.used(), ids.len());
    let next = original
        .iter()
        .enumerate()
        .max_by(|a, b| a.1.total_cmp(b.1))
        .map(|(i, _)| u32::try_from(i))
        .transpose()?
        .ok_or("empty logits")?;
    let decoded = gpu.forward(&[next], &mut whole)?;
    close(&decoded, &gpu.forward(&[next], &mut split)?);
    whole.truncate(ids.len())?;
    close(&gpu.forward(&[next], &mut whole)?, &decoded);
    whole.clear();
    close(&gpu.forward(&ids, &mut whole)?, &original);
    let mut independent = gpu.new_cache(ids.len() + 8)?;
    close(&gpu.forward(&ids, &mut independent)?, &original);
    let held = whole.used();
    assert!(gpu.forward(&[u32::MAX], &mut whole).is_err());
    assert_eq!(whole.used(), held);
    let mut invalid_late = vec![ids[0]; 513];
    invalid_late.push(u32::MAX);
    let mut rollback = gpu.new_cache(520)?;
    assert!(gpu.forward(&invalid_late, &mut rollback).is_err());
    assert_eq!(rollback.used(), 0);
    let code = "<|start_of_role|>user<|end_of_role|>Write a Python function that adds two integers.<|end_of_text|>\n<|start_of_role|>assistant<|end_of_role|>";
    let code_ids = dir.tokenizer.encode(code)?;
    let mut whole_code = gpu.new_cache(code_ids.len() + 4)?;
    let code_logits = gpu.forward(&code_ids, &mut whole_code)?;
    let mut split_code = gpu.new_cache(code_ids.len() + 4)?;
    gpu.forward(&code_ids[..8], &mut split_code)?;
    close(&gpu.forward(&code_ids[8..], &mut split_code)?, &code_logits);
    whole_code.clear();
    close(&gpu.forward(&code_ids, &mut whole_code)?, &code_logits);
    Ok(())
}

#[test]
#[ignore = "requires official Granite checkpoint, HF reference and CUDA GPU"]
fn official_cuda_greedy_matches_hf() -> Result<()> {
    let dir = ModelDir::open(Path::new(&std::env::var("MARIN_GRANITE_MODEL")?))?;
    let reference: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        std::env::var("MARIN_GRANITE_REFERENCE")?,
    )?)?;
    let cases = reference["cases"]
        .as_array()
        .ok_or("HF cases are missing")?;
    let gpu = GraniteGpu::load(&dir, cuda::Gpu::new()?, |_, _| {})?;
    let mut matched = 0;
    for case in cases {
        let prompt = case["prompt"].as_str().ok_or("HF prompt is missing")?;
        let ids = dir.tokenizer.encode(prompt)?;
        let steps = case["steps"].as_array().ok_or("HF steps are missing")?;
        let mut cache = gpu.new_cache(ids.len() + steps.len() + 1)?;
        let mut logits = gpu.forward(&ids, &mut cache)?;
        for step in steps {
            let expected = step["token"].as_u64().ok_or("HF token is missing")?;
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index as u64)
                .ok_or("empty logits")?;
            assert_eq!(next, expected, "prompt {prompt}");
            matched += 1;
            logits = gpu.forward(&[u32::try_from(next)?], &mut cache)?;
        }
    }
    assert_eq!(matched, 24);
    Ok(())
}

#[test]
#[ignore = "requires official Granite Q8 artifact and independent quantized HF reference"]
fn official_q8_greedy_matches_quantized_hf() -> Result<()> {
    let dir = ModelDir::open(Path::new(&std::env::var("MARIN_GRANITE_PACKED")?))?;
    let reference: serde_json::Value = serde_json::from_str(&std::fs::read_to_string(
        std::env::var("MARIN_GRANITE_REFERENCE_Q8")?,
    )?)?;
    let cases = reference["cases"]
        .as_array()
        .ok_or("HF cases are missing")?;
    let scheme = QuantScheme::new(QuantBits::Int8, 128)?;
    let model = Granite::load_quantized(&dir, Some(scheme), |_, _| {})?;
    let mut matched = 0;
    for case in cases {
        let prompt = case["prompt"].as_str().ok_or("HF prompt is missing")?;
        let ids = dir.tokenizer.encode(prompt)?;
        let steps = case["steps"].as_array().ok_or("HF steps are missing")?;
        let mut cache = model.new_cache(ids.len() + steps.len() + 1)?;
        let mut logits = model.forward(&ids, &mut cache)?;
        let prompt_logits = logits.clone();
        for step in steps {
            let expected = step["token"].as_u64().ok_or("HF token is missing")?;
            let next = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(index, _)| index as u64)
                .ok_or("empty logits")?;
            assert_eq!(next, expected, "prompt {prompt}");
            matched += 1;
            logits = model.forward(&[u32::try_from(next)?], &mut cache)?;
        }
        cache.clear();
        close(&model.forward(&ids, &mut cache)?, &prompt_logits);
        let mut split = model.new_cache(ids.len() + steps.len() + 1)?;
        let middle = ids.len() / 2;
        model.forward(&ids[..middle], &mut split)?;
        close(&model.forward(&ids[middle..], &mut split)?, &prompt_logits);
        let mut independent = model.new_cache(ids.len() + steps.len() + 1)?;
        close(&model.forward(&ids, &mut independent)?, &prompt_logits);
        let held = cache.len();
        assert!(model.forward(&[u32::MAX], &mut cache).is_err());
        assert_eq!(cache.len(), held);
    }
    assert_eq!(matched, 24);
    Ok(())
}
