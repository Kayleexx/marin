use super::gpu::Workspace;
use super::gpu_weights::Mixer;
use super::{Qwen35, Qwen35Gpu};
use cuda::{DeltaDims, DeltaInputs, Gpu};
use loader::ModelDir;
use runtime::{CausalLm, SequenceCache};
use std::collections::BTreeMap;

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

fn close(actual: &[f32], expected: &[f32], name: &str) {
    assert_eq!(actual.len(), expected.len(), "{name} shape");
    let max = expected.iter().map(|x| x.abs()).fold(1.0_f32, f32::max);
    let diff = actual
        .iter()
        .zip(expected)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(diff / max <= 1e-4, "{name}: max normalized {}", diff / max);
}

#[test]
fn delta_device_steps_match_cpu_reference() -> Result {
    let gpu = match Gpu::new() {
        Ok(gpu) => gpu,
        Err(e) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
            eprintln!("skipping Qwen3.5 GPU test: {e}");
            return Ok(());
        }
        Err(e) => return Err(e.into()),
    };
    let path =
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35");
    let dir = ModelDir::open(&path)?;
    let cpu = Qwen35::load(&dir, |_, _| {})?;
    let model = Qwen35Gpu::load(&dir, gpu, None, |_, _| {})?;
    let c = &model.config.block;
    let x: Vec<f32> = (0..c.hidden_size)
        .map(|i| (i as f32 - 3.0) * 0.17)
        .collect();
    let block = cpu.blocks.first().ok_or("no block")?;
    let mut state = block.new_state(8, usize::MAX)?;
    let mut seen = BTreeMap::new();
    block.forward(
        &x,
        &mut state,
        usize::MAX,
        Some(&mut |name, values| {
            seen.insert(name.to_string(), values.to_vec());
        }),
    )?;
    let layer = model.layers.first().ok_or("no GPU layer")?;
    let Mixer::Delta(w) = &layer.mixer else {
        return Err("expected Delta".into());
    };
    let gpu = &model.gpu;
    let mut ws = Workspace::new(gpu, &model.config, (1, 1))?;
    ws.hidden = gpu.upload(&x)?;
    gpu.rms_norm_into(
        &ws.hidden,
        &layer.common.input_norm,
        &mut ws.normed,
        c.rms_norm_eps,
    )?;
    close(
        &gpu.download(&ws.normed)?,
        &seen["input_norm"],
        "input_norm",
    );
    for (weight, out, name) in [
        (&w.qkv, &mut ws.qkv, "qkv"),
        (&w.z, &mut ws.z, "z"),
        (&w.a, &mut ws.a, "a"),
        (&w.b, &mut ws.b, "b"),
    ] {
        weight.matmul_precise(gpu, &ws.normed, 1, out)?;
        close(&gpu.download(out)?, &seen[name], name);
    }
    let dims = DeltaDims {
        key_heads: c.linear_num_key_heads,
        value_heads: c.linear_num_value_heads,
        key_dim: c.linear_key_head_dim,
        value_dim: c.linear_value_head_dim,
        kernel: c.linear_conv_kernel_dim,
    };
    let mut conv = gpu.zeros(w.conv.len())?;
    let mut recurrent =
        gpu.zeros(c.linear_num_value_heads * c.linear_key_head_dim * c.linear_value_head_dim)?;
    gpu.q35_convolution(&ws.qkv, &w.conv, &mut conv, &mut ws.mixed, dims, 1)?;
    close(&gpu.download(&ws.mixed)?, &seen["conv"], "conv");
    gpu.q35_recurrent(
        DeltaInputs {
            mixed: &ws.mixed,
            z: &ws.z,
            a: &ws.a,
            b: &ws.b,
            a_log: &w.a_log,
            dt_bias: &w.dt_bias,
            norm: &w.norm,
        },
        &mut recurrent,
        &mut ws.core,
        dims,
        1,
        c.rms_norm_eps,
    )?;
    close(
        &gpu.download(&conv)?,
        state.convolution().ok_or("CPU conv")?,
        "conv state",
    );
    close(
        &gpu.download(&recurrent)?,
        state.recurrent().ok_or("CPU state")?,
        "recurrent state",
    );
    close(&gpu.download(&ws.core)?, &seen["gated"], "gated");
    Ok(())
}

pub(super) fn contents(model: &Qwen35Gpu, cache: &super::GpuCache) -> Result<(Vec<f32>, Vec<u8>)> {
    let mut state = Vec::new();
    for part in cache.delta.iter().flatten() {
        state.extend(model.gpu.download(&part.conv)?);
        state.extend(model.gpu.download(&part.recurrent)?);
    }
    let mut kv = Vec::new();
    for page in 0..cuda::pages_for(cache.kv.len()) {
        let rows = cache
            .kv
            .len()
            .saturating_sub(page.saturating_mul(cuda::PAGE_TOKENS))
            .min(cuda::PAGE_TOKENS);
        kv.extend(cache.kv.export_page(&model.gpu, page, rows)?);
    }
    Ok((state, kv))
}

#[test]
fn gpu_sessions_replay_reset_and_late_failure_are_transactional() -> Result {
    let fixture = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures");
    let mut cases = vec![
        (fixture.join("tiny-qwen35"), None),
        (
            fixture.join("tiny-qwen35-int8"),
            Some(tensor::QuantScheme::new(tensor::QuantBits::Int8, 32)?),
        ),
    ];
    if let Some(real) = std::env::var_os("MARIN_QWEN35_4B_Q8") {
        cases.push((
            std::path::PathBuf::from(real),
            Some(tensor::QuantScheme::new(tensor::QuantBits::Int8, 128)?),
        ));
    }
    for (path, quant) in cases {
        let gpu = match Gpu::new() {
            Ok(gpu) => gpu,
            Err(e) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
                eprintln!("skipping Qwen3.5 GPU test: {e}");
                return Ok(());
            }
            Err(e) => return Err(e.into()),
        };
        let dir = ModelDir::open(&path)?;
        let mut model = Qwen35Gpu::load(&dir, gpu, quant, |_, _| {})?;
        let ids = [21, 27, 31, 37, 41, 43];
        let mut whole = model.new_cache(64)?;
        let mut split = model.new_cache(64)?;
        let independent = model.new_cache(64)?;
        let original = contents(&model, &independent)?;
        let logits = model.forward(&ids, &mut whole)?;
        let mut incremental = Vec::new();
        for id in ids {
            incremental = model.forward(&[id], &mut split)?;
        }
        close(&logits, &incremental, "whole/tokenwise logits");
        close(
            &contents(&model, &whole)?.0,
            &contents(&model, &split)?.0,
            "whole/tokenwise recurrent state",
        );
        assert_eq!(contents(&model, &independent)?, original);

        let original_history = contents(&model, &whole)?;
        let checkpoint = model.checkpoint_cache(&whole)?;
        assert_eq!(contents(&model, &checkpoint)?, original_history);
        model.forward(&[53], &mut whole)?;
        whole = checkpoint;
        assert_eq!(contents(&model, &whole)?, original_history);
        close(
            &model.forward(&[5], &mut whole)?,
            &model.forward(&[5], &mut split)?,
            "checkpoint continuation",
        );

        whole.truncate(3)?;
        let edited = [21, 27, 31, 50, 51];
        let replay = model.forward(&edited[3..], &mut whole)?;
        let mut fresh = model.new_cache(64)?;
        let expected = model.forward(&edited, &mut fresh)?;
        close(&replay, &expected, "edited replay logits");
        assert_eq!(contents(&model, &whole)?, contents(&model, &fresh)?);
        whole.clear();
        close(
            &model.forward(&edited, &mut whole)?,
            &expected,
            "reset logits",
        );

        let before = contents(&model, &whole)?;
        let ledger = whole.tokens.clone();
        let bad = model
            .gpu
            .upload(&vec![f32::NAN; model.config.block.hidden_size])?;
        let saved = std::mem::replace(&mut model.norm, bad);
        assert!(model.forward(&[52], &mut whole).is_err());
        model.norm = saved;
        assert_eq!(whole.tokens, ledger);
        assert_eq!(contents(&model, &whole)?, before);
        let continuation = model.forward(&[52], &mut whole)?;
        let fresh_continuation = model.forward(&[52], &mut fresh)?;
        close(
            &continuation,
            &fresh_continuation,
            "failure rollback continuation",
        );
        assert_eq!(contents(&model, &whole)?, contents(&model, &fresh)?);

        let mut pair = Vec::new();
        for prompt in [&[21_u32, 27, 31][..], &[11, 12][..]] {
            let (mut live, mut twin) = (model.new_cache(64)?, model.new_cache(64)?);
            model.forward(prompt, &mut live)?;
            model.forward(prompt, &mut twin)?;
            pair.push((live, twin));
        }
        let bad = model
            .gpu
            .upload(&vec![f32::NAN; model.config.block.hidden_size])?;
        let saved = std::mem::replace(&mut model.norm, bad);
        let mut refs: Vec<_> = pair.iter_mut().map(|(live, _)| live).collect();
        assert!(model.forward_batch(&[5, 6], &mut refs).is_err());
        model.norm = saved;
        for ((live, twin), (token, used)) in pair.iter_mut().zip([(5, 3), (6, 2)]) {
            assert_eq!(live.used(), used);
            close(
                &model.forward(&[token], live)?,
                &model.forward(&[token], twin)?,
                "batch failure continuation",
            );
            assert_eq!(contents(&model, live)?, contents(&model, twin)?);
        }
    }
    Ok(())
}
