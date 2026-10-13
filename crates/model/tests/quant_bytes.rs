//! What a quantized model takes in memory: counted from the tensors, quantized where the loader quantizes them.

use std::fs;
use std::path::Path;

use loader::Weights;
use model::{QuantBits, QuantScheme, Qwen3Config, weight_bytes};

const CONFIG: &str = r#"{
    "model_type": "qwen3", "hidden_size": 64, "intermediate_size": 128, "num_hidden_layers": 2,
    "num_attention_heads": 4, "num_key_value_heads": 2, "head_dim": 16, "vocab_size": 40,
    "max_position_embeddings": 32, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "tie_word_embeddings": TIED
}"#;

/// (name, shape) of every tensor this config needs.
fn tensors(tied: bool) -> Vec<(String, Vec<usize>)> {
    let mut all = vec![
        ("model.embed_tokens.weight".to_string(), vec![40, 64]),
        ("model.norm.weight".to_string(), vec![64]),
    ];
    if !tied {
        all.push(("lm_head.weight".to_string(), vec![40, 64]));
    }
    for n in 0..2 {
        let layer = |s: &str| format!("model.layers.{n}.{s}");
        all.extend([
            (layer("input_layernorm.weight"), vec![64]),
            (layer("self_attn.q_proj.weight"), vec![64, 64]),
            (layer("self_attn.k_proj.weight"), vec![32, 64]),
            (layer("self_attn.v_proj.weight"), vec![32, 64]),
            (layer("self_attn.o_proj.weight"), vec![64, 64]),
            (layer("self_attn.q_norm.weight"), vec![16]),
            (layer("self_attn.k_norm.weight"), vec![16]),
            (layer("post_attention_layernorm.weight"), vec![64]),
            (layer("mlp.gate_proj.weight"), vec![128, 64]),
            (layer("mlp.up_proj.weight"), vec![128, 64]),
            (layer("mlp.down_proj.weight"), vec![64, 128]),
        ]);
    }
    all
}

fn write(dir: &Path, all: &[(String, Vec<usize>)]) -> std::io::Result<()> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in all {
        let bytes = shape.iter().product::<usize>().saturating_mul(2);
        let end = data.len().saturating_add(bytes);
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [data.len(), end]}),
        );
        data.extend(std::iter::repeat_n(0_u8, bytes));
    }
    let header = serde_json::to_vec(&header).map_err(std::io::Error::other)?;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(&data);
    fs::write(dir.join("model.safetensors"), file)
}

/// The bytes the loader should report, worked out tensor by tensor.
fn expected(all: &[(String, Vec<usize>)], layers: QuantScheme) -> Option<u64> {
    let table = layers.table();
    let mut total = 0_u64;
    for (name, shape) in all {
        let bytes = match shape.as_slice() {
            [len] => u64::try_from(len.checked_mul(4)?).ok()?,
            [rows, cols] if name.ends_with("proj.weight") => layers.matrix_bytes(*rows, *cols)?,
            [rows, cols] => table.matrix_bytes(*rows, *cols)?,
            _ => return None,
        };
        total = total.checked_add(bytes)?;
    }
    Some(total)
}

fn load(
    tied: bool,
    tag: &str,
) -> Result<(Weights, Qwen3Config, std::path::PathBuf), Box<dyn std::error::Error>> {
    let dir = std::env::temp_dir().join(format!("marin-quant-bytes-{tag}-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir)?;
    write(&dir, &tensors(tied))?;
    let config = Qwen3Config::parse(&CONFIG.replace("TIED", &tied.to_string()), None)?;
    Ok((Weights::open(&dir)?, config, dir))
}

#[test]
fn quantized_models_count_codes_and_scales_with_int8_tables() {
    for tied in [true, false] {
        let (weights, config, dir) = load(tied, if tied { "tied" } else { "own" }).unwrap();
        let plain = weight_bytes(&weights, &config, None).unwrap();
        for (bits, group) in [
            (QuantBits::Int8, 32),
            (QuantBits::Int8, 64),
            (QuantBits::Int4, 32),
            (QuantBits::Int4, 64),
        ] {
            let scheme = QuantScheme { bits, group };
            let got = weight_bytes(&weights, &config, Some(scheme)).unwrap();
            assert_eq!(
                Some(got),
                expected(&tensors(tied), scheme),
                "tied {tied} {bits:?} group {group}"
            );
            assert!(got < plain);
        }
        let _ = fs::remove_dir_all(dir);
    }
}

#[test]
fn a_group_that_does_not_divide_a_projection_is_refused_with_the_size() {
    let (weights, config, dir) = load(true, "bad-group").unwrap();
    let scheme = QuantScheme {
        bits: QuantBits::Int8,
        group: 96,
    };
    let text = weight_bytes(&weights, &config, Some(scheme))
        .unwrap_err()
        .to_string();
    assert!(
        text.contains("not a multiple") && text.contains("96"),
        "{text}"
    );
    let _ = fs::remove_dir_all(dir);
}
