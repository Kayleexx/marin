use std::fs;
use std::path::Path;

use loader::Weights;
use model::{Qwen3Config, weight_bytes};

const CONFIG: &str = r#"{
    "model_type": "qwen3", "hidden_size": 8, "intermediate_size": 16, "num_hidden_layers": 1,
    "num_attention_heads": 2, "num_key_value_heads": 1, "head_dim": 4, "vocab_size": 10,
    "max_position_embeddings": 32, "rms_norm_eps": 1e-6, "rope_theta": 10000.0,
    "tie_word_embeddings": true
}"#;

/// (name, shape) of every tensor the config needs, plus whatever extras the test adds.
fn tensors(extra: &[(&str, &[usize])]) -> Vec<(String, Vec<usize>)> {
    let layer = |s: &str| format!("model.layers.0.{s}");
    let mut all: Vec<(String, Vec<usize>)> = vec![
        ("model.embed_tokens.weight".into(), vec![10, 8]),
        ("model.norm.weight".into(), vec![8]),
        (layer("input_layernorm.weight"), vec![8]),
        (layer("self_attn.q_proj.weight"), vec![8, 8]),
        (layer("self_attn.k_proj.weight"), vec![4, 8]),
        (layer("self_attn.v_proj.weight"), vec![4, 8]),
        (layer("self_attn.o_proj.weight"), vec![8, 8]),
        (layer("self_attn.q_norm.weight"), vec![4]),
        (layer("self_attn.k_norm.weight"), vec![4]),
        (layer("post_attention_layernorm.weight"), vec![8]),
        (layer("mlp.gate_proj.weight"), vec![16, 8]),
        (layer("mlp.up_proj.weight"), vec![16, 8]),
        (layer("mlp.down_proj.weight"), vec![8, 16]),
    ];
    all.extend(extra.iter().map(|(n, s)| (n.to_string(), s.to_vec())));
    all
}

fn write_safetensors(dir: &Path, all: &[(String, Vec<usize>)]) -> std::io::Result<u64> {
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for (name, shape) in all {
        let bytes = shape.iter().product::<usize>().saturating_mul(2);
        header.insert(
            name.clone(),
            serde_json::json!({"dtype": "BF16", "shape": shape, "data_offsets": [data.len(), data.len().saturating_add(bytes)]}),
        );
        data.extend(std::iter::repeat_n(0_u8, bytes));
    }
    let header = serde_json::to_vec(&header).map_err(std::io::Error::other)?;
    let mut file = (header.len() as u64).to_le_bytes().to_vec();
    file.extend(header);
    file.extend(&data);
    fs::write(dir.join("model.safetensors"), file)?;
    Ok(data.len() as u64)
}

#[test]
fn only_the_tensors_that_are_loaded_are_counted_not_unused_extras() {
    let dir = std::env::temp_dir().join(format!("marin-weight-bytes-{}", std::process::id()));
    let _ = fs::remove_dir_all(&dir);
    fs::create_dir_all(&dir).unwrap();
    let with_extra = tensors(&[("lm_head.weight", &[10, 8])]);
    let file_data_bytes = write_safetensors(&dir, &with_extra).unwrap();

    let weights = Weights::open(&dir).unwrap();
    let config = Qwen3Config::parse(CONFIG, None).unwrap();
    let needed = weight_bytes(&weights, &config).unwrap();

    let extra_bytes: u64 = 10 * 8 * 2;
    assert_eq!(
        weights.total_bytes(),
        file_data_bytes,
        "the file holds the extra tensor"
    );
    assert_eq!(
        needed,
        file_data_bytes - extra_bytes,
        "tied models do not load lm_head"
    );

    let missing = tensors(&[])
        .into_iter()
        .filter(|(n, _)| !n.contains("k_proj"))
        .collect::<Vec<_>>();
    write_safetensors(&dir, &missing).unwrap();
    let err = weight_bytes(&Weights::open(&dir).unwrap(), &config)
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("missing tensor model.layers.0.self_attn.k_proj.weight"),
        "{err}"
    );
    let _ = fs::remove_dir_all(&dir);
}
