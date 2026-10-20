//! The float32 reference outputs `verify` compares against: what a reference file holds, whether it was made for a
//! given model, and how much room it needs.

use std::fs;
use std::path::Path;

use serde::Deserialize;

use crate::CliError;
use crate::load::Loaded;

#[derive(Deserialize)]
pub(crate) struct Reference {
    format: u32,
    /// How the reference's weights were stored: `bfloat16` or `int8-dequantized`.
    #[serde(default)]
    pub(crate) weights: String,
    pub(crate) cases: Vec<Case>,
}

#[derive(Deserialize)]
pub(crate) struct Case {
    pub(crate) prompt: String,
    pub(crate) prompt_ids: Vec<u32>,
    pub(crate) layer_norms: Vec<f64>,
    #[serde(default)]
    pub(crate) layer_values: Vec<Vec<f32>>,
    #[serde(default)]
    pub(crate) router_values: Vec<Vec<f32>>,
    #[serde(default)]
    pub(crate) router_top: Vec<Vec<usize>>,
    #[serde(default)]
    pub(crate) expert_values: Vec<Vec<f32>>,
    pub(crate) steps: Vec<Step>,
}

#[derive(Deserialize)]
pub(crate) struct Step {
    pub(crate) token: u32,
    pub(crate) top: Vec<(u32, f64)>,
    #[serde(default)]
    pub(crate) logits: Vec<f32>,
    #[serde(default)]
    pub(crate) router_top: Vec<Vec<usize>>,
    #[serde(default)]
    pub(crate) layer_values: Vec<Vec<f32>>,
    #[serde(default)]
    pub(crate) expert_values: Vec<Vec<f32>>,
}

pub(crate) fn read_reference(path: &Path) -> Result<Reference, CliError> {
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

/// How a reference names the way the weights of `loaded` are stored.
pub(crate) fn stored_weights(loaded: &Loaded) -> &'static str {
    match loaded.model.quantization() {
        None => "bfloat16",
        Some(scheme) if scheme.bits == model::QuantBits::Int8 => "int8-dequantized",
        Some(_) => "another quantization",
    }
}

impl Reference {
    /// The context window the longest case needs.
    pub(crate) fn needed_context(&self) -> usize {
        self.cases
            .iter()
            .map(|c| {
                c.prompt_ids
                    .len()
                    .saturating_add(c.steps.len())
                    .saturating_add(1)
            })
            .max()
            .unwrap_or(1)
    }

    pub(crate) fn cases(&self) -> usize {
        self.cases.len()
    }

    /// Whether the reference was made for the model in `loaded`: the same weights storage, the same number of layers
    /// and the same tokens for every prompt. A reference for something else says nothing about this model.
    pub(crate) fn fits(&self, loaded: &Loaded) -> Result<(), String> {
        let stored = stored_weights(loaded);
        if self.weights != stored {
            return Err(format!(
                "it was made with {} weights and this model stores {stored}",
                if self.weights.is_empty() {
                    "unnamed"
                } else {
                    &self.weights
                }
            ));
        }
        let layers = loaded.model.config().layers;
        for case in &self.cases {
            // One norm per layer; the hybrid and MoE models also record the final one.
            let recorded = case.layer_norms.len();
            if recorded != layers && recorded != layers.saturating_add(1) {
                return Err(format!(
                    "it has {recorded} layers per prompt and this model has {layers}"
                ));
            }
            if loaded.dir.tokenizer.encode(&case.prompt).ok().as_ref() != Some(&case.prompt_ids) {
                return Err("this model's tokenizer splits its prompts differently".into());
            }
        }
        Ok(())
    }
}
