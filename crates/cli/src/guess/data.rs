use std::fs;
use std::path::Path;

use serde::de::DeserializeOwned;
use serde::{Deserialize, Serialize};

use crate::CliError;

pub const FORMAT: u32 = 1;

/// What the model said for a set of prompts, saved so the guessers can be tried again without the model.
#[derive(Serialize, Deserialize)]
pub struct Transcripts {
    pub format: u32,
    pub model: String,
    pub quantization: String,
    pub backend: String,
    pub prompts: Vec<PromptRun>,
}

#[derive(Serialize, Deserialize)]
pub struct PromptRun {
    pub class: String,
    pub name: String,
    pub prompt: Vec<u32>,
    pub replies: Vec<Reply>,
}

#[derive(Serialize, Deserialize)]
pub struct Reply {
    /// How the tokens were chosen: `greedy` or `chat` (the chat defaults, fixed seed).
    pub mode: String,
    pub tokens: Vec<u32>,
}

/// Time of a forward pass over `tokens` new tokens after `context` tokens already held.
#[derive(Serialize, Deserialize)]
pub struct Costs {
    pub format: u32,
    pub model: String,
    pub quantization: String,
    pub backend: String,
    pub rounds: usize,
    /// Passes counted per size and round.
    pub reps: usize,
    pub curves: Vec<Curve>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Curve {
    pub context: usize,
    pub points: Vec<Point>,
}

#[derive(Clone, Serialize, Deserialize)]
pub struct Point {
    pub tokens: usize,
    pub median_ms: f64,
    pub min_ms: f64,
    pub max_ms: f64,
}

pub fn read_saved<T: DeserializeOwned>(path: &Path) -> Result<T, CliError> {
    let saved = |reason: String| CliError::Saved {
        path: path.display().to_string(),
        reason,
    };
    let text = fs::read_to_string(path).map_err(|e| saved(e.to_string()))?;
    serde_json::from_str(&text).map_err(|e| saved(e.to_string()))
}

impl Curve {
    /// Median pass time for `tokens`, straight lines between measured sizes and the last slope beyond them.
    pub fn ms(&self, tokens: usize) -> f64 {
        let at = tokens as f64;
        let mut points = self.points.iter().map(|p| (p.tokens as f64, p.median_ms));
        let Some(mut before) = points.next() else {
            return 0.0;
        };
        if at <= before.0 {
            return before.1;
        }
        let mut slope = 0.0;
        for next in points {
            slope = (next.1 - before.1) / (next.0 - before.0);
            if at <= next.0 {
                return before.1 + slope * (at - before.0);
            }
            before = next;
        }
        before.1 + slope * (at - before.0)
    }
}

/// Pass times measured at several context lengths; a pass is timed with the nearest one.
pub struct CostModel {
    curves: Vec<Curve>,
}

impl CostModel {
    pub fn new(curves: Vec<Curve>) -> Self {
        Self { curves }
    }

    pub fn is_empty(&self) -> bool {
        self.curves.is_empty()
    }

    pub fn pass_ms(&self, context: usize, tokens: usize) -> f64 {
        let distance = |c: &Curve| {
            ((c.context as f64 + 1.0) / (context as f64 + 1.0))
                .ln()
                .abs()
        };
        self.curves
            .iter()
            .min_by(|a, b| distance(a).total_cmp(&distance(b)))
            .map_or(0.0, |curve| curve.ms(tokens))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn curve(context: usize, points: &[(usize, f64)]) -> Curve {
        Curve {
            context,
            points: points
                .iter()
                .map(|&(tokens, ms)| Point {
                    tokens,
                    median_ms: ms,
                    min_ms: ms,
                    max_ms: ms,
                })
                .collect(),
        }
    }

    #[test]
    fn interpolates_between_measured_sizes_and_extends_the_last_slope() {
        let c = curve(128, &[(1, 10.0), (3, 12.0), (5, 18.0)]);
        assert_eq!(c.ms(1), 10.0);
        assert_eq!(c.ms(2), 11.0);
        assert_eq!(c.ms(4), 15.0);
        assert_eq!(c.ms(7), 24.0);
    }

    #[test]
    fn a_pass_uses_the_curve_with_the_nearest_context() {
        let model = CostModel::new(vec![
            curve(128, &[(1, 10.0)]),
            curve(1024, &[(1, 20.0)]),
            curve(4096, &[(1, 40.0)]),
        ]);
        assert_eq!(model.pass_ms(90, 1), 10.0);
        assert_eq!(model.pass_ms(500, 1), 20.0);
        assert_eq!(model.pass_ms(3000, 1), 40.0);
    }
}
