/// Plain f64 causal grouped-query attention for the last `tokens` positions.
pub fn reference_attention(
    q: &[f32],
    keys: &[f32],
    values: &[f32],
    (heads, kv_heads, head_dim, start, tokens): (usize, usize, usize, usize, usize),
) -> Vec<f32> {
    let kv_stride = kv_heads.saturating_mul(head_dim);
    let group = heads.checked_div(kv_heads).unwrap_or(1);
    let q_dim = heads.saturating_mul(head_dim);
    let mut out = Vec::new();
    for (t, q_token) in q.chunks_exact(q_dim).enumerate().take(tokens) {
        let visible = start.saturating_add(t).saturating_add(1);
        for (h, query) in q_token.chunks_exact(head_dim).enumerate() {
            let kv_head = h.checked_div(group).unwrap_or(0);
            let head_rows = |data: &'_ [f32]| -> Vec<Vec<f32>> {
                data.chunks_exact(kv_stride)
                    .take(visible)
                    .map(|row| {
                        row.chunks_exact(head_dim)
                            .nth(kv_head)
                            .unwrap_or(&[])
                            .to_vec()
                    })
                    .collect()
            };
            let (head_keys, head_values) = (head_rows(keys), head_rows(values));
            let scores: Vec<f64> = head_keys
                .iter()
                .map(|key| {
                    let dot: f64 = query.iter().zip(key).map(|(a, b)| f64::from(a * b)).sum();
                    dot / (head_dim as f64).sqrt()
                })
                .collect();
            let max = scores.iter().copied().fold(f64::NEG_INFINITY, f64::max);
            let weights: Vec<f64> = scores.iter().map(|s| (s - max).exp()).collect();
            let total: f64 = weights.iter().sum();
            for d in 0..head_dim {
                let acc: f64 = weights
                    .iter()
                    .zip(&head_values)
                    .map(|(w, v)| w * f64::from(v.get(d).copied().unwrap_or(0.0)))
                    .sum();
                out.push((acc / total) as f32);
            }
        }
    }
    out
}
