use serde_json::{Value, json};

use super::*;

fn stat(min: f64, median: f64, max: f64) -> Value {
    json!({"min": min, "median": median, "max": max})
}

fn saved(first_token: Value, decode: Value) -> Saved {
    serde_json::from_value(json!({
        "model": "m", "precision": "bf16", "quantization": "none",
        "context_length": 100, "batch_size": 1, "prompt_tokens": 8, "output_tokens": 4,
        "sampler": {"temperature": 0.0, "top_k": 0, "top_p": 1.0, "seed": 0},
        "runtime": "marin 0.0.0 (abc)",
        "hardware": {"cpu": "c", "backend": "cpu", "gpu": null},
        "results": {
            "first_token_ms": first_token,
            "prefill_tokens_per_s": stat(100.0, 100.0, 100.0),
            "decode_tokens_per_s": decode,
            "token_latency_p50_ms": stat(1.0, 1.0, 1.0),
            "token_latency_p95_ms": stat(1.0, 1.0, 1.0),
            "sampling_ms_per_token": stat(0.1, 0.1, 0.1),
            "weight_bandwidth_gb_per_s": stat(10.0, 10.0, 10.0),
        }
    }))
    .unwrap()
}

fn report(old: &Saved, new: &Saved) -> String {
    render(old, new, Path::new("old.json"), Path::new("new.json"))
}

#[test]
fn separated_ranges_are_called_faster_or_slower_in_the_right_direction() {
    let old = saved(stat(100.0, 105.0, 110.0), stat(40.0, 42.0, 44.0));
    let new = saved(stat(80.0, 85.0, 90.0), stat(30.0, 32.0, 34.0));
    let text = report(&old, &new);
    let first = text.lines().find(|l| l.contains("first token")).unwrap();
    assert!(
        first.contains("-19.0%") && first.ends_with("faster"),
        "{first}"
    );
    let decode = text.lines().find(|l| l.contains("decode")).unwrap();
    assert!(
        decode.contains("-23.8%") && decode.ends_with("slower"),
        "{decode}"
    );
}

#[test]
fn overlapping_ranges_are_noise_not_a_win() {
    let old = saved(stat(100.0, 105.0, 110.0), stat(40.0, 42.0, 44.0));
    let new = saved(stat(108.0, 112.0, 120.0), stat(41.0, 43.0, 45.0));
    let text = report(&old, &new);
    let first = text.lines().find(|l| l.contains("first token")).unwrap();
    assert!(
        first.contains("(ranges overlap)") && first.contains("+6.7%"),
        "{first}"
    );
    assert!(
        text.contains("same build"),
        "identical runtimes are called out"
    );
}

#[test]
fn a_shared_dirty_label_is_not_called_the_same_build() {
    let mut old = saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    let mut new = saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    old.runtime = "marin 0.0.0 (abc-dirty)".into();
    new.runtime = "marin 0.0.0 (abc-dirty)".into();
    let text = report(&old, &new);
    assert!(text.contains("uncommitted code"), "{text}");
    assert!(!text.contains("same build"), "{text}");
    new.runtime = "marin 0.0.0 (def)".into();
    let text = report(&old, &new);
    assert!(
        !text.contains("same build") && !text.contains("uncommitted"),
        "{text}"
    );
}

fn with_gpu(mut saved: Saved, clock: f64, cap: f64) -> Saved {
    saved.results.gpu_sampling = Some(Sampling {
        status: "sampled".into(),
        clock_mhz: Some(Reading {
            mean: clock,
            max: clock,
        }),
        power_cap_percent: Some(cap),
    });
    saved
}

#[test]
fn different_sampler_settings_are_different_setups() {
    let old = saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    let mut new = saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    new.sampler.temperature = 0.7;
    new.sampler.top_k = 20;
    let found = mismatches(&old, &new);
    assert_eq!(found.len(), 1, "{found:?}");
    assert!(
        found[0].starts_with("sampler: temperature 0 top-k 0"),
        "{found:?}"
    );
}

#[test]
fn a_change_in_gpu_clocks_or_power_cap_is_called_out() {
    let base = || saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    let (old, new) = (
        with_gpu(base(), 1600.0, 100.0),
        with_gpu(base(), 1250.0, 100.0),
    );
    let note = conditions(&old, &new).unwrap();
    assert!(
        note.contains("1600 vs 1250 MHz (-22%)") && note.contains("not the code"),
        "{note}"
    );
    assert!(report(&old, &new).contains("note: the runs were not made under the same conditions"));
    let same = with_gpu(base(), 1550.0, 90.0);
    assert!(
        conditions(&old, &same).is_none(),
        "small differences are not worth a warning"
    );
    let capped = with_gpu(base(), 1600.0, 20.0);
    assert!(conditions(&old, &capped).unwrap().contains("100% vs 20%"));
    assert!(
        conditions(&base(), &base()).is_none(),
        "no GPU readings, no note"
    );
}

#[test]
fn a_difference_in_power_source_or_power_limit_is_called_out() {
    let base = || saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    let (mut ac, mut battery) = (base(), base());
    ac.hardware.on_ac_power = Some(true);
    battery.hardware.on_ac_power = Some(false);
    let note = conditions(&ac, &battery).unwrap();
    assert!(
        note.contains("one run was on AC power and the other on battery"),
        "{note}"
    );
    let (mut low, mut high) = (base(), base());
    low.hardware.gpu_power_limit_watts = Some(40.0);
    high.hardware.gpu_power_limit_watts = Some(60.0);
    assert!(
        conditions(&low, &high)
            .unwrap()
            .contains("the GPU power limit was 40 W and 60 W")
    );
    let mut both = base();
    both.hardware.on_ac_power = Some(true);
    both.hardware.gpu_power_limit_watts = Some(40.0);
    assert!(
        conditions(&ac, &both).is_none(),
        "the same setup says nothing"
    );
    assert!(
        conditions(&ac, &base()).is_none(),
        "an unknown setup is not a difference"
    );
}

#[test]
fn a_different_pause_between_tokens_is_a_different_setup() {
    let old = saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    let mut new = saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    new.token_pause_ms = 5.0;
    let found = mismatches(&old, &new);
    assert_eq!(found, ["pause after each token: 0 ms vs 5 ms"]);
}

#[test]
fn a_zero_baseline_has_no_percentage() {
    let old = saved(stat(0.0, 0.0, 0.0), stat(1.0, 1.0, 1.0));
    let new = saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    let text = report(&old, &new);
    assert!(
        text.lines()
            .find(|l| l.contains("first token"))
            .unwrap()
            .contains("n/a")
    );
    assert!(!text.contains("NaN") && !text.contains("inf"));
}

#[test]
fn setups_that_differ_are_listed_not_compared() {
    let old = saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    let mut new = saved(stat(1.0, 1.0, 1.0), stat(1.0, 1.0, 1.0));
    assert!(mismatches(&old, &new).is_empty());
    new.prompt_tokens = 512;
    new.hardware.backend = "cuda".into();
    new.hardware.gpu = Some(Gpu { name: "RTX".into() });
    let found = mismatches(&old, &new);
    assert_eq!(found.len(), 3, "{found:?}");
    assert!(found.iter().any(|f| f == "prompt tokens: 8 vs 512"));
    assert!(found.iter().any(|f| f.starts_with("backend:")));
    assert!(found.iter().any(|f| f.starts_with("gpu:")));
}

#[test]
fn a_change_of_quantization_is_compared_and_stated() {
    let old = saved(stat(100.0, 105.0, 110.0), stat(40.0, 42.0, 44.0));
    let mut new = saved(stat(100.0, 105.0, 110.0), stat(60.0, 62.0, 64.0));
    new.quantization = "int8, groups of 128".into();
    assert!(mismatches(&old, &new).is_empty());
    let text = report(&old, &new);
    assert!(
        text.contains("quantization: none -> int8, groups of 128"),
        "{text}"
    );
    assert!(!report(&old, &old).contains("quantization:"));
}
