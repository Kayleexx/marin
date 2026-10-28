use super::Cache;
use super::model::Qwen35;
use runtime::{CausalLm, SequenceCache};
use std::path::Path;

fn load() -> Qwen35 {
    let dir = loader::ModelDir::open(
        &Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35"),
    )
    .unwrap();
    Qwen35::load(&dir, |_, _| {}).unwrap()
}

fn read(m: &Qwen35, c: &mut Cache, tokens: &[u32]) -> Vec<f32> {
    m.forward(tokens, c).unwrap()
}

#[test]
fn returning_to_a_mark_reads_only_what_came_after_and_equals_a_full_replay() {
    let m = load();
    let (history, question, answer) = ([17, 4, 81, 29], [5, 63, 8], [12, 44]);
    let mut c = m.new_cache(32).unwrap();
    read(&m, &mut c, &history);
    let mark = m.mark_cache(&c).unwrap();
    let first = read(&m, &mut c, &question);
    read(&m, &mut c, &answer);
    m.rewind_cache(&mut c, mark).unwrap();
    assert_eq!(c.used(), history.len());
    assert!(!c.replays());
    let again = read(&m, &mut c, &question);
    assert_eq!(
        first, again,
        "the same pass after the rewind gives the same logits"
    );
    let mut fresh = m.new_cache(32).unwrap();
    let all: Vec<u32> = history.iter().chain(&question).copied().collect();
    assert_eq!(
        again,
        read(&m, &mut fresh, &all),
        "equal to reading everything"
    );
}

#[test]
fn a_mark_is_refused_after_a_cut_back_a_restart_or_for_another_cache() {
    let m = load();
    let mut c = m.new_cache(32).unwrap();
    read(&m, &mut c, &[17, 4, 81, 29]);
    let cut = m.mark_cache(&c).unwrap();
    read(&m, &mut c, &[5, 63]);
    c.truncate(2).unwrap();
    assert!(c.replays());
    assert!(m.rewind_cache(&mut c, cut).is_err(), "history was cut back");
    assert!(
        m.mark_cache(&c).is_err(),
        "a cache to be read again cannot be marked"
    );

    let mut a = m.new_cache(32).unwrap();
    let mut b = m.new_cache(32).unwrap();
    read(&m, &mut a, &[17, 4]);
    read(&m, &mut b, &[17, 4]);
    let foreign = m.mark_cache(&a).unwrap();
    assert!(
        m.rewind_cache(&mut b, foreign).is_err(),
        "another cache's mark"
    );
    let stale = m.mark_cache(&b).unwrap();
    b.clear();
    read(&m, &mut b, &[17, 4]);
    assert!(m.rewind_cache(&mut b, stale).is_err(), "restarted since");
    assert_eq!(b.used(), 2, "a refused rewind changes nothing");
}
