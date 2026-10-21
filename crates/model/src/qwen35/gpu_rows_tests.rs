//! One pass over a few tokens returns the logits after each of them. Whether those equal the logits of feeding the
//! tokens one at a time decides if guessed tokens can be checked without changing what the model says.

use super::gpu_forward::FAIL_AFTER_LAYERS;
use super::gpu_mark_tests::{ids, model};
use super::gpu_tests::contents;
use runtime::{CausalLm, SequenceCache};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;

#[test]
fn rows_of_one_pass_match_the_logits_of_feeding_the_tokens_one_at_a_time() -> Result {
    let Some(model) = model()? else { return Ok(()) };
    let vocab = model.config().vocab_size;
    let (mut rows, mut alone) = (model.new_cache(1024)?, model.new_cache(1024)?);
    model.forward(&ids(10, 70), &mut rows)?;
    model.forward(&ids(10, 70), &mut alone)?;
    let mut worst = 0.0_f32;
    let mut exact = true;
    for step in [2_usize, 3, 4, 1, 4] {
        let fed = ids(100, step);
        let each = model.forward_rows(&fed, &mut rows)?;
        let mut one_by_one = Vec::new();
        for token in &fed {
            one_by_one.extend(model.forward(&[*token], &mut alone)?);
        }
        assert_eq!(each.len(), step * vocab);
        exact &= each == one_by_one;
        worst = each
            .iter()
            .zip(&one_by_one)
            .fold(worst, |w, (a, b)| w.max((a - b).abs()));
    }
    eprintln!("rows vs one at a time: bit-identical {exact}, largest difference {worst:e}");
    assert!(worst < 1e-5, "largest difference {worst:e}");
    assert_eq!(contents(&model, &rows)?, contents(&model, &alone)?);
    Ok(())
}

#[test]
fn a_cut_back_inside_a_pass_of_rows_reads_only_what_was_kept_and_matches_a_cache_that_never_saw_the_rest()
-> Result {
    let Some(model) = model()? else { return Ok(()) };
    for kept in 0..=4_usize {
        let (mut live, mut twin) = (model.new_cache(1024)?, model.new_cache(1024)?);
        model.forward(&ids(10, 70), &mut live)?;
        model.forward(&ids(10, 70), &mut twin)?;
        let fed = ids(100, 4);
        model.forward_rows(&fed, &mut live)?;
        live.truncate(70 + kept)?;
        assert_eq!(live.used(), 70 + kept);
        assert!(!live.replays(), "kept {kept}: only the tail is read again");
        let next = ids(200, 1);
        let got = model.forward(&next, &mut live)?;
        let kept_ids = fed.get(..kept).unwrap_or(&[]);
        let want = if kept == fed.len() {
            model.forward(kept_ids, &mut twin)?;
            model.forward(&next, &mut twin)?
        } else {
            model.forward(&[kept_ids, &next].concat(), &mut twin)?
        };
        assert!(got == want, "kept {kept}: logits differ");
        assert_eq!(contents(&model, &live)?, contents(&model, &twin)?);
    }
    Ok(())
}

#[test]
fn a_pass_that_fails_after_a_cut_back_inside_rows_leaves_a_cache_that_reads_everything_again()
-> Result {
    let Some(model) = model()? else { return Ok(()) };
    let mut live = model.new_cache(1024)?;
    model.forward(&ids(10, 70), &mut live)?;
    let fed = ids(100, 4);
    model.forward_rows(&fed, &mut live)?;
    live.truncate(72)?;
    FAIL_AFTER_LAYERS.set(Some(3));
    assert!(model.forward(&ids(200, 1), &mut live).is_err());
    FAIL_AFTER_LAYERS.set(None);
    assert!(
        live.replays() && live.used() == 72,
        "the whole history is to be read again"
    );
    let got = model.forward(&ids(200, 1), &mut live)?;
    let everything = [
        ids(10, 70),
        fed.get(..2).unwrap_or(&[]).to_vec(),
        ids(200, 1),
    ]
    .concat();
    let mut fresh = model.new_cache(1024)?;
    assert!(got == model.forward(&everything, &mut fresh)?);
    let twin = fresh;
    assert_eq!(contents(&model, &live)?, contents(&model, &twin)?);
    Ok(())
}

#[test]
fn many_cuts_in_a_row_never_read_the_history_again_and_follow_a_cache_that_read_what_was_kept()
-> Result {
    let Some(model) = model()? else { return Ok(()) };
    let (mut live, mut twin) = (model.new_cache(1024)?, model.new_cache(1024)?);
    model.forward(&ids(10, 70), &mut live)?;
    model.forward(&ids(10, 70), &mut twin)?;
    for (round, kept) in [1_usize, 0, 3, 2, 4, 0, 1].into_iter().enumerate() {
        let fed = ids(100 + 7 * u32::try_from(round)?, 4);
        let got = model.forward_rows(&fed, &mut live)?;
        let before = twin.used();
        assert!(
            got == model.forward_rows(&fed, &mut twin)?,
            "round {round}: rows differ"
        );
        live.truncate(before + kept)?;
        twin.truncate(before + kept)?;
        assert!(
            !live.replays(),
            "round {round}: only the tail is read again"
        );
        assert_eq!(live.used(), before + kept);
    }
    let last = ids(250, 1);
    assert!(model.forward(&last, &mut live)? == model.forward(&last, &mut twin)?);
    assert_eq!(contents(&model, &live)?, contents(&model, &twin)?);
    Ok(())
}

#[test]
fn a_mark_taken_before_a_reply_still_returns_the_cache_after_the_reply_was_cut_back_inside_rows()
-> Result {
    let Some(model) = model()? else { return Ok(()) };
    let (mut live, mut twin) = (model.new_cache(1024)?, model.new_cache(1024)?);
    model.forward(&ids(10, 70), &mut live)?;
    model.forward(&ids(10, 70), &mut twin)?;
    let boundary = model.mark_cache(&live)?;
    model.forward(&ids(300, 20), &mut live)?;
    model.forward_rows(&ids(100, 4), &mut live)?;
    live.truncate(92)?;
    model.forward_rows(&ids(110, 4), &mut live)?;
    live.truncate(95)?;
    model.rewind_cache(&mut live, boundary)?;
    assert_eq!(live.used(), 70);
    assert!(!live.replays());
    assert_eq!(contents(&model, &live)?, contents(&model, &twin)?);
    assert!(model.forward(&[5], &mut live)? == model.forward(&[5], &mut twin)?);

    let at = live.used();
    let boundary = model.mark_cache(&live)?;
    model.forward_rows(&ids(100, 4), &mut live)?;
    live.truncate(at + 1)?;
    live.truncate(at)?;
    model.rewind_cache(&mut live, boundary)?;
    assert_eq!(contents(&model, &live)?, contents(&model, &twin)?);

    let boundary = model.mark_cache(&live)?;
    model.forward(&ids(300, 6), &mut live)?;
    model.forward_rows(&ids(100, 4), &mut live)?;
    live.truncate(at + 2)?;
    live.truncate(at + 1)?;
    assert!(
        model.rewind_cache(&mut live, boundary).is_err(),
        "a cut below the mark refuses it"
    );
    Ok(())
}
