//! Without a GPU these tests skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod attention_reference;
mod common;
mod mirror;

use common::gpu;
use cuda::{DeviceKv, PAGE_TOKENS, SLAB_PAGES, pages_for};
use mirror::{LAYERS, Mirror, Result, SMALL};

#[test]
fn attention_reads_the_right_page_and_row_at_every_boundary() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for total in [
        0, 1, 2, 31, 32, 33, 62, 63, 64, 65, 66, 126, 127, 128, 129, 130, 191, 192, 193, 320, 321,
    ] {
        let mut mirror = Mirror::new(&gpu, SMALL, 400)?;
        if total > 0 {
            mirror.append(&gpu, total)?;
        }
        assert_eq!(mirror.kv.pages_in_use(), pages_for(total), "{total} tokens");
        mirror.check_every_kernel(&gpu)?;
        assert_eq!(
            mirror.kv.pages_in_use(),
            pages_for(total),
            "{total}: checks leave no page behind"
        );
    }
    Ok(())
}

#[test]
fn growing_one_token_at_a_time_crosses_pages_correctly() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 260)?;
    for len in 1..=200 {
        mirror.append(&gpu, 1)?;
        assert_eq!(mirror.kv.pages_in_use(), pages_for(len));
        if [1, 63, 64, 65, 127, 128, 129, 191, 192, 193, 200].contains(&len) {
            mirror.check(&gpu, 1)?;
            mirror.check(&gpu, 4)?;
        }
    }
    Ok(())
}

#[test]
fn a_realistic_head_layout_reads_pages_correctly() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for total in [63, 64, 65, 129] {
        let mut mirror = Mirror::new(&gpu, (16, 8, 128), 300)?;
        mirror.append(&gpu, total)?;
        mirror.check_every_kernel(&gpu)?;
    }
    Ok(())
}

#[test]
fn a_cleared_and_refilled_cache_reads_through_a_reversed_page_table() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 600)?;
    let tokens = 5 * PAGE_TOKENS + 17;
    mirror.append(&gpu, tokens)?;
    assert_eq!(mirror.kv.page_ids(), [0, 1, 2, 3, 4, 5]);
    mirror.truncate(0)?;
    assert_eq!(mirror.kv.pages_in_use(), 0);
    mirror.append(&gpu, tokens)?;
    assert_eq!(
        mirror.kv.page_ids(),
        [5, 4, 3, 2, 1, 0],
        "new data in the other physical pages"
    );
    mirror.check_every_kernel(&gpu)?;
    assert_eq!(
        mirror.kv.pages_held(),
        6,
        "the same six pages served both fills"
    );
    Ok(())
}

#[test]
fn clear_gives_every_page_back_like_truncating_to_zero() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 300)?;
    mirror.append(&gpu, 150)?;
    mirror.kv.clear();
    assert_eq!((mirror.kv.len(), mirror.kv.pages_in_use()), (0, 0));
    assert_eq!(mirror.kv.pages_held(), 3);
    for layer in &mut mirror.layers {
        layer.keys.clear();
        layer.vals.clear();
    }
    mirror.append(&gpu, 150)?;
    mirror.check_every_kernel(&gpu)
}

#[test]
fn rolling_back_into_a_page_and_regrowing_mixes_the_page_order() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 600)?;
    mirror.append(&gpu, 400)?;
    assert_eq!(mirror.kv.page_ids(), [0, 1, 2, 3, 4, 5, 6]);
    mirror.truncate(70)?;
    assert_eq!(mirror.kv.page_ids(), [0, 1]);
    mirror.append(&gpu, 330)?;
    assert_eq!(mirror.kv.page_ids(), [0, 1, 6, 5, 4, 3, 2]);
    mirror.check_every_kernel(&gpu)?;
    assert_eq!(mirror.kv.pages_held(), 7);
    Ok(())
}

#[test]
fn truncating_gives_pages_back_exactly_when_a_page_empties() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 300)?;
    mirror.append(&gpu, 200)?;
    assert_eq!(mirror.kv.pages_in_use(), 4);
    for (len, pages) in [
        (193, 4),
        (192, 3),
        (129, 3),
        (128, 2),
        (65, 2),
        (64, 1),
        (1, 1),
        (0, 0),
    ] {
        mirror.truncate(len)?;
        assert_eq!(mirror.kv.pages_in_use(), pages, "truncated to {len}");
        assert_eq!(mirror.kv.pages_held(), 4, "pages stay allocated for reuse");
    }
    mirror.append(&gpu, 200)?;
    mirror.check_every_kernel(&gpu)?;
    assert_eq!(
        mirror.kv.pages_held(),
        4,
        "the regrown sequence reused them"
    );
    Ok(())
}

#[test]
fn memory_follows_the_live_tokens_and_not_the_capacity() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 8192)?;
    assert_eq!((mirror.kv.pages_held(), mirror.kv.bytes_held()), (0, 0));
    let page = LAYERS * 2 * PAGE_TOKENS * 2 * 16 * 4;
    let slab = SLAB_PAGES * page;
    assert_eq!(
        (mirror.kv.page_bytes(), mirror.kv.slab_bytes()),
        (page, slab)
    );
    mirror.append(&gpu, 10)?;
    assert_eq!((mirror.kv.pages_held(), mirror.kv.bytes_held()), (1, slab));
    mirror.append(&gpu, 54)?;
    assert_eq!(mirror.kv.pages_held(), 1, "64 tokens still fit one page");
    mirror.append(&gpu, 1)?;
    assert_eq!((mirror.kv.pages_held(), mirror.kv.bytes_held()), (2, slab));
    mirror.append(&gpu, 8 * PAGE_TOKENS - 65)?;
    assert_eq!((mirror.kv.pages_held(), mirror.kv.bytes_held()), (8, slab));
    mirror.append(&gpu, 1)?;
    assert_eq!(
        (mirror.kv.pages_held(), mirror.kv.bytes_held()),
        (9, 2 * slab)
    );
    mirror.append(&gpu, 1000)?;
    assert_eq!(mirror.kv.pages_held(), pages_for(8 * PAGE_TOKENS + 1001));
    assert_eq!(mirror.kv.bytes_held(), 3 * slab);
    Ok(())
}

#[test]
fn repeated_appends_and_rollbacks_like_guessing_ahead_stay_exact() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 400)?;
    mirror.append(&gpu, 50)?;
    let mut state = 0x2545_f491_4f6c_dd1d_u64;
    for step in 0..150_usize {
        state ^= state.wrapping_shl(13);
        state ^= state.wrapping_shr(7);
        state ^= state.wrapping_shl(17);
        let grow = usize::try_from(state & 7)? + 1;
        let back = usize::try_from(state.wrapping_shr(8) & 7)?.min(grow);
        let room = 400_usize.saturating_sub(mirror.kv.len()).saturating_sub(1);
        mirror.append(&gpu, grow.min(room))?;
        mirror.truncate(mirror.kv.len().saturating_sub(back))?;
        assert_eq!(
            mirror.kv.pages_in_use(),
            pages_for(mirror.kv.len()),
            "step {step}"
        );
        if step % 7 == 0 {
            mirror.check(&gpu, if step % 2 == 0 { 1 } else { 3 })?;
        }
        if mirror.kv.len() > 330 {
            mirror.truncate(60)?;
        }
    }
    Ok(())
}

#[test]
fn a_failed_write_keeps_no_pages_and_attention_refuses_pages_that_do_not_exist() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut kv = DeviceKv::new(&gpu, 1, SMALL, 100)?;
    let too_many = gpu.zeros(101 * 32)?;
    assert!(kv.write(&gpu, 0, &too_many, &too_many).is_err());
    assert_eq!((kv.pages_in_use(), kv.pages_held()), (0, 0));
    // Advanced without a write, so these pages are unmapped.
    kv.advance(70)?;
    for tokens in [1, 2, 12] {
        let q = gpu.zeros(tokens * 4 * 16)?;
        let mut out = gpu.zeros(tokens * 4 * 16)?;
        assert!(
            gpu.attention(&q, &mut kv, 0, 4, tokens, &mut out).is_err(),
            "{tokens} tokens"
        );
    }
    Ok(())
}

#[test]
fn a_capacity_that_is_not_a_page_multiple_fills_exactly() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 130)?;
    mirror.append(&gpu, 129)?;
    mirror.check(&gpu, 1)?;
    mirror.append(&gpu, 1)?;
    assert_eq!((mirror.kv.len(), mirror.kv.pages_in_use()), (130, 3));
    let extra = gpu.zeros(32)?;
    assert!(
        mirror.kv.write(&gpu, 0, &extra, &extra).is_err(),
        "one past capacity"
    );
    assert!(mirror.kv.advance(1).is_err());
    Ok(())
}

#[test]
fn two_sequences_do_not_see_each_others_pages() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut a = Mirror::new(&gpu, SMALL, 300)?;
    let mut b = Mirror::new(&gpu, SMALL, 300)?;
    for round in 0..6_usize {
        a.append(&gpu, 37)?;
        b.append(&gpu, 45)?;
        if round % 2 == 1 {
            a.truncate(a.kv.len().saturating_sub(50))?;
        }
        a.check(&gpu, 1)?;
        b.check(&gpu, 3)?;
    }
    Ok(())
}
