//! Pages and slabs at their boundaries: the rows either side of a slab, truncating across one, rolling guesses back
//! across one, and what regrowing reuses.
//! Without a GPU these tests skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod attention_reference;
mod common;
mod kv_round;
mod mirror;

use common::gpu;
use cuda::{KvFormat, PAGE_TOKENS, SLAB_PAGES, pages_for};
use mirror::{Mirror, Result, SMALL};

#[test]
fn slab_boundaries_read_the_right_page_and_row() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for boundary in [8 * PAGE_TOKENS, 16 * PAGE_TOKENS] {
        for total in boundary - 3..=boundary + 2 {
            let mut mirror = Mirror::new(&gpu, SMALL, 1100, KvFormat::F32)?;
            mirror.append(&gpu, total)?;
            let slabs = pages_for(total).div_ceil(SLAB_PAGES);
            assert_eq!(mirror.kv.pages_in_use(), pages_for(total), "{total}");
            assert_eq!(
                mirror.kv.bytes_held(),
                slabs * mirror.kv.slab_bytes(),
                "{total}"
            );
            mirror.check_every_kernel(&gpu)?;
            assert_eq!(
                mirror.kv.pages_in_use(),
                pages_for(total),
                "{total}: checks leave no page"
            );
        }
    }
    Ok(())
}

#[test]
fn growing_one_token_at_a_time_crosses_a_slab_correctly() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let boundary = 8 * PAGE_TOKENS;
    let mut mirror = Mirror::new(&gpu, SMALL, 600, KvFormat::F32)?;
    mirror.append(&gpu, boundary - 6)?;
    for len in boundary - 5..=boundary + 8 {
        mirror.append(&gpu, 1)?;
        assert_eq!(mirror.kv.pages_in_use(), pages_for(len), "{len}");
        mirror.check(&gpu, 1)?;
    }
    assert_eq!(mirror.kv.bytes_held(), 2 * mirror.kv.slab_bytes());
    Ok(())
}

#[test]
fn rejected_guesses_across_a_page_and_a_slab_boundary_stay_exact() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for boundary in [PAGE_TOKENS, 8 * PAGE_TOKENS] {
        let mut mirror = Mirror::new(&gpu, SMALL, 600, KvFormat::F32)?;
        mirror.append(&gpu, boundary - 3)?;
        for round in 0..6 {
            mirror.append(&gpu, 8)?;
            let past = mirror.kv.len() > boundary;
            assert_eq!(
                mirror.kv.pages_in_use(),
                pages_for(mirror.kv.len()),
                "round {round}"
            );
            mirror.truncate(boundary - 3 + round % 3)?;
            assert_eq!(
                mirror.kv.pages_in_use(),
                pages_for(mirror.kv.len()),
                "round {round}"
            );
            assert!(past, "the guess crossed the boundary {boundary}");
            mirror.check(&gpu, 1)?;
        }
    }
    Ok(())
}

#[test]
fn truncating_across_a_slab_boundary_keeps_the_slabs_and_regrows_exactly() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 1100, KvFormat::F32)?;
    mirror.append(&gpu, 1050)?;
    let slabs = 3 * mirror.kv.slab_bytes();
    assert_eq!(mirror.kv.bytes_held(), slabs);
    for len in [1024, 600, 512, 500, 300] {
        mirror.truncate(len)?;
        assert_eq!(
            mirror.kv.pages_in_use(),
            pages_for(len),
            "truncated to {len}"
        );
        assert_eq!(mirror.kv.bytes_held(), slabs, "no slab is given back");
    }
    mirror.append(&gpu, 700)?;
    assert_eq!(mirror.kv.bytes_held(), slabs);
    mirror.check_every_kernel(&gpu)
}

#[test]
fn pages_freed_by_a_rollback_are_reused_without_allocating() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 1100, KvFormat::F32)?;
    mirror.append(&gpu, 600)?;
    let mut owned: Vec<usize> = mirror.kv.page_ids().to_vec();
    owned.sort_unstable();
    let (held, bytes) = (mirror.kv.pages_held(), mirror.kv.bytes_held());
    let before = gpu.traffic();
    mirror.truncate(100)?;
    mirror.append(&gpu, 500)?;
    let spent = gpu.traffic().since(&before);
    assert_eq!(
        spent.allocations, 0,
        "regrowing took nothing new from the GPU"
    );
    assert_eq!(
        (mirror.kv.pages_held(), mirror.kv.bytes_held()),
        (held, bytes)
    );
    let mut again: Vec<usize> = mirror.kv.page_ids().to_vec();
    again.sort_unstable();
    assert_eq!(again, owned, "the same physical pages came back");
    mirror.check_every_kernel(&gpu)
}

#[test]
fn a_window_smaller_than_a_page_works_and_refuses_one_past_it() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for capacity in [1, 40] {
        let mut mirror = Mirror::new(&gpu, SMALL, capacity, KvFormat::F32)?;
        mirror.append(&gpu, capacity - 1)?;
        mirror.check(&gpu, 1)?;
        mirror.append(&gpu, 1)?;
        assert_eq!((mirror.kv.len(), mirror.kv.pages_in_use()), (capacity, 1));
        let extra = gpu.zeros(32)?;
        assert!(
            mirror.kv.write(&gpu, 0, &extra, &extra).is_err(),
            "capacity {capacity}"
        );
        assert!(mirror.kv.advance(1).is_err());
        assert_eq!(mirror.kv.pages_in_use(), 1);
    }
    Ok(())
}
