use crate::DeviceKv;
use crate::pages::{PAGE_TOKENS, PagePool, PageTable, PagesExhausted, SLAB_PAGES, pages_for};

fn grown(pool: &mut PagePool, table: &mut PageTable, tokens: usize) -> usize {
    table.grow_to(pool, pages_for(tokens)).unwrap()
}

#[test]
fn pages_are_counted_up_at_every_page_boundary() {
    for (tokens, pages) in [
        (0, 0),
        (1, 1),
        (63, 1),
        (64, 1),
        (65, 2),
        (127, 2),
        (128, 2),
        (129, 3),
        (4096, 64),
        (4097, 65),
    ] {
        assert_eq!(pages_for(tokens), pages, "{tokens} tokens");
    }
}

#[test]
fn the_page_size_matches_the_one_the_kernels_were_written_for() {
    let kernels = include_str!("attention.cu");
    assert!(kernels.contains("#define PAGE_SHIFT 6"));
    assert_eq!(PAGE_TOKENS, 1 << 6);
    assert!(kernels.contains(&format!("#define SLAB_PAGES {SLAB_PAGES}")));
}

#[test]
fn a_fresh_pool_hands_pages_out_in_order_and_a_table_only_grows_to_what_it_needs() {
    let mut pool = PagePool::new(8);
    let mut table = PageTable::default();
    assert_eq!(grown(&mut pool, &mut table, 0), 0);
    assert!(table.is_empty());
    assert_eq!(grown(&mut pool, &mut table, 1), 0);
    assert_eq!(table.ids(), [0]);
    assert_eq!(
        grown(&mut pool, &mut table, 64),
        1,
        "64 tokens still fit one page"
    );
    assert_eq!(table.ids(), [0]);
    assert_eq!(grown(&mut pool, &mut table, 65), 1);
    assert_eq!(table.ids(), [0, 1]);
    assert_eq!(grown(&mut pool, &mut table, 200), 2);
    assert_eq!(table.ids(), [0, 1, 2, 3]);
    assert_eq!(
        grown(&mut pool, &mut table, 10),
        4,
        "never shrinks by itself"
    );
    assert_eq!((pool.issued(), pool.available()), (4, 4));
}

#[test]
fn shrinking_gives_back_whole_pages_only_and_never_a_page_still_in_use() {
    let mut pool = PagePool::new(8);
    let mut table = PageTable::default();
    grown(&mut pool, &mut table, 3 * PAGE_TOKENS);
    table.shrink_to(&mut pool, pages_for(3 * PAGE_TOKENS - 1));
    assert_eq!(table.len(), 3, "63 tokens into the last page keeps it");
    table.shrink_to(&mut pool, pages_for(2 * PAGE_TOKENS));
    assert_eq!(table.len(), 2);
    assert_eq!(pool.available(), 6);
    table.shrink_to(&mut pool, pages_for(2 * PAGE_TOKENS + 1));
    assert_eq!(
        table.len(),
        2,
        "shrinking to more than held changes nothing"
    );
    table.shrink_to(&mut pool, 0);
    assert!(table.is_empty());
    assert_eq!((pool.issued(), pool.available()), (3, 8));
}

#[test]
fn freed_pages_are_reused_before_new_ones_are_issued() {
    let mut pool = PagePool::new(8);
    let mut table = PageTable::default();
    grown(&mut pool, &mut table, 5 * PAGE_TOKENS);
    for _ in 0..10 {
        table.shrink_to(&mut pool, 1);
        grown(&mut pool, &mut table, 5 * PAGE_TOKENS);
        assert_eq!(pool.issued(), 5, "no page beyond the five ever issued");
    }
}

#[test]
fn a_cleared_and_refilled_sequence_gets_its_pages_back_in_reverse_order() {
    let mut pool = PagePool::new(8);
    let mut table = PageTable::default();
    grown(&mut pool, &mut table, 5 * PAGE_TOKENS);
    assert_eq!(table.ids(), [0, 1, 2, 3, 4]);
    table.shrink_to(&mut pool, 0);
    grown(&mut pool, &mut table, 5 * PAGE_TOKENS);
    assert_eq!(table.ids(), [4, 3, 2, 1, 0]);
    table.shrink_to(&mut pool, 2);
    grown(&mut pool, &mut table, 5 * PAGE_TOKENS);
    assert_eq!(table.ids(), [4, 3, 0, 1, 2], "a mix, still every page once");
}

#[test]
fn running_out_of_pages_is_an_error_that_changes_nothing() {
    let mut pool = PagePool::new(pages_for(130));
    let mut table = PageTable::default();
    grown(&mut pool, &mut table, 130);
    assert_eq!(table.len(), 3);
    let before = (table.clone(), pool.clone());
    assert_eq!(
        table.grow_to(&mut pool, 4),
        Err(PagesExhausted {
            needed: 1,
            available: 0
        })
    );
    assert_eq!((table, pool), before);
}

#[test]
fn two_sequences_share_one_pool_without_ever_sharing_a_page() {
    let mut pool = PagePool::new(8);
    let (mut a, mut b) = (PageTable::default(), PageTable::default());
    for tokens in [65, 129, 193] {
        grown(&mut pool, &mut a, tokens);
        grown(&mut pool, &mut b, tokens.saturating_sub(64));
    }
    let mut all: Vec<usize> = a.ids().iter().chain(b.ids()).copied().collect();
    all.sort_unstable();
    all.dedup();
    assert_eq!(all.len(), a.len() + b.len());
    assert_eq!((a.len(), b.len(), pool.available()), (4, 3, 1));
    assert!(
        b.grow_to(&mut pool, 8).is_err(),
        "a holds what b would need"
    );
    a.shrink_to(&mut pool, 0);
    b.grow_to(&mut pool, 8).unwrap();
    assert_eq!(
        (b.len(), pool.available()),
        (8, 0),
        "b may now take every page"
    );
    assert!(b.grow_to(&mut pool, 9).is_err());
}

struct Rng(u64);

impl Rng {
    fn below(&mut self, n: usize) -> usize {
        self.0 ^= self.0 << 13;
        self.0 ^= self.0 >> 7;
        self.0 ^= self.0 << 17;
        let bound = u64::try_from(n).unwrap();
        usize::try_from(self.0.checked_rem(bound).unwrap()).unwrap()
    }
}

#[test]
fn random_use_keeps_every_page_owned_exactly_once_and_counts_exact() {
    let capacity = 700;
    let mut pool = PagePool::new(pages_for(capacity) * 2);
    let mut tables = [PageTable::default(), PageTable::default()];
    let mut tokens = [0_usize; 2];
    let mut rng = Rng(0x9e37_79b9_7f4a_7c15);
    for step in 0..5000 {
        let who = rng.below(2);
        let (Some(table), Some(held)) = (tables.get_mut(who), tokens.get_mut(who)) else {
            continue;
        };
        match rng.below(4) {
            0 | 1 => {
                let to = held.saturating_add(rng.below(150)).min(capacity);
                table.grow_to(&mut pool, pages_for(to)).unwrap();
                *held = to;
            }
            2 => {
                let to = rng.below(held.saturating_add(1));
                table.shrink_to(&mut pool, pages_for(to));
                *held = to;
            }
            _ => {
                table.shrink_to(&mut pool, 0);
                *held = 0;
            }
        }
        let mut seen = Vec::new();
        for (table, held) in tables.iter().zip(tokens) {
            assert_eq!(table.len(), pages_for(held), "step {step}");
            seen.extend_from_slice(table.ids());
        }
        let owned = seen.len();
        seen.sort_unstable();
        seen.dedup();
        assert_eq!(seen.len(), owned, "a page is owned twice at step {step}");
        assert!(seen.iter().all(|id| *id < pool.issued()));
        assert_eq!(
            pool.available(),
            pool.limit().saturating_sub(owned),
            "step {step}"
        );
    }
}

#[test]
fn startup_needs_one_slab_and_the_scratch_not_the_window() {
    let dims = (16, 8, 128);
    let slab = 28 * 2 * SLAB_PAGES * PAGE_TOKENS * 8 * 128 * 4;
    let window = 8192 * 28 * 2 * 8 * 128 * 4;
    let startup = DeviceKv::startup_bytes(28, dims, 8192).unwrap();
    assert!(
        startup > slab && startup < slab + 16 * 1024 * 1024,
        "{startup}"
    );
    assert!(
        startup < window / 8,
        "{startup} against a window of {window}"
    );
    assert!(
        DeviceKv::startup_bytes(28, dims, 64).unwrap() > slab,
        "a short window still takes a whole slab"
    );
    assert_eq!(DeviceKv::startup_bytes(usize::MAX, dims, 8192), None);
}
