//! The cache stores keys and values as f16 as well as f32. Each format is checked against an f64 reference
//! fed the values the cache should hold after rounding, so a wrong page, row or scale shows up and rounding itself
//! does not. Without a GPU these tests skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod attention_reference;
mod common;
mod kv_round;
mod mirror;

use common::{gpu, values};
use cuda::{DeviceKv, KvFormat, PAGE_TOKENS, SLAB_PAGES, pages_for};
use kv_round::round_trip;
use mirror::{LAYERS, Mirror, Result, SMALL};

const LOSSY: [KvFormat; 1] = [KvFormat::F16];

#[test]
fn a_single_position_reads_back_exactly_what_the_rounding_gives() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let (heads, kv_heads, head_dim) = (4, 2, 16);
    let stride = kv_heads * head_dim;
    let mut samples = values(stride, 5);
    samples[3] = 0.0;
    samples[17] = 1000.0;
    samples[18] = -7.5e-6;
    for format in KvFormat::ALL {
        let mut kv = DeviceKv::new(&gpu, 1, (heads, kv_heads, head_dim), 8, format)?;
        let zeros = gpu.zeros(stride)?;
        kv.write(&gpu, 0, &zeros, &gpu.upload(&samples)?)?;
        let q = gpu.upload(&values(heads * head_dim, 3))?;
        let mut out = gpu.zeros(heads * head_dim)?;
        gpu.attention(&q, &mut kv, 0, heads, 1, &mut out)?;
        let got = gpu.download(&out)?;
        let held = round_trip(format, &samples);
        for head in 0..heads {
            let kv_head = head / (heads / kv_heads);
            let (a, b) = (head * head_dim, kv_head * head_dim);
            assert_eq!(
                &got[a..a + head_dim],
                &held[b..b + head_dim],
                "{} head {head}",
                format.label()
            );
        }
    }
    Ok(())
}

#[test]
fn values_beyond_the_f16_range_saturate_instead_of_becoming_infinite() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let (heads, kv_heads, head_dim) = (2, 2, 32);
    let mut row = vec![0.0_f32; kv_heads * head_dim];
    row[head_dim + 5] = 1.0e6;
    row[head_dim + 6] = -1.0e6;
    row[head_dim + 7] = 300.0;
    let mut kv = DeviceKv::new(&gpu, 1, (heads, kv_heads, head_dim), 4, KvFormat::F16)?;
    let zeros = gpu.zeros(row.len())?;
    kv.write(&gpu, 0, &zeros, &gpu.upload(&row)?)?;
    let mut out = gpu.zeros(heads * head_dim)?;
    let q = gpu.upload(&values(heads * head_dim, 9))?;
    gpu.attention(&q, &mut kv, 0, heads, 1, &mut out)?;
    let got = gpu.download(&out)?;
    assert!(got[..head_dim].iter().all(|x| *x == 0.0));
    assert_eq!(got[head_dim + 5..head_dim + 8], [65504.0, -65504.0, 300.0]);
    Ok(())
}

#[test]
fn every_format_reads_the_right_page_and_row_at_every_boundary() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for format in LOSSY {
        for total in [0, 1, 63, 64, 65, 127, 128, 129, 193, 330] {
            let mut mirror = Mirror::new(&gpu, SMALL, 400, format)?;
            if total > 0 {
                mirror.append(&gpu, total)?;
            }
            assert_eq!(mirror.kv.pages_in_use(), pages_for(total));
            mirror.check_every_kernel(&gpu)?;
        }
    }
    Ok(())
}

#[test]
fn the_real_head_layout_reads_correctly_in_every_format() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for format in LOSSY {
        for total in [63, 64, 65, 129] {
            let mut mirror = Mirror::new(&gpu, (16, 8, 128), 300, format)?;
            mirror.append(&gpu, total)?;
            mirror.check_every_kernel(&gpu)?;
        }
    }
    Ok(())
}

#[test]
fn rolling_back_and_regrowing_mixes_the_pages_in_every_format() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for format in LOSSY {
        let mut mirror = Mirror::new(&gpu, SMALL, 600, format)?;
        mirror.append(&gpu, 400)?;
        mirror.truncate(70)?;
        mirror.append(&gpu, 330)?;
        assert_eq!(mirror.kv.page_ids(), [0, 1, 6, 5, 4, 3, 2]);
        mirror.check_every_kernel(&gpu)?;
    }
    Ok(())
}

#[test]
fn repeated_appends_and_rollbacks_like_guessing_ahead_stay_exact_in_every_format() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for format in LOSSY {
        let mut mirror = Mirror::new(&gpu, SMALL, 400, format)?;
        mirror.append(&gpu, 50)?;
        let mut state = 0x9e37_79b9_7f4a_7c15_u64;
        for step in 0..80_usize {
            state ^= state.wrapping_shl(13);
            state ^= state.wrapping_shr(7);
            state ^= state.wrapping_shl(17);
            let grow = usize::try_from(state & 7)? + 1;
            let back = usize::try_from(state.wrapping_shr(8) & 7)?.min(grow);
            let room = 400_usize.saturating_sub(mirror.kv.len()).saturating_sub(1);
            mirror.append(&gpu, grow.min(room))?;
            mirror.truncate(mirror.kv.len().saturating_sub(back))?;
            if step % 5 == 0 {
                mirror.check(&gpu, if step % 2 == 0 { 1 } else { 3 })?;
            }
            if mirror.kv.len() > 330 {
                mirror.truncate(60)?;
            }
        }
    }
    Ok(())
}

#[test]
fn a_slab_is_sized_by_the_format_and_a_token_costs_what_the_row_says() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let (kv_heads, head_dim) = (8, 128);
    let mut per_token = Vec::new();
    for format in KvFormat::ALL {
        let mut mirror = Mirror::new(&gpu, (16, kv_heads, head_dim), 8192, format)?;
        let row = format.row_bytes(kv_heads, head_dim).unwrap();
        let page = LAYERS * 2 * PAGE_TOKENS * row;
        assert_eq!(
            (mirror.kv.page_bytes(), mirror.kv.slab_bytes()),
            (page, SLAB_PAGES * page)
        );
        mirror.append(&gpu, 10)?;
        assert_eq!(mirror.kv.bytes_held(), SLAB_PAGES * page);
        assert_eq!(
            DeviceKv::startup_bytes(LAYERS, (16, kv_heads, head_dim), 64, format).unwrap()
                - SLAB_PAGES * page,
            DeviceKv::startup_bytes(LAYERS, (16, kv_heads, head_dim), 64, KvFormat::F32).unwrap()
                - SLAB_PAGES * LAYERS * 2 * PAGE_TOKENS * 4096,
            "only the slab depends on the format"
        );
        per_token.push(2 * row);
    }
    assert_eq!(per_token, [8192, 4096]);
    Ok(())
}

#[test]
fn a_cache_in_one_format_is_independent_of_another() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut a = Mirror::new(&gpu, SMALL, 300, KvFormat::F16)?;
    let mut b = Mirror::new(&gpu, SMALL, 300, KvFormat::F32)?;
    for round in 0..4_usize {
        a.append(&gpu, 37)?;
        b.append(&gpu, 45)?;
        if round % 2 == 1 {
            a.truncate(a.kv.len().saturating_sub(40))?;
        }
        a.check(&gpu, 1)?;
        b.check(&gpu, 3)?;
    }
    Ok(())
}
