//! Pages exported from one cache and imported into another must read back identically, whatever physical pages each
//! cache uses. Without a GPU these tests skip; set MARIN_REQUIRE_CUDA=1 to make that a failure.

mod attention_reference;
mod common;
mod kv_round;
mod mirror;

use common::{gpu, values};
use cuda::{DeviceKv, Gpu, KvFormat, PAGE_TOKENS, pages_for};
use mirror::{LAYERS, Mirror, Result, SMALL};

fn export_all(gpu: &Gpu, kv: &DeviceKv) -> Result<Vec<Vec<u8>>> {
    let len = kv.len();
    (0..pages_for(len))
        .map(|page| {
            let rows = PAGE_TOKENS.min(len.saturating_sub(page.saturating_mul(PAGE_TOKENS)));
            Ok(kv.export_page(gpu, page, rows)?)
        })
        .collect()
}

fn import_all(gpu: &Gpu, kv: &mut DeviceKv, pages: &[Vec<u8>], tokens: usize) -> Result<()> {
    kv.reserve_empty(gpu, tokens)?;
    for (page, bytes) in pages.iter().enumerate() {
        let rows = PAGE_TOKENS.min(tokens.saturating_sub(page.saturating_mul(PAGE_TOKENS)));
        kv.import_page(gpu, page, rows, bytes)?;
    }
    kv.advance(tokens)?;
    Ok(())
}

fn attend(gpu: &Gpu, kv: &mut DeviceKv, layer: usize, tokens: usize) -> Result<Vec<f32>> {
    let (heads, kv_heads, head_dim) = SMALL;
    let stride = kv_heads.saturating_mul(head_dim);
    let (k, v) = (
        gpu.upload(&values(tokens.saturating_mul(stride), 71))?,
        gpu.upload(&values(tokens.saturating_mul(stride), 72))?,
    );
    kv.write(gpu, layer, &k, &v)?;
    let q = gpu.upload(&values(
        tokens.saturating_mul(heads).saturating_mul(head_dim),
        73,
    ))?;
    let mut out = gpu.zeros(tokens.saturating_mul(heads).saturating_mul(head_dim))?;
    gpu.attention(&q, kv, layer, heads, tokens, &mut out)?;
    Ok(gpu.download(&out)?)
}

#[test]
fn an_imported_cache_reads_exactly_like_the_one_it_came_from() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for format in KvFormat::ALL {
        for total in [1, 63, 64, 65, 127, 128, 129, 300, 517] {
            let mut source = Mirror::new(&gpu, SMALL, 600, format)?;
            source.append(&gpu, total)?;
            let pages = export_all(&gpu, &source.kv)?;
            let mut copy = DeviceKv::new(&gpu, LAYERS, SMALL, 600, format)?;
            import_all(&gpu, &mut copy, &pages, total)?;
            assert_eq!(copy.len(), total);
            assert_eq!(copy.pages_in_use(), pages_for(total));
            for tokens in [1, 3, 12] {
                for layer in 0..LAYERS {
                    let a = attend(&gpu, &mut source.kv, layer, tokens)?;
                    let b = attend(&gpu, &mut copy, layer, tokens)?;
                    assert_eq!(
                        a, b,
                        "{format:?} {total} tokens, {tokens} new, layer {layer}"
                    );
                    source.kv.truncate(total)?;
                    copy.truncate(total)?;
                }
            }
        }
    }
    Ok(())
}

#[test]
fn pages_can_come_from_a_reversed_page_table() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut source = Mirror::new(&gpu, SMALL, 600, KvFormat::F16)?;
    let tokens = 5 * PAGE_TOKENS + 17;
    source.append(&gpu, tokens)?;
    source.truncate(0)?;
    source.append(&gpu, tokens)?;
    assert_eq!(source.kv.page_ids(), [5, 4, 3, 2, 1, 0]);
    let pages = export_all(&gpu, &source.kv)?;
    let mut copy = DeviceKv::new(&gpu, LAYERS, SMALL, 600, KvFormat::F16)?;
    import_all(&gpu, &mut copy, &pages, tokens)?;
    assert_eq!(copy.page_ids(), [0, 1, 2, 3, 4, 5]);
    assert_eq!(
        attend(&gpu, &mut source.kv, 1, 2)?,
        attend(&gpu, &mut copy, 1, 2)?
    );
    Ok(())
}

#[test]
fn the_payload_has_the_documented_size() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    for (format, per_value) in [(KvFormat::F32, 4), (KvFormat::F16, 2)] {
        let mut mirror = Mirror::new(&gpu, SMALL, 200, format)?;
        mirror.append(&gpu, 100)?;
        let (_, kv_heads, head_dim) = SMALL;
        let page = mirror.kv.export_page(&gpu, 0, 64)?;
        assert_eq!(
            page.len(),
            LAYERS * 2 * 64 * kv_heads * head_dim * per_value
        );
        let tail = mirror.kv.export_page(&gpu, 1, 36)?;
        assert_eq!(
            tail.len(),
            LAYERS * 2 * 36 * kv_heads * head_dim * per_value
        );
        assert_eq!(tail.len(), mirror.kv.page_payload_bytes(36));
    }
    Ok(())
}

#[test]
fn bad_requests_are_refused_and_leave_the_cache_usable() -> Result<()> {
    let Some(gpu) = gpu()? else { return Ok(()) };
    let mut mirror = Mirror::new(&gpu, SMALL, 200, KvFormat::F32)?;
    mirror.append(&gpu, 70)?;
    for (page, rows) in [(0, 0), (0, 65), (1, 7), (2, 1), (9, 1)] {
        assert!(
            mirror.kv.export_page(&gpu, page, rows).is_err(),
            "{page} {rows}"
        );
    }
    let bytes = mirror.kv.export_page(&gpu, 0, 64)?;
    assert!(mirror.kv.reserve_empty(&gpu, 10).is_err(), "not empty");
    assert!(
        mirror.kv.import_page(&gpu, 0, 64, &bytes).is_err(),
        "not empty"
    );
    mirror.check_every_kernel(&gpu)?;

    let mut copy = DeviceKv::new(&gpu, LAYERS, SMALL, 100, KvFormat::F32)?;
    assert!(copy.reserve_empty(&gpu, 101).is_err(), "beyond capacity");
    copy.reserve_empty(&gpu, 64)?;
    assert!(copy.import_page(&gpu, 0, 64, &bytes[1..]).is_err(), "short");
    assert!(
        copy.import_page(&gpu, 1, 64, &bytes).is_err(),
        "page not reserved"
    );
    copy.clear();
    import_all(&gpu, &mut copy, &[bytes], 64)?;
    Ok(())
}
