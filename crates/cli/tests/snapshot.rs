//! A saved prefix must give exactly what the live cache gives, and anything wrong with the file must fall back to
//! reading the prompt in full. GPU only; without a GPU these tests skip (set MARIN_REQUIRE_CUDA=1 to make that a failure).

mod snapshot_common;

use std::fs;

use cli::snapshot::Reject;
use model::KvFormat;
use runtime::SequenceCache;
use snapshot_common::{Result, Setup, fixture, precision, run, saved, then, tokens, work_dir};

#[test]
fn a_restored_cache_continues_exactly_like_the_live_one() -> Result<()> {
    let dir = work_dir("live")?;
    for kv in [KvFormat::F32, KvFormat::F16] {
        for int8 in [false, true] {
            for temperature in [0.0, 1.2] {
                let file = dir.join("s.bin");
                let mut setup = Setup::new(precision(kv, int8)).at(&file);
                setup.temperature = temperature;
                let Some((mut live, held)) = saved(&setup, &file, &tokens(150, 1))? else {
                    return Ok(());
                };
                let next = then(&live, 7);
                let expected = run(&mut live, &next)?;
                let mut restarted = setup.open()?.unwrap();
                let got = run(&mut restarted, &next)?;
                let label = format!("{kv:?} int8 {int8} temperature {temperature}");
                assert_eq!(got.tokens, expected.tokens, "{label}");
                assert_eq!(
                    got.outcome.reused,
                    held.len(),
                    "{label}: restored everything"
                );
                let restored = restarted.restored.as_ref().unwrap().as_ref().unwrap();
                assert_eq!(restored.ids, held, "{label}");
            }
        }
    }
    Ok(())
}

#[test]
fn restoring_follows_the_page_rule_at_every_boundary() -> Result<()> {
    let dir = work_dir("pages")?;
    let file = dir.join("s.bin");
    let setup = Setup::new(precision(KvFormat::F32, false)).at(&file);
    let cold = Setup::new(precision(KvFormat::F32, false));
    for length in [
        30, 62, 63, 64, 65, 126, 127, 128, 129, 510, 511, 512, 513, 700,
    ] {
        let Some((_, held)) = saved(&setup, &file, &tokens(length, 2))? else {
            return Ok(());
        };
        for cut in [held.len(), held.len() - 1, 129, 128, 127, 65, 64, 63, 5] {
            let cut = cut.min(held.len());
            let mut prompt = held[..cut].to_vec();
            prompt.extend(tokens(6, 40 + cut));
            let mut restarted = setup.open()?.unwrap();
            let got = run(&mut restarted, &prompt)?;
            let expected_rows = if cut >= held.len() {
                held.len()
            } else {
                cut / 64 * 64
            };
            match restarted.restored.as_ref().unwrap() {
                Ok(restored) => {
                    assert_eq!(restored.ids.len(), expected_rows, "{length} cut {cut}");
                    assert!(expected_rows > 0);
                }
                Err(reject) => {
                    assert_eq!(
                        (expected_rows, reject),
                        (0, &Reject::NoMatch),
                        "{length} cut {cut}"
                    );
                }
            }
            let mut fresh = cold.open()?.unwrap();
            assert_eq!(
                got.tokens,
                run(&mut fresh, &prompt)?.tokens,
                "{length} cut {cut}"
            );
        }
    }
    Ok(())
}

fn expect_fallback(setup: &Setup, prompt: &[u32], cold: &[u32], label: &str) -> Result<()> {
    let mut session = setup.open()?.ok_or("no gpu")?;
    let got = run(&mut session, prompt)?;
    let restored = session.restored.as_ref().ok_or("no restore was tried")?;
    assert!(restored.is_err(), "{label}");
    assert_eq!(got.outcome.reused, 0, "{label}");
    assert_eq!(got.tokens, cold, "{label}");
    assert_eq!(session.retained().len(), session.kv().used(), "{label}");
    Ok(())
}

#[test]
fn damaged_or_incomplete_files_fall_back_to_reading_the_prompt() -> Result<()> {
    let dir = work_dir("damage")?;
    let file = dir.join("s.bin");
    let setup = Setup::new(precision(KvFormat::F32, false)).at(&file);
    let Some((live, held)) = saved(&setup, &file, &tokens(150, 3))? else {
        return Ok(());
    };
    let good = fs::read(&file)?;
    let prompt = then(&live, 5);
    let cold = run(
        &mut Setup::new(precision(KvFormat::F32, false)).open()?.unwrap(),
        &prompt,
    )?
    .tokens;
    let pages = 80 + 4 * held.len();
    let n = good.len();
    for cut in [
        0,
        1,
        8,
        79,
        80,
        83,
        pages - 1,
        pages,
        pages + 10,
        n / 2,
        n - 17,
        n - 16,
        n - 8,
        n - 1,
    ] {
        fs::write(&file, &good[..cut])?;
        expect_fallback(&setup, &prompt, &cold, &format!("cut at {cut}"))?;
    }
    for at in [
        0,
        9,
        20,
        40,
        60,
        79,
        80,
        80 + 4 * held.len() - 1,
        pages,
        pages + 100,
        n / 2,
        n - 25,
        n - 20,
        n - 9,
        n - 3,
    ] {
        let mut bad = good.clone();
        bad[at] ^= 0x10;
        fs::write(&file, &bad)?;
        expect_fallback(&setup, &prompt, &cold, &format!("flipped byte {at}"))?;
    }
    let mut longer = good.clone();
    longer.push(0);
    fs::write(&file, &longer)?;
    expect_fallback(&setup, &prompt, &cold, "extra byte")?;
    fs::remove_file(&file)?;
    expect_fallback(&setup, &prompt, &cold, "missing file")?;
    fs::write(&file, &good)?;
    let mut session = setup.open()?.unwrap();
    run(&mut session, &prompt)?;
    assert!(
        session.restored.as_ref().unwrap().is_ok(),
        "the untouched file works"
    );
    Ok(())
}

#[test]
fn a_snapshot_from_another_setup_is_refused() -> Result<()> {
    let dir = work_dir("foreign")?;
    let file = dir.join("s.bin");
    let base = Setup::new(precision(KvFormat::F32, false)).at(&file);
    let Some((live, _)) = saved(&base, &file, &tokens(100, 4))? else {
        return Ok(());
    };
    let prompt = then(&live, 5);
    let reject = |setup: Setup| -> Result<Reject> {
        let mut session = setup.open()?.unwrap();
        run(&mut session, &prompt)?;
        Ok(session
            .restored
            .take()
            .unwrap()
            .err()
            .expect("must be refused"))
    };
    assert_eq!(
        reject(Setup::new(precision(KvFormat::F16, false)).at(&file))?,
        Reject::Foreign("KV type")
    );
    assert_eq!(
        reject(Setup::new(precision(KvFormat::F32, true)).at(&file))?,
        Reject::Foreign("weight quantization")
    );
    let other = dir.join("other");
    fs::create_dir_all(&other)?;
    for entry in fs::read_dir(fixture())? {
        let entry = entry?;
        fs::copy(entry.path(), other.join(entry.file_name()))?;
    }
    let with_edit = |name: &str, change: &dyn Fn(&mut Vec<u8>)| -> Result<Reject> {
        let original = fs::read(fixture().join(name))?;
        let mut edited = original.clone();
        change(&mut edited);
        fs::write(other.join(name), &edited)?;
        let mut setup = Setup::new(precision(KvFormat::F32, false)).at(&file);
        setup.model = other.clone();
        let rejected = reject(setup);
        fs::write(other.join(name), original)?;
        rejected
    };
    let space = |bytes: &mut Vec<u8>| bytes.push(b' ');
    let config = Reject::Foreign("model configuration or tokenizer");
    assert_eq!(with_edit("config.json", &space)?, config);
    assert_eq!(with_edit("tokenizer.json", &space)?, config);
    assert_eq!(
        with_edit("model.safetensors", &|bytes| {
            let last = bytes.len() - 5;
            bytes[last] ^= 0x01;
        })?,
        Reject::Foreign("model weights")
    );
    let mut same = Setup::new(precision(KvFormat::F32, false)).at(&file);
    same.model = other;
    let mut session = same.open()?.unwrap();
    run(&mut session, &prompt)?;
    assert!(
        session.restored.as_ref().unwrap().is_ok(),
        "identical files in another folder"
    );
    Ok(())
}
