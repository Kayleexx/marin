//! Snapshot saving and restoring must never leave half a state behind. GPU only; without a GPU these tests skip.

mod snapshot_common;

use std::fs;

use cli::load::load_for_snapshot;
use cli::session::{Session, Settings};
use cli::snapshot::{self, Reject};
use cli::style::Style;
use model::{BackendChoice, KvFormat, Precision};
use runtime::SequenceCache;
use snapshot_common::{Result, Setup, fixture, precision, run, saved, then, tokens, work_dir};

#[test]
fn a_failure_in_the_last_page_leaves_the_cache_empty() -> Result<()> {
    let dir = work_dir("partial")?;
    let file = dir.join("s.bin");
    let setup = Setup::new(precision(KvFormat::F32, false)).at(&file);
    let Some((live, held)) = saved(&setup, &file, &tokens(200, 5))? else {
        return Ok(());
    };
    let mut bytes = fs::read(&file)?;
    let last_page_byte = bytes.len() - 16 - 8 - 5;
    bytes[last_page_byte] ^= 0x40;
    fs::write(&file, &bytes)?;
    let prompt = then(&live, 5);

    let mut session = Setup::new(precision(KvFormat::F32, false)).open()?.unwrap();
    let hash = session
        .loaded
        .weights_hash
        .take()
        .unwrap()
        .join()
        .unwrap()?;
    let identity = snapshot::identity(&session.loaded.model, &session.loaded.dir, hash).unwrap();
    let mut cache = session.loaded.model.new_cache(1024)?;
    let failed = snapshot::restore(&file, &session.loaded.model, &mut cache, &prompt, &identity);
    assert_eq!(failed.err(), Some(Reject::Damaged("page data")));
    assert_eq!(
        cache.used(),
        0,
        "pages 0 and 1 were copied, and none of it is kept"
    );

    let mut session = setup.open()?.unwrap();
    run(&mut session, &prompt)?;
    assert_eq!(
        session.restored.as_ref().unwrap().as_ref().err(),
        Some(&Reject::Damaged("page data"))
    );
    assert_eq!(session.retained().len(), session.kv().used());
    assert_eq!(
        session.last.as_ref().unwrap().reused,
        0,
        "{} positions were not trusted",
        held.len()
    );
    Ok(())
}

#[test]
fn saving_is_all_or_nothing() -> Result<()> {
    let dir = work_dir("save")?;
    let file = dir.join("s.bin");
    let setup = Setup::new(precision(KvFormat::F16, false)).at(&file);
    let Some((mut session, held)) = saved(&setup, &file, &tokens(150, 6))? else {
        return Ok(());
    };
    let good = fs::read(&file)?;

    let hash = session
        .loaded
        .dir
        .weights
        .spawn_content_hash()
        .join()
        .unwrap()?;
    let identity = snapshot::identity(&session.loaded.model, &session.loaded.dir, hash).unwrap();
    let mut too_many = held.clone();
    too_many.extend(tokens(100, 3));
    let failed = snapshot::save(
        &file,
        &session.loaded.model,
        session.kv(),
        &too_many,
        identity,
    );
    assert!(
        failed.is_err(),
        "the cache cannot supply pages it does not have"
    );
    let fewer = &held[..held.len() - 3];
    let failed = snapshot::save(&file, &session.loaded.model, session.kv(), fewer, identity);
    assert!(
        failed.is_err(),
        "a record that disagrees with the cache is not saved"
    );
    assert_eq!(fs::read(&file)?, good, "the old snapshot is untouched");
    let leftovers: Vec<_> = fs::read_dir(&dir)?.collect();
    assert_eq!(leftovers.len(), 1, "no temporary file is left behind");

    fs::write(dir.join("s.bin.99999.tmp"), b"junk left by a crashed save")?;
    let prompt = then(&session, 5);
    let mut restarted = setup.open()?.unwrap();
    run(&mut restarted, &prompt)?;
    assert!(
        restarted.restored.as_ref().unwrap().is_ok(),
        "a stale temporary file is ignored"
    );

    session.engine_and_kv();
    fs::remove_file(&file)?;
    assert!(
        session
            .save_snapshot()
            .map_err(|e| e.to_string())?
            .is_none(),
        "an untrusted cache is not saved"
    );
    assert!(!file.exists());

    let blocked = dir.join("missing-folder").join("s.bin");
    let mut elsewhere = Setup::new(precision(KvFormat::F16, false));
    elsewhere.snapshot = Some(&blocked);
    let mut third = elsewhere.open()?.unwrap();
    run(&mut third, &tokens(100, 8))?;
    assert!(third.save_snapshot().is_err(), "no folder, no snapshot");
    assert!(!blocked.exists());
    Ok(())
}

/// A prompt whose greedy first reply token completes an eight-token match with earlier text, so guessing starts at once.
fn guessing_prompt(session: &mut Session) -> Result<Vec<u32>> {
    let filler = tokens(40, 5);
    for seed in 0..6 {
        let tail = tokens(7, 70_usize.saturating_add(seed));
        for next in 5..320_u32 {
            let mut prompt = tail.clone();
            prompt.push(next);
            prompt.extend(&filler);
            prompt.extend(&tail);
            let logits = session.next_logits(&prompt)?;
            let best = logits
                .iter()
                .enumerate()
                .max_by(|a, b| a.1.total_cmp(b.1))
                .map(|(id, _)| id);
            if best == Some(next as usize) {
                return Ok(prompt);
            }
        }
    }
    Err("no prompt makes the model start a match".into())
}

#[test]
fn guessing_ahead_before_the_save_changes_nothing() -> Result<()> {
    let dir = work_dir("guess")?;
    let file = dir.join("s.bin");
    for kv in [KvFormat::F32, KvFormat::F16] {
        let mut setup = Setup::new(precision(kv, true)).at(&file);
        setup.speculate = true;
        setup.max_tokens = 60;
        let Some(mut live) = setup.open()? else {
            return Ok(());
        };
        let prompt = guessing_prompt(&mut live)?;
        let first = run(&mut live, &prompt)?;
        let stats = &first.outcome.speculation;
        assert!(
            stats.guessed_passes > 0 && stats.drafted > stats.accepted,
            "{kv:?}: guessing must have happened and been rolled back: {stats:?}"
        );
        assert!(live.save_snapshot().map_err(|e| e.to_string())?.is_some());
        let next = then(&live, 4);
        let expected = run(&mut live, &next)?;
        let mut restarted = setup.open()?.unwrap();
        assert_eq!(
            run(&mut restarted, &next)?.tokens,
            expected.tokens,
            "{kv:?}"
        );
        assert!(restarted.restored.as_ref().unwrap().is_ok());
    }
    Ok(())
}

#[test]
fn the_cpu_never_saves_or_restores() -> Result<()> {
    let dir = work_dir("cpu")?;
    let file = dir.join("s.bin");
    let loaded = load_for_snapshot(
        &fixture(),
        Some(256),
        256,
        BackendChoice::Cpu,
        Precision::default(),
        &Style::plain(),
        false,
    )?;
    let settings = Settings {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(1),
    };
    let mut session = Session::new(loaded, settings, Style::plain())?;
    session.use_snapshot(file.clone());
    run(&mut session, &tokens(70, 1))?;
    assert!(
        session
            .save_snapshot()
            .map_err(|e| e.to_string())?
            .is_none()
    );
    assert!(!file.exists() && session.restored.is_none());
    Ok(())
}
