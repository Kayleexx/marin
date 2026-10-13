mod snapshot_common;

use cli::snapshot::{self, Reject};
use model::KvFormat;
use runtime::SequenceCache;
use snapshot_common::{Result, Setup, fixture, precision, run, saved, then, tokens, work_dir};
use std::fs;

#[test]
fn packed_gpu_snapshots_restart_restore_and_reject_foreign_weights() -> Result<()> {
    let dir = work_dir("packed")?;
    let packed = dir.join("model");
    model::export_int8(&loader::ModelDir::open(&fixture())?, &packed, 32)?;
    for format in [KvFormat::F32, KvFormat::F16] {
        let file = dir.join(format!("{format:?}.bin"));
        let mut setup = Setup::new(precision(format, true)).at(&file);
        setup.model = packed.clone();
        let Some((mut live, held)) = saved(&setup, &file, &tokens(150, 3))? else {
            return Ok(());
        };
        let prompt = then(&live, 5);
        let expected = run(&mut live, &prompt)?.tokens;
        let mut restarted = setup.open()?.ok_or("CUDA required")?;
        let got = run(&mut restarted, &prompt)?.tokens;
        assert_eq!(got, expected);
        assert!(restarted.restored.as_ref().ok_or("restore result")?.is_ok());
        assert!(restarted.last.as_ref().ok_or("reply")?.reused > 0);

        let hash = restarted
            .loaded
            .dir
            .weights
            .spawn_content_hash()
            .join()
            .map_err(|_| "hash thread")??;
        let identity = snapshot::identity(&restarted.loaded.model, &restarted.loaded.dir, hash)
            .ok_or("identity")?;
        let good = fs::read(&file)?;
        let mut bytes = good.clone();
        let n = bytes.len() - 24;
        bytes[n] ^= 1;
        fs::write(&file, bytes)?;
        let mut cache = restarted.loaded.model.new_cache(1024)?;
        assert!(matches!(
            snapshot::restore(
                &file,
                &restarted.loaded.model,
                &mut cache,
                &prompt,
                &identity
            ),
            Err(Reject::Damaged(_))
        ));
        assert_eq!(cache.used(), 0);
        fs::write(&file, good)?;
        let foreign = Setup::new(precision(format, true)).at(&file);
        let mut session = foreign.open()?.ok_or("CUDA required")?;
        run(&mut session, &prompt)?;
        assert!(session.restored.as_ref().ok_or("foreign restore")?.is_err());
        assert_eq!(session.last.as_ref().ok_or("reply")?.reused, 0);
        assert!(!held.is_empty());
    }
    Ok(())
}
