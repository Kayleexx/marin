//! A checkpoint written from one session loads into another and then continues bit for bit; a file that is damaged,
//! cut, extended or made for something else is refused and leaves the session usable. Needs a GPU.

#[path = "common/qwen35.rs"]
mod qwen35;

use std::fs::{self, File};
use std::path::Path;

use cli::checkpoint::{self, Model, Source};
use cli::load::{ChatPrecision, load_chat};
use cli::session::{Session, Settings};
use cli::setup::precision_for_pick;
use cli::snapshot::Reject;
use cli::style::Style;
use model::BackendChoice;
use qwen35::{Res, Scratch, have_gpu, packed_tiny};
use runtime::{CausalLm, SequenceCache};

fn session(packed: &Path, ctx: usize) -> Res<Session> {
    let asked = ChatPrecision {
        weights: None,
        kv: None,
    };
    let loaded = load_chat(
        packed,
        (Some(ctx), ctx),
        BackendChoice::Cuda,
        precision_for_pick(asked, packed),
        &Style::plain(),
        (false, false),
    )?;
    let settings = Settings {
        max_tokens: 4,
        temperature: 0.0,
        top_k: 0,
        top_p: 1.0,
        seed: Some(1),
    };
    Ok(Session::new(loaded, settings, Style::plain())?)
}

fn source() -> Source {
    Source {
        session: "work".into(),
        revision: 3,
        mode: "chat".into(),
        model: Model {
            name: "tiny".into(),
            config: "c".into(),
            weights: "w".into(),
            weight_bits: 8,
            weight_group: 32,
        },
    }
}

fn history() -> Vec<u32> {
    (10..).take(300).map(|n: u32| n % 300).collect()
}

fn fill(session: &mut Session, ids: &[u32]) -> Res {
    let (engine, kv) = session.engine_and_kv();
    engine.forward(ids, kv)?;
    Ok(())
}

fn written(session: &Session, dir: &Path) -> Res<Vec<u8>> {
    let path = dir.join("state");
    checkpoint::write(session, &source(), &mut File::create(&path)?)?;
    Ok(fs::read(path)?)
}

fn restore(
    session: &mut Session,
    dir: &Path,
    bytes: &[u8],
    expect: &Source,
) -> Result<Vec<u32>, Reject> {
    let path = dir.join("copy");
    fs::write(&path, bytes).map_err(|e| Reject::Failed(e.to_string()))?;
    let file = File::open(&path).map_err(|e| Reject::Failed(e.to_string()))?;
    checkpoint::restore(session, file, bytes.len() as u64, expect)
}

#[test]
fn a_restored_session_continues_bit_for_bit_and_a_bad_file_is_refused() -> Res {
    if !have_gpu()? {
        return Ok(());
    }
    let scratch = Scratch::new("ckpt")?;
    let dir = scratch.0.clone();
    let packed = packed_tiny(&dir)?;
    (|| -> Res {
        let ids = history();
        let mut original = session(&packed, 512)?;
        fill(&mut original, &ids)?;
        let bytes = written(&original, &dir)?;

        let mut copy = session(&packed, 512)?;
        assert_eq!(
            restore(&mut copy, &dir, &bytes, &source()).as_deref(),
            Ok(ids.as_slice())
        );
        assert_eq!(
            (copy.kv().used(), copy.retained().tokens()),
            (300, ids.as_slice())
        );
        let tail = [200_u32, 201, 202];
        let (a, b) = (original.engine_and_kv(), copy.engine_and_kv());
        assert!(
            a.0.forward(&tail, a.1)? == b.0.forward(&tail, b.1)?,
            "bit for bit"
        );

        let mut spare = session(&packed, 512)?;
        let mut refused = |bytes: &[u8], expect: &Source, what: &str| -> Res {
            let result = restore(&mut spare, &dir, bytes, expect);
            assert!(result.is_err(), "{what} was accepted");
            assert_eq!(spare.kv().used(), 0, "{what} left something in the cache");
            fill(&mut spare, &[1, 2, 3])?;
            spare.release_cache()?;
            Ok(())
        };
        for at in [
            0,
            9,
            14,
            20,
            100,
            bytes.len() / 3,
            bytes.len() / 2,
            bytes.len() - 20,
            bytes.len() - 1,
        ] {
            let mut flipped = bytes.clone();
            if let Some(byte) = flipped.get_mut(at) {
                *byte ^= 0x40;
            }
            refused(&flipped, &source(), &format!("a changed byte at {at}"))?;
        }
        for len in [0, 8, 16, 200, bytes.len() / 2, bytes.len() - 1] {
            refused(
                bytes.get(..len).unwrap_or_default(),
                &source(),
                &format!("{len} bytes"),
            )?;
        }
        let mut longer = bytes.clone();
        longer.push(0);
        refused(&longer, &source(), "a trailing byte")?;
        for (what, change) in [
            (
                "another revision",
                Source {
                    revision: 4,
                    ..source()
                },
            ),
            (
                "another session",
                Source {
                    session: "other".into(),
                    ..source()
                },
            ),
            (
                "another mode",
                Source {
                    mode: "code".into(),
                    ..source()
                },
            ),
            (
                "another model",
                Source {
                    model: Model {
                        name: "big".into(),
                        ..source().model
                    },
                    ..source()
                },
            ),
            (
                "other weights",
                Source {
                    model: Model {
                        weights: "x".into(),
                        ..source().model
                    },
                    ..source()
                },
            ),
            (
                "other precision",
                Source {
                    model: Model {
                        weight_bits: 4,
                        ..source().model
                    },
                    ..source()
                },
            ),
            (
                "other configuration",
                Source {
                    model: Model {
                        config: "x".into(),
                        ..source().model
                    },
                    ..source()
                },
            ),
        ] {
            refused(&bytes, &change, what)?;
        }
        let mut small = session(&packed, 256)?;
        let result = restore(&mut small, &dir, &bytes, &source());
        assert!(
            matches!(result, Err(Reject::Foreign(_))),
            "300 tokens do not fit 256: {result:?}"
        );

        let mut empty = session(&packed, 512)?;
        let mut sink = File::create(dir.join("none"))?;
        assert!(
            checkpoint::write(&empty, &source(), &mut sink).is_err(),
            "an empty cache is not kept"
        );
        fill(&mut empty, &ids)?;
        empty.engine_and_kv().1.truncate(100)?;
        assert!(
            checkpoint::write(&empty, &source(), &mut sink).is_err(),
            "a cache that must be replayed is not kept"
        );
        Ok(())
    })()
}
