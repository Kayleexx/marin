use model::qwen35::Qwen35;
use model::{Backend, Engine, QuantBits, QuantScheme, export_int8};
use runtime::{CausalLm, SequenceCache};
use std::fs;
use std::path::{Path, PathBuf};

type Result<T = ()> = std::result::Result<T, Box<dyn std::error::Error>>;
struct Temp(PathBuf);
impl Temp {
    fn new() -> Result<Self> {
        let stamp = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)?
            .as_nanos();
        let p =
            std::env::temp_dir().join(format!("marin-hybrid-q8-{}-{stamp}", std::process::id()));
        fs::create_dir(&p)?;
        Ok(Self(p))
    }
}
impl Drop for Temp {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}
fn fixture() -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR")).join("../../tests/fixtures/tiny-qwen35-int8")
}
fn scheme() -> Result<QuantScheme> {
    Ok(QuantScheme::new(QuantBits::Int8, 32)?)
}
#[test]
fn direct_packed_matches_inline_with_reuse_replay_reset_and_threads() -> Result {
    let temp = Temp::new()?;
    let source = loader::ModelDir::open(&fixture())?;
    let path = temp.0.join("q8");
    export_int8(&source, &path, 32)?;
    let packed = loader::ModelDir::open(&path)?;
    assert!(!packed.weights.names().any(|n| {
        packed
            .weights
            .info(n)
            .is_some_and(|i| i.dtype == loader::Dtype::Bf16)
    }));
    for name in source.weights.names().filter(|n| {
        n.ends_with("in_proj_a.weight")
            || n.ends_with("in_proj_b.weight")
            || n.ends_with("conv1d.weight")
            || n.ends_with("A_log")
    }) {
        let values = if source.weights.info(name).ok_or("missing")?.dtype == loader::Dtype::F32 {
            source.weights.read_f32(name)?
        } else {
            source
                .weights
                .read_bf16(name)?
                .into_iter()
                .map(tensor::bf16_to_f32)
                .collect()
        };
        assert_eq!(packed.weights.read_f32(name)?, values);
    }
    assert!(Qwen35::load(&packed, |_, _| {}).is_err());
    assert!(
        Qwen35::load_quantized(
            &packed,
            Some(QuantScheme::new(QuantBits::Int8, 64)?),
            |_, _| {}
        )
        .is_err()
    );
    let a = Qwen35::load_quantized(&source, Some(scheme()?), |_, _| {})?;
    let b = Qwen35::load_quantized(&packed, Some(scheme()?), |_, _| {})?;
    for threads in [1, 8] {
        tensor::set_threads(threads);
        let mut x = a.new_cache(128)?;
        let mut y = b.new_cache(128)?;
        let untouched = b.new_cache(128)?;
        let ids: Vec<_> = (1..66).collect();
        assert_eq!(a.forward(&ids, &mut x)?, b.forward(&ids, &mut y)?);
        for token in [4, 17, 31] {
            assert_eq!(a.forward(&[token], &mut x)?, b.forward(&[token], &mut y)?);
        }
        let used = y.used();
        assert!(b.forward(&[320], &mut y).is_err());
        assert_eq!(y.used(), used);
        assert!(b.forward(&[1; 100], &mut y).is_err());
        assert_eq!(y.used(), used);
        assert_eq!(untouched.used(), 0);
        x.truncate(17)?;
        y.truncate(17)?;
        assert_eq!(a.forward(&[2; 16], &mut x)?, b.forward(&[2; 16], &mut y)?);
        x.clear();
        y.clear();
        assert_eq!(a.forward(&ids, &mut x)?, b.forward(&ids, &mut y)?);
    }
    Ok(())
}
#[test]
fn corruption_and_wrong_architecture_are_refused_without_overwriting_source() -> Result {
    let temp = Temp::new()?;
    let source = loader::ModelDir::open(&fixture())?;
    let path = temp.0.join("q8");
    export_int8(&source, &path, 32)?;
    assert!(export_int8(&source, &path, 32).is_err());
    let file = path.join("model.safetensors");
    let mut bytes = fs::read(&file)?;
    let byte = bytes.last_mut().ok_or("empty artifact")?;
    *byte ^= 1;
    fs::write(file, bytes)?;
    let packed = loader::ModelDir::open(&path)?;
    assert!(Qwen35::load_quantized(&packed, Some(scheme()?), |_, _| {}).is_err());
    let manifest = path.join(loader::packed::MANIFEST);
    let mut data: serde_json::Value = serde_json::from_slice(&fs::read(&manifest)?)?;
    *data.get_mut("architecture").ok_or("architecture")? = "qwen3".into();
    fs::write(&manifest, serde_json::to_vec(&data)?)?;
    assert!(loader::ModelDir::open(&path).is_err());
    fs::remove_file(manifest)?;
    let unmarked = loader::ModelDir::open(&path)?;
    assert!(model::qwen35::weight_bytes(&unmarked.weights, None).is_err());
    Ok(())
}

#[test]
fn cuda_history_checkpoint_restores_delta_and_attention_state() -> Result {
    let gpu = match model::Gpu::new() {
        Ok(gpu) => gpu,
        Err(error) if std::env::var_os("MARIN_REQUIRE_CUDA").is_none() => {
            eprintln!("CUDA unavailable: {error}");
            return Ok(());
        }
        Err(error) => return Err(error.into()),
    };
    let temp = Temp::new()?;
    let source = loader::ModelDir::open(&fixture())?;
    let path = temp.0.join("q8");
    export_int8(&source, &path, 32)?;
    let packed = loader::ModelDir::open(&path)?;
    let engine = Engine::load(
        &packed,
        Backend::Cuda(Box::new(gpu)),
        Some(scheme()?),
        |_, _| {},
    )?;
    let mut live = engine.new_cache(128)?;
    let mut fresh = engine.new_cache(128)?;
    let history: Vec<u32> = (1..66).collect();
    assert_eq!(
        engine.forward(&history, &mut live)?,
        engine.forward(&history, &mut fresh)?
    );
    let checkpoint = engine
        .session_checkpoint(&live)?
        .ok_or("checkpoint refused")?;
    engine.forward(&[11, 12, 13], &mut live)?;
    live = checkpoint;
    assert_eq!(live.used(), history.len());
    assert_eq!(
        engine.forward(&[21, 22], &mut live)?,
        engine.forward(&[21, 22], &mut fresh)?
    );
    assert_eq!(
        engine.forward(&[23], &mut live)?,
        engine.forward(&[23], &mut fresh)?
    );
    Ok(())
}
