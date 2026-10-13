use std::collections::BTreeMap;
use std::io::Read;

use sha2::{Digest, Sha256};

use crate::packed::{bad, io};
use crate::{Dtype, LoaderError, Weights};

impl Weights {
    pub fn packed_group(&self) -> Option<usize> {
        self.packed.as_ref().map(|m| m.group)
    }

    pub(crate) fn validate_packed(&self) -> Result<(), LoaderError> {
        let Some(manifest) = &self.packed else {
            return Ok(());
        };
        if self.files.len() != 1
            || self.owner.len() != manifest.tensors.len()
            || self
                .names()
                .any(|name| !manifest.tensors.contains_key(name))
        {
            return Err(bad("tensor digest/name mismatch or sharded artifact"));
        }
        let file = self
            .files
            .first()
            .ok_or_else(|| bad("missing artifact weights"))?;
        let mut spans: Vec<_> = file.tensors().values().collect();
        spans.sort_by_key(|i| i.begin);
        let mut end = 0;
        for info in spans {
            if info.begin != end || !matches!(info.dtype, Dtype::I8 | Dtype::F32) {
                return Err(bad("overlapping/noncontiguous spans or incompatible dtype"));
            }
            end = info.end;
        }
        if end != file.data_len()? {
            return Err(bad("trailing/unindexed payload"));
        }
        Ok(())
    }

    fn packed_chunks(
        &self,
        name: &str,
        mut use_chunk: impl FnMut(&[u8]),
    ) -> Result<(), LoaderError> {
        let manifest = self
            .packed
            .as_ref()
            .ok_or_else(|| bad("not a packed artifact"))?;
        let want = manifest
            .tensors
            .get(name)
            .ok_or_else(|| bad("missing tensor digest"))?;
        let file = self
            .owner
            .get(name)
            .and_then(|i| self.files.get(*i))
            .ok_or_else(|| bad("unknown tensor"))?;
        let info = self.info(name).ok_or_else(|| bad("unknown tensor"))?;
        let mut read = file.reader(name)?;
        let mut remaining = info.byte_len();
        let mut scratch =
            vec![0; usize::try_from(remaining.min(1 << 20)).map_err(|e| bad(e.to_string()))?];
        let mut hash = Sha256::new();
        while remaining > 0 {
            let n = usize::try_from(remaining.min(scratch.len() as u64))
                .map_err(|e| bad(e.to_string()))?;
            let chunk = scratch
                .get_mut(..n)
                .ok_or_else(|| bad("invalid chunk size"))?;
            read.read_exact(chunk).map_err(|e| io(file.path(), e))?;
            hash.update(&*chunk);
            use_chunk(chunk);
            remaining = remaining.saturating_sub(n as u64);
        }
        let found: [u8; 32] = hash.finalize().into();
        if found != *want {
            return Err(bad(format!("checksum mismatch: {name}")));
        }
        Ok(())
    }

    pub(crate) fn verify_packed(&self, name: &str) -> Result<(), LoaderError> {
        self.packed_chunks(name, |_| {})
    }

    pub fn read_i8(&self, name: &str) -> Result<Vec<u8>, LoaderError> {
        let info = self.info(name).ok_or_else(|| bad("unknown codes tensor"))?;
        if info.dtype != Dtype::I8 {
            return Err(bad("expected I8 codes"));
        }
        let mut codes = Vec::new();
        codes
            .try_reserve_exact(usize::try_from(info.byte_len()).map_err(|e| bad(e.to_string()))?)
            .map_err(|e| bad(e.to_string()))?;
        self.packed_chunks(name, |chunk| codes.extend_from_slice(chunk))?;
        Ok(codes)
    }

    pub fn read_f32(&self, name: &str) -> Result<Vec<f32>, LoaderError> {
        if self.packed.is_none() {
            let file = self
                .owner
                .get(name)
                .and_then(|i| self.files.get(*i))
                .ok_or_else(|| bad("unknown FP32 tensor"))?;
            return file.read_f32(name);
        }

        let info = self.info(name).ok_or_else(|| bad("unknown FP32 tensor"))?;
        if info.dtype != Dtype::F32 {
            return Err(bad("expected F32 tensor"));
        }
        let mut values = Vec::new();
        values
            .try_reserve_exact(
                usize::try_from(info.byte_len() / 4).map_err(|e| bad(e.to_string()))?,
            )
            .map_err(|e| bad(e.to_string()))?;
        self.packed_chunks(name, |chunk| {
            values.extend(
                chunk
                    .chunks_exact(4)
                    .filter_map(|b| <[u8; 4]>::try_from(b).ok())
                    .map(f32::from_le_bytes),
            )
        })?;
        if values.iter().any(|v| !v.is_finite()) {
            return Err(bad(format!("nonfinite tensor: {name}")));
        }
        Ok(values)
    }
}

pub(crate) fn unique<'de, D, T>(de: D) -> Result<BTreeMap<String, T>, D::Error>
where
    D: serde::Deserializer<'de>,
    T: serde::Deserialize<'de>,
{
    struct Map<T>(std::marker::PhantomData<T>);
    impl<'de, T: serde::Deserialize<'de>> serde::de::Visitor<'de> for Map<T> {
        type Value = BTreeMap<String, T>;
        fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
            f.write_str("unique keys")
        }
        fn visit_map<A: serde::de::MapAccess<'de>>(
            self,
            mut access: A,
        ) -> Result<Self::Value, A::Error> {
            let mut map = BTreeMap::new();
            while let Some((key, value)) = access.next_entry::<String, T>()? {
                if map.insert(key.clone(), value).is_some() {
                    return Err(serde::de::Error::custom(format!("duplicate key: {key}")));
                }
            }
            Ok(map)
        }
    }
    de.deserialize_map(Map(std::marker::PhantomData))
}

pub(crate) fn parse<'de, T: serde::Deserialize<'de>>(
    bytes: &'de [u8],
) -> Result<BTreeMap<String, T>, serde_json::Error> {
    let mut de = serde_json::Deserializer::from_slice(bytes);
    let result = unique(&mut de)?;
    de.end()?;
    Ok(result)
}
