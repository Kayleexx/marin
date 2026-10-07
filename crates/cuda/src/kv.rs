use cudarc::driver::{CudaSlice, DevicePtr, LaunchConfig, PushKernelArg};

use crate::attention::{MAX_HEAD_DIM, scratch_lens, split_scratch};
use crate::ops::to_u32;
use crate::pages::{PAGE_TOKENS, PagePool, PageTable, SLAB_PAGES, pages_for};
use crate::{Buffer, CudaError, Gpu, KvFormat};

/// Pages of `PAGE_TOKENS` positions, allocated in slabs of `SLAB_PAGES` pages on first use. A slab is laid out
/// `[layer][keys, values][page][row]`, so a layer's keys stay contiguous across the slab's pages. A row is the
/// `[kv_head][head_dim]` values of one position in the cache's `KvFormat`.
pub struct DeviceKv {
    slabs: Vec<Option<CudaSlice<u8>>>,
    pub(crate) format: KvFormat,
    pool: PagePool,
    table: PageTable,
    /// Only the first `table.len()` entries are valid.
    pub(crate) page_ptrs: CudaSlice<u64>,
    pub(crate) layers: usize,
    pub(crate) heads: usize,
    pub(crate) kv_heads: usize,
    pub(crate) head_dim: usize,
    /// Partial results of the split decode attention, sized once for the longest context so a pass never allocates.
    pub(crate) partial_acc: Buffer,
    pub(crate) partial_ml: Buffer,
    pub(crate) capacity: usize,
    pub(crate) len: usize,
}

fn shape(what: &'static str, expected: usize, actual: usize) -> CudaError {
    CudaError::Shape {
        what,
        expected,
        actual,
    }
}

impl DeviceKv {
    pub fn new(
        gpu: &Gpu,
        layers: usize,
        (heads, kv_heads, head_dim): (usize, usize, usize),
        capacity: usize,
        format: KvFormat,
    ) -> Result<Self, CudaError> {
        slab_bytes(layers, (kv_heads, head_dim), format)
            .ok_or(CudaError::Overflow { what: "kv cache" })?;
        let limit = pages_for(capacity);
        let (partial_acc, partial_ml) = split_scratch(gpu, (heads, kv_heads, head_dim), capacity)?;
        Ok(Self {
            slabs: Vec::new(),
            format,
            pool: PagePool::new(limit),
            table: PageTable::default(),
            page_ptrs: gpu.stream.alloc_zeros::<u64>(limit.max(1))?,
            layers,
            heads,
            kv_heads,
            head_dim,
            partial_acc,
            partial_ml,
            capacity,
            len: 0,
        })
    }

    fn stride(&self) -> usize {
        self.kv_heads.saturating_mul(self.head_dim)
    }

    pub(crate) fn row_bytes(&self) -> usize {
        self.format
            .row_bytes(self.kv_heads, self.head_dim)
            .unwrap_or(usize::MAX)
    }

    /// Bytes of one layer's keys (or values) in a slab.
    fn region_bytes(&self) -> usize {
        self.row_bytes()
            .saturating_mul(PAGE_TOKENS)
            .saturating_mul(SLAB_PAGES)
    }

    pub fn format(&self) -> KvFormat {
        self.format
    }

    pub fn len(&self) -> usize {
        self.len
    }

    pub fn is_empty(&self) -> bool {
        self.len == 0
    }

    pub fn capacity(&self) -> usize {
        self.capacity
    }

    /// Pages that have storage. Pages given back stay allocated for reuse, so this is the high-water mark.
    pub fn pages_held(&self) -> usize {
        self.slabs
            .iter()
            .enumerate()
            .filter(|(_, slab)| slab.is_some())
            .map(|(index, _)| {
                let first = index.saturating_mul(SLAB_PAGES);
                SLAB_PAGES.min(self.pool.issued().saturating_sub(first))
            })
            .sum()
    }

    pub fn pages_in_use(&self) -> usize {
        self.table.len()
    }

    pub fn page_ids(&self) -> &[usize] {
        self.table.ids()
    }

    /// Keys and values of every layer for one page.
    pub fn page_bytes(&self) -> usize {
        self.row_bytes()
            .saturating_mul(PAGE_TOKENS)
            .saturating_mul(self.layers)
            .saturating_mul(2)
    }

    pub fn slab_bytes(&self) -> usize {
        self.region_bytes()
            .saturating_mul(self.layers)
            .saturating_mul(2)
    }

    /// GPU memory allocated for pages, a whole slab at a time. Excludes the split-attention scratch.
    pub fn bytes_held(&self) -> usize {
        self.slabs
            .iter()
            .flatten()
            .count()
            .saturating_mul(self.slab_bytes())
    }

    /// GPU memory a cache for `capacity` positions needs before its first token: the first slab, the page table and the
    /// split-attention scratch. The rest of the window is allocated as the sequence grows.
    pub fn startup_bytes(
        layers: usize,
        (heads, kv_heads, head_dim): (usize, usize, usize),
        capacity: usize,
        format: KvFormat,
    ) -> Option<usize> {
        let slab = slab_bytes(layers, (kv_heads, head_dim), format)?;
        let table = pages_for(capacity).max(1).checked_mul(size_of::<u64>())?;
        let (acc, ml) = scratch_lens((heads, kv_heads, head_dim), capacity)?;
        let scratch = acc.checked_add(ml)?.checked_mul(size_of::<f32>())?;
        slab.checked_add(table)?.checked_add(scratch)
    }

    pub fn clear(&mut self) {
        self.len = 0;
        self.table.shrink_to(&mut self.pool, 0);
    }

    /// Forgets every position from `len` on, so the next write lands there, and gives back emptied pages.
    pub fn truncate(&mut self, len: usize) -> Result<(), CudaError> {
        if len > self.len {
            return Err(shape("kv rollback", self.len, len));
        }
        self.len = len;
        self.table.shrink_to(&mut self.pool, pages_for(len));
        Ok(())
    }

    pub fn advance(&mut self, tokens: usize) -> Result<(), CudaError> {
        let new_len = self
            .len
            .checked_add(tokens)
            .filter(|n| *n <= self.capacity)
            .ok_or(shape(
                "kv positions",
                self.capacity,
                self.len.saturating_add(tokens),
            ))?;
        self.len = new_len;
        Ok(())
    }

    pub(crate) fn key_offset(&self, layer: usize) -> Result<u32, CudaError> {
        if layer >= self.layers {
            return Err(shape("kv layer", self.layers, layer));
        }
        let offset = self.region_bytes().saturating_mul(2).saturating_mul(layer);
        u32::try_from(offset).map_err(|_| CudaError::Overflow {
            what: "kv layer offset",
        })
    }

    pub(crate) fn mapped_pages(&self) -> usize {
        self.table.len()
    }

    /// Idempotent, so every layer of a pass can ask for the same positions.
    fn reserve(&mut self, gpu: &Gpu, tokens: usize) -> Result<(), CudaError> {
        let first_new = self
            .table
            .grow_to(&mut self.pool, pages_for(tokens))
            .map_err(|e| shape("kv pages", e.available, e.needed))?;
        let mapped = self.map_new_pages(gpu, first_new);
        if mapped.is_err() {
            self.table.shrink_to(&mut self.pool, first_new);
        }
        mapped
    }

    fn map_new_pages(&mut self, gpu: &Gpu, first_new: usize) -> Result<(), CudaError> {
        let new_ids: Vec<usize> = self.table.ids().get(first_new..).unwrap_or(&[]).to_vec();
        let mut addresses = Vec::with_capacity(new_ids.len());
        for id in new_ids {
            addresses.push(self.address(gpu, id)?);
        }
        if addresses.is_empty() {
            return Ok(());
        }
        let end = first_new.saturating_add(addresses.len());
        gpu.stream
            .memcpy_htod(&addresses, &mut self.page_ptrs.slice_mut(first_new..end))?;
        Ok(())
    }

    /// The device address of page `id` inside its slab, allocating the slab the first time.
    fn address(&mut self, gpu: &Gpu, id: usize) -> Result<u64, CudaError> {
        let (slab, slot) = (id / SLAB_PAGES, id % SLAB_PAGES);
        if self.slabs.len() <= slab {
            self.slabs.resize_with(slab.saturating_add(1), || None);
        }
        let bytes = self.slab_bytes();
        let page_bytes = self.row_bytes().saturating_mul(PAGE_TOKENS);
        let (held, positions) = (self.bytes_held(), self.len);
        let entry = self
            .slabs
            .get_mut(slab)
            .ok_or(shape("kv slab", slab, slab))?;
        if entry.is_none() {
            gpu.counters.allocation();
            let storage =
                gpu.stream
                    .alloc_zeros::<u8>(bytes)
                    .map_err(|source| CudaError::KvMemory {
                        held,
                        needed: bytes,
                        positions,
                        source,
                    })?;
            *entry = Some(storage);
        }
        let storage = entry.as_ref().ok_or(shape("kv slab", slab, slab))?;
        let (base, _guard) = storage.device_ptr(&gpu.stream);
        Ok(base.saturating_add(u64::try_from(slot.saturating_mul(page_bytes)).unwrap_or(0)))
    }

    /// Stores keys and values for the chunk starting at the committed length.
    pub fn write(
        &mut self,
        gpu: &Gpu,
        layer: usize,
        keys: &Buffer,
        values: &Buffer,
    ) -> Result<(), CudaError> {
        let stride = self.stride();
        let tokens = keys.len.checked_div(stride).unwrap_or(0);
        if keys.len != tokens.saturating_mul(stride) || values.len != keys.len {
            return Err(shape("kv chunk", tokens.saturating_mul(stride), keys.len));
        }
        let end_pos = self.len.saturating_add(tokens);
        if end_pos > self.capacity {
            return Err(shape("kv positions", self.capacity, end_pos));
        }
        if self.head_dim > MAX_HEAD_DIM {
            return Err(CudaError::Unsupported {
                what: "head layout",
                max: MAX_HEAD_DIM,
                actual: self.head_dim,
            });
        }
        let key_offset = self.key_offset(layer)?;
        self.reserve(gpu, end_pos)?;
        if tokens == 0 {
            return Ok(());
        }
        let function = gpu.kernels.kv_store.get(self.format.index()).ok_or(shape(
            "kv format",
            KvFormat::ALL.len(),
            self.format.index(),
        ))?;
        let config = LaunchConfig {
            grid_dim: (
                to_u32(tokens, "kv tokens")?,
                to_u32(self.kv_heads, "kv heads")?,
                1,
            ),
            block_dim: (
                to_u32(self.head_dim.next_multiple_of(32), "block size")?,
                1,
                1,
            ),
            shared_mem_bytes: to_u32(32_usize.saturating_mul(size_of::<f32>()), "shared memory")?,
        };
        let (row_bytes, start) = (
            to_u32(self.row_bytes(), "kv row")?,
            to_u32(self.len, "kv start")?,
        );
        let (kv_heads, head_dim) = (
            to_u32(self.kv_heads, "kv heads")?,
            to_u32(self.head_dim, "head size")?,
        );
        gpu.launch(function, config, |a| {
            a.arg(&keys.data).arg(&values.data).arg(&self.page_ptrs);
            a.arg(&key_offset).arg(&row_bytes).arg(&start);
            a.arg(&kv_heads).arg(&head_dim);
        })
    }
}

/// Bytes of a slab: every layer's keys and values for `SLAB_PAGES` pages.
fn slab_bytes(
    layers: usize,
    (kv_heads, head_dim): (usize, usize),
    format: KvFormat,
) -> Option<usize> {
    format
        .row_bytes(kv_heads, head_dim)?
        .checked_mul(PAGE_TOKENS)?
        .checked_mul(SLAB_PAGES)?
        .checked_mul(layers.checked_mul(2)?)
}
