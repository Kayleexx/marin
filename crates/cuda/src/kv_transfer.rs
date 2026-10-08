//! Moving whole pages between the cache and host memory, for snapshots. A page's bytes are laid out
//! `[layer][keys, values][row]`, only the rows that hold positions.

use cudarc::driver::CudaSlice;

use super::{DeviceKv, shape};
use crate::pages::{PAGE_TOKENS, SLAB_PAGES};
use crate::{CudaError, Gpu};

impl DeviceKv {
    /// Bytes `export_page` returns and `import_page` takes for a page holding `rows` positions.
    pub fn page_payload_bytes(&self, rows: usize) -> usize {
        self.row_bytes()
            .saturating_mul(rows)
            .saturating_mul(self.layers)
            .saturating_mul(2)
    }

    fn check_page(&self, page: usize, rows: usize, positions: usize) -> Result<(), CudaError> {
        let end = page.saturating_mul(PAGE_TOKENS).saturating_add(rows);
        if rows == 0 || rows > PAGE_TOKENS || end > positions || page >= self.table.len() {
            return Err(shape("kv page rows", positions, end));
        }
        Ok(())
    }

    /// Start of one layer's keys or values for the page in slot `slot` of its slab, in bytes.
    fn block_start(&self, layer: usize, values: bool, slot: usize) -> Result<usize, CudaError> {
        let region = if values { self.region_bytes() } else { 0 };
        let page = slot
            .saturating_mul(PAGE_TOKENS)
            .saturating_mul(self.row_bytes());
        Ok((self.key_offset(layer)? as usize)
            .saturating_add(region)
            .saturating_add(page))
    }

    fn slot_of(&self, page: usize) -> Result<(usize, usize), CudaError> {
        let id = *self
            .table
            .ids()
            .get(page)
            .ok_or(shape("kv page", self.table.len(), page))?;
        Ok((id / SLAB_PAGES, id % SLAB_PAGES))
    }

    /// The first `rows` positions of logical page `page`, which must lie below the committed length.
    pub fn export_page(&self, gpu: &Gpu, page: usize, rows: usize) -> Result<Vec<u8>, CudaError> {
        self.check_page(page, rows, self.len)?;
        let (slab, slot) = self.slot_of(page)?;
        let storage = self
            .slabs
            .get(slab)
            .and_then(Option::as_ref)
            .ok_or(shape("kv slab", slab, slab))?;
        let block = self.row_bytes().saturating_mul(rows);
        let mut out = vec![0_u8; self.page_payload_bytes(rows)];
        let mut blocks = out.chunks_exact_mut(block.max(1));
        for layer in 0..self.layers {
            for values in [false, true] {
                let start = self.block_start(layer, values, slot)?;
                let target = blocks.next().ok_or(shape("kv blocks", 0, layer))?;
                gpu.stream
                    .memcpy_dtoh(&storage.slice(start..start.saturating_add(block)), target)?;
            }
        }
        gpu.synchronize()?;
        Ok(out)
    }

    /// Maps pages for `tokens` positions of an empty cache, before `import_page` fills them and `advance` commits them.
    pub fn reserve_empty(&mut self, gpu: &Gpu, tokens: usize) -> Result<(), CudaError> {
        if self.len != 0 || tokens > self.capacity {
            return Err(shape("kv import", self.capacity, tokens));
        }
        self.reserve(gpu, tokens)
    }

    /// Fills logical page `page` with `rows` positions; the cache must be empty and `tokens` reserved.
    pub fn import_page(
        &mut self,
        gpu: &Gpu,
        page: usize,
        rows: usize,
        bytes: &[u8],
    ) -> Result<(), CudaError> {
        if self.len != 0 || bytes.len() != self.page_payload_bytes(rows) {
            return Err(shape(
                "kv page bytes",
                self.page_payload_bytes(rows),
                bytes.len(),
            ));
        }
        let end = page.saturating_mul(PAGE_TOKENS).saturating_add(rows);
        self.check_page(page, rows, end)?;
        let (slab, slot) = self.slot_of(page)?;
        let block = self.row_bytes().saturating_mul(rows);
        let mut starts = Vec::new();
        for layer in 0..self.layers {
            for values in [false, true] {
                starts.push(self.block_start(layer, values, slot)?);
            }
        }
        let storage: &mut CudaSlice<u8> = self
            .slabs
            .get_mut(slab)
            .and_then(Option::as_mut)
            .ok_or(shape("kv slab", slab, slab))?;
        for (start, source) in starts.into_iter().zip(bytes.chunks_exact(block.max(1))) {
            gpu.stream.memcpy_htod(
                source,
                &mut storage.slice_mut(start..start.saturating_add(block)),
            )?;
        }
        Ok(())
    }
}
