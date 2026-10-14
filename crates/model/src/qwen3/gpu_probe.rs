use cuda::{Gpu, Mark};
use metrics::{Stage, StageTimes};

use crate::ModelError;

/// Times GPU stages on the GPU's own clock: a mark before and after each stage, read once the pass is queued,
/// so profiling does not stall the stream between stages. Off, it runs the stage and nothing else.
pub(crate) struct GpuProbe<'a> {
    gpu: &'a Gpu,
    times: Option<&'a mut StageTimes>,
    layer: Option<usize>,
    marks: Vec<(Option<usize>, Stage, Mark, Mark)>,
}

impl<'a> GpuProbe<'a> {
    pub(crate) fn new(gpu: &'a Gpu, times: Option<&'a mut StageTimes>) -> Self {
        Self {
            gpu,
            times,
            layer: None,
            marks: Vec::new(),
        }
    }

    /// Tags the stages timed from now on with `layer`.
    pub(crate) fn layer(&mut self, layer: Option<usize>) {
        self.layer = layer;
    }

    pub(crate) fn stage<T>(
        &mut self,
        stage: Stage,
        work: impl FnOnce() -> Result<T, ModelError>,
    ) -> Result<T, ModelError> {
        if self.times.is_none() {
            return work();
        }
        let before = self.gpu.mark()?;
        let out = work()?;
        let after = self.gpu.mark()?;
        self.marks.push((self.layer, stage, before, after));
        Ok(out)
    }

    pub(crate) fn finish(self) -> Result<(), ModelError> {
        if let Some(times) = self.times {
            for (layer, stage, before, after) in &self.marks {
                times.set_layer(*layer);
                times.add(*stage, after.since(before)?);
            }
            times.set_layer(None);
        }
        Ok(())
    }
}
