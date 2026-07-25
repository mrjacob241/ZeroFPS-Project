use std::{
    sync::{Arc, Condvar, Mutex, mpsc},
    time::{Duration, Instant},
};

use zerofps_assets::TextureAsset;

use crate::compositor_graph::{CompiledGraph, CpuGraphExecutor, GraphExecutor};

pub struct CpuGraphResult {
    pub generation: u64,
    pub texture: Result<Arc<TextureAsset>, String>,
    pub worker_time: Duration,
}

/// Latest-only CPU graph worker. The UI and CPU backend exchange the same
/// immutable IR used by Vulkan; neither executor walks mutable editor state.
pub struct CpuGraphWorker {
    pending: Arc<(Mutex<Option<Arc<CompiledGraph>>>, Condvar)>,
    results: mpsc::Receiver<CpuGraphResult>,
}

impl CpuGraphWorker {
    pub fn new() -> Result<Self, String> {
        let pending = Arc::new((Mutex::new(None::<Arc<CompiledGraph>>), Condvar::new()));
        let worker_pending = Arc::clone(&pending);
        let (sender, results) = mpsc::channel();
        std::thread::Builder::new()
            .name("zerofps-cpu-compositor".into())
            .spawn(move || {
                let mut executor = CpuGraphExecutor::default();
                loop {
                    let graph = {
                        let (lock, ready) = &*worker_pending;
                        let guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        let mut guard = ready
                            .wait_while(guard, |graph| graph.is_none())
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        guard.take().expect("CPU graph request became available")
                    };
                    let started = Instant::now();
                    let texture = executor
                        .execute(&graph)
                        .map(|image| Arc::new(image.to_texture_asset_clamped()))
                        .map_err(|error| error.to_string());
                    if sender
                        .send(CpuGraphResult {
                            generation: graph.generation,
                            texture,
                            worker_time: started.elapsed(),
                        })
                        .is_err()
                    {
                        break;
                    }
                }
            })
            .map_err(|error| error.to_string())?;
        Ok(Self { pending, results })
    }

    pub fn submit_latest(&self, graph: Arc<CompiledGraph>) {
        let (lock, ready) = &*self.pending;
        *lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(graph);
        ready.notify_one();
    }

    pub fn try_result(&self) -> Option<CpuGraphResult> {
        self.results.try_recv().ok()
    }
}
