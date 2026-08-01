use std::{
    sync::{
        Arc, Condvar, Mutex,
        atomic::{AtomicBool, Ordering},
        mpsc,
    },
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
    stopping: Arc<AtomicBool>,
    thread: Option<std::thread::JoinHandle<()>>,
    results: mpsc::Receiver<CpuGraphResult>,
}

impl CpuGraphWorker {
    pub fn new() -> Result<Self, String> {
        let pending = Arc::new((Mutex::new(None::<Arc<CompiledGraph>>), Condvar::new()));
        let worker_pending = Arc::clone(&pending);
        let stopping = Arc::new(AtomicBool::new(false));
        let worker_stopping = Arc::clone(&stopping);
        let (sender, results) = mpsc::channel();
        let thread = std::thread::Builder::new()
            .name("zerofps-cpu-compositor".into())
            .spawn(move || {
                let mut executor = CpuGraphExecutor::default();
                loop {
                    let graph = {
                        let (lock, ready) = &*worker_pending;
                        let guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        let mut guard = ready
                            .wait_while(guard, |graph| {
                                graph.is_none() && !worker_stopping.load(Ordering::Acquire)
                            })
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        if worker_stopping.load(Ordering::Acquire) {
                            break;
                        }
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
        Ok(Self {
            pending,
            stopping,
            thread: Some(thread),
            results,
        })
    }

    pub fn submit_latest(&self, graph: Arc<CompiledGraph>) {
        if self.stopping.load(Ordering::Acquire) {
            return;
        }
        let (lock, ready) = &*self.pending;
        *lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(graph);
        ready.notify_one();
    }

    pub fn shutdown(&mut self) {
        self.stopping.store(true, Ordering::Release);
        *self
            .pending
            .0
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner()) = None;
        self.pending.1.notify_all();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
        while self.results.try_recv().is_ok() {}
    }

    pub fn try_result(&self) -> Option<CpuGraphResult> {
        self.results.try_recv().ok()
    }
}

impl Drop for CpuGraphWorker {
    fn drop(&mut self) {
        self.shutdown();
    }
}
