use std::sync::{Arc, OnceLock};

pub struct VulkanRuntime {
    pub device: Arc<wgpu::Device>,
    pub queue: Arc<wgpu::Queue>,
    pub device_name: String,
}

static RUNTIME: OnceLock<Result<Arc<VulkanRuntime>, String>> = OnceLock::new();

pub fn install_runtime(
    device: wgpu::Device,
    queue: wgpu::Queue,
    device_name: String,
) -> Result<(), String> {
    RUNTIME
        .set(Ok(Arc::new(VulkanRuntime {
            device: Arc::new(device),
            queue: Arc::new(queue),
            device_name,
        })))
        .map_err(|_| "shared Vulkan runtime was already initialized".to_owned())
}

pub fn shared_runtime() -> Result<Arc<VulkanRuntime>, String> {
    RUNTIME
        .get_or_init(|| pollster::block_on(create_runtime()))
        .clone()
}

async fn create_runtime() -> Result<Arc<VulkanRuntime>, String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or_else(|| "no Vulkan adapter found".to_owned())?;
    let info = adapter.get_info();
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ZeroFPS shared Vulkan device"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|error| error.to_string())?;
    Ok(Arc::new(VulkanRuntime {
        device: Arc::new(device),
        queue: Arc::new(queue),
        device_name: info.name,
    }))
}

pub struct GpuImage {
    pub _texture: Arc<wgpu::Texture>,
    /// Linear/unorm view used by compute passes and ordinary render targets.
    pub view: Arc<wgpu::TextureView>,
    /// The stored bytes encode sRGB color but the compute-compatible view is
    /// unorm, so a consumer must decode them before lighting.
    pub encoded_srgb: bool,
    pub width: u32,
    pub height: u32,
}

impl GpuImage {
    /// Explicit compatibility path for CPU-only graph nodes and export.
    /// Normal Vulkan viewport/compositor flow must pass `GpuImage` handles.
    pub fn readback_rgba8(&self) -> Result<Vec<u8>, String> {
        let runtime = shared_runtime()?;
        let row_bytes = self.width * 4;
        let padded_row = row_bytes.div_ceil(wgpu::COPY_BYTES_PER_ROW_ALIGNMENT)
            * wgpu::COPY_BYTES_PER_ROW_ALIGNMENT;
        let buffer = runtime.device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("explicit GPU image readback"),
            size: padded_row as u64 * self.height as u64,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
            mapped_at_creation: false,
        });
        let mut encoder = runtime.device.create_command_encoder(&Default::default());
        encoder.copy_texture_to_buffer(
            self._texture.as_image_copy(),
            wgpu::TexelCopyBufferInfo {
                buffer: &buffer,
                layout: wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(padded_row),
                    rows_per_image: Some(self.height),
                },
            },
            wgpu::Extent3d {
                width: self.width,
                height: self.height,
                depth_or_array_layers: 1,
            },
        );
        runtime.queue.submit([encoder.finish()]);
        let slice = buffer.slice(..);
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        slice.map_async(wgpu::MapMode::Read, move |result| {
            let _ = sender.send(result);
        });
        runtime.device.poll(wgpu::Maintain::Wait);
        receiver
            .recv()
            .map_err(|_| "GPU readback callback disconnected".to_owned())?
            .map_err(|error| error.to_string())?;
        let mapped = slice.get_mapped_range();
        let mut pixels = Vec::with_capacity((row_bytes * self.height) as usize);
        for row in mapped
            .chunks_exact(padded_row as usize)
            .take(self.height as usize)
        {
            pixels.extend_from_slice(&row[..row_bytes as usize]);
        }
        drop(mapped);
        buffer.unmap();
        Ok(pixels)
    }
}
