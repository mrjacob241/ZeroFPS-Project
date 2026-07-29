use std::{
    cell::RefCell,
    collections::HashMap,
    sync::{Arc, Condvar, Mutex, mpsc},
    time::{Duration, Instant},
};

use wgpu::util::DeviceExt;
use zerofps_assets::TextureAsset;

use crate::{
    compositor_graph::{
        AlgebraInstruction, CompiledGraph, GraphExecutor, GraphOperation, GraphSource,
    },
    vulkan_runtime::{GpuImage, shared_runtime},
};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct GraphParameters {
    operation: u32,
    variant: u32,
    connected: u32,
    point_count: u32,
    values: [f32; 4],
    points: [[f32; 4]; 32],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
    a_width: u32,
    a_height: u32,
    b_width: u32,
    b_height: u32,
    output_width: u32,
    output_height: u32,
    alpha_bits: u32,
    operation: u32,
}

pub struct VulkanCompositor {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    layout: wgpu::BindGroupLayout,
    pipeline: wgpu::ComputePipeline,
    input_cache: RefCell<HashMap<usize, (Arc<TextureAsset>, Arc<wgpu::Buffer>)>>,
    graph_layout: wgpu::BindGroupLayout,
    graph_pipeline: wgpu::ComputePipeline,
    graph_color_layout: wgpu::BindGroupLayout,
    graph_color_pipeline: wgpu::ComputePipeline,
    graph_input_cache: RefCell<HashMap<usize, (Arc<TextureAsset>, Arc<GpuImage>)>>,
    graph_dummy: Arc<GpuImage>,
    pub device_name: String,
}

struct GraphRequest {
    graph: Arc<CompiledGraph>,
}

pub struct GraphResult {
    pub generation: u64,
    pub texture: Result<Arc<GpuImage>, String>,
    pub worker_time: Duration,
}

/// Latest-wins graph worker. A graph is compiled on the UI/CPU side and is
/// executed dependency-first without reading editor state or pixels back.
pub struct VulkanGraphWorker {
    pending: Arc<(Mutex<Option<GraphRequest>>, Condvar)>,
    results: mpsc::Receiver<GraphResult>,
    pub device_name: String,
}

impl VulkanGraphWorker {
    pub fn new() -> Result<Self, String> {
        let mut compositor = VulkanCompositor::new()?;
        let device_name = compositor.device_name.clone();
        let pending = Arc::new((Mutex::new(None::<GraphRequest>), Condvar::new()));
        let worker_pending = Arc::clone(&pending);
        let (sender, results) = mpsc::channel();
        std::thread::Builder::new()
            .name("zerofps-vulkan-graph".into())
            .spawn(move || {
                loop {
                    let request = {
                        let (lock, ready) = &*worker_pending;
                        let guard = lock.lock().unwrap_or_else(|error| error.into_inner());
                        let mut guard = ready
                            .wait_while(guard, |request| request.is_none())
                            .unwrap_or_else(|error| error.into_inner());
                        guard.take().expect("graph request became available")
                    };
                    let generation = request.graph.generation;
                    let started = Instant::now();
                    let texture = compositor.execute(&request.graph);
                    if sender
                        .send(GraphResult {
                            generation,
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
            results,
            device_name,
        })
    }

    pub fn submit_latest(&self, graph: Arc<CompiledGraph>) {
        let (lock, ready) = &*self.pending;
        *lock.lock().unwrap_or_else(|error| error.into_inner()) = Some(GraphRequest { graph });
        ready.notify_one();
    }

    pub fn try_result(&self) -> Option<GraphResult> {
        self.results.try_recv().ok()
    }
}

struct MixRequest {
    generation: u64,
    node_id: usize,
    lod: u32,
    a: Arc<TextureAsset>,
    b: Arc<TextureAsset>,
    mode: usize,
    operation: usize,
    alpha: f32,
}

pub struct MixResult {
    pub generation: u64,
    pub node_id: usize,
    pub lod: u32,
    pub texture: Result<Arc<GpuImage>, String>,
    pub worker_time: Duration,
}

pub struct VulkanCompositorWorker {
    pending: Arc<(Mutex<Option<MixRequest>>, Condvar)>,
    results: mpsc::Receiver<MixResult>,
    next_generation: u64,
    pub device_name: String,
}

impl VulkanCompositorWorker {
    pub fn new() -> Result<Self, String> {
        let compositor = VulkanCompositor::new()?;
        let device_name = compositor.device_name.clone();
        let pending = Arc::new((Mutex::new(None::<MixRequest>), Condvar::new()));
        let worker_pending = Arc::clone(&pending);
        let (sender, results) = mpsc::channel();
        std::thread::Builder::new()
            .name("zerofps-vulkan-compositor".into())
            .spawn(move || {
                loop {
                    let request = {
                        let (lock, ready) = &*worker_pending;
                        let guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        let mut guard = ready
                            .wait_while(guard, |request| request.is_none())
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        guard.take().expect("Vulkan request became available")
                    };
                    let started = Instant::now();
                    let texture = compositor.combine(
                        &request.a,
                        &request.b,
                        request.mode,
                        request.operation,
                        request.alpha,
                    );
                    if sender
                        .send(MixResult {
                            generation: request.generation,
                            node_id: request.node_id,
                            lod: request.lod,
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
            results,
            next_generation: 0,
            device_name,
        })
    }

    pub fn submit_latest(
        &mut self,
        node_id: usize,
        lod: u32,
        a: Arc<TextureAsset>,
        b: Arc<TextureAsset>,
        mode: usize,
        operation: usize,
        alpha: f32,
    ) -> u64 {
        self.next_generation = self.next_generation.wrapping_add(1);
        let generation = self.next_generation;
        let (lock, ready) = &*self.pending;
        *lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(MixRequest {
            generation,
            node_id,
            lod,
            a,
            b,
            mode,
            operation,
            alpha,
        });
        ready.notify_one();
        generation
    }

    pub fn try_result(&self) -> Option<MixResult> {
        self.results.try_recv().ok()
    }
}

impl VulkanCompositor {
    pub fn new() -> Result<Self, String> {
        let runtime = shared_runtime()?;
        let device = Arc::clone(&runtime.device);
        let queue = Arc::clone(&runtime.queue);
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("ZeroFPS texture combine"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan_mix.wgsl").into()),
        });
        let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("Vulkan compositor bindings"),
            entries: &[
                storage_entry(0, true),
                storage_entry(1, true),
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::StorageTexture {
                        access: wgpu::StorageTextureAccess::WriteOnly,
                        format: wgpu::TextureFormat::Rgba8Unorm,
                        view_dimension: wgpu::TextureViewDimension::D2,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 3,
                    visibility: wgpu::ShaderStages::COMPUTE,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("Vulkan compositor pipeline layout"),
            bind_group_layouts: &[&layout],
            push_constant_ranges: &[],
        });
        let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: Some("Vulkan texture combine"),
            layout: Some(&pipeline_layout),
            module: &shader,
            entry_point: Some("main"),
            compilation_options: Default::default(),
            cache: None,
        });
        let (graph_layout, graph_pipeline) =
            create_graph_pipeline(&device, wgpu::TextureFormat::Rgba32Float);
        let (graph_color_layout, graph_color_pipeline) =
            create_graph_pipeline(&device, wgpu::TextureFormat::Rgba8Unorm);
        let graph_dummy = create_rgba32f_image(&device, 1, 1, "graph dummy");
        queue.write_texture(
            graph_dummy._texture.as_image_copy(),
            bytemuck::cast_slice(&[0.0f32, 0.0, 0.0, 1.0]),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        Ok(Self {
            device,
            queue,
            layout,
            pipeline,
            input_cache: RefCell::new(HashMap::new()),
            graph_layout,
            graph_pipeline,
            graph_color_layout,
            graph_color_pipeline,
            graph_input_cache: RefCell::new(HashMap::new()),
            graph_dummy,
            device_name: runtime.device_name.clone(),
        })
    }

    pub fn combine(
        &self,
        a: &Arc<TextureAsset>,
        b: &Arc<TextureAsset>,
        mode: usize,
        operation: usize,
        alpha: f32,
    ) -> Result<Arc<GpuImage>, String> {
        let width = a.width.max(b.width).max(1);
        let height = a.height.max(b.height).max(1);
        let a_buffer = self.storage_input("image A", a);
        let b_buffer = self.storage_input("image B", b);
        let output = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("Vulkan compositor output"),
            size: wgpu::Extent3d {
                width,
                height,
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8Unorm,
            usage: wgpu::TextureUsages::STORAGE_BINDING
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let output_view = Arc::new(output.create_view(&Default::default()));
        let parameters = Parameters {
            a_width: a.width,
            a_height: a.height,
            b_width: b.width,
            b_height: b.height,
            output_width: width,
            output_height: height,
            alpha_bits: alpha.to_bits(),
            operation: if mode == 1 { 8 } else { operation as u32 },
        };
        let parameters = self
            .device
            .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                label: Some("Vulkan compositor parameters"),
                contents: bytemuck::bytes_of(&parameters),
                usage: wgpu::BufferUsages::UNIFORM,
            });
        let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("Vulkan compositor resources"),
            layout: &self.layout,
            entries: &[
                binding(0, &a_buffer),
                binding(1, &b_buffer),
                wgpu::BindGroupEntry {
                    binding: 2,
                    resource: wgpu::BindingResource::TextureView(&output_view),
                },
                binding(3, &parameters),
            ],
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: Some("Vulkan texture combine pass"),
                timestamp_writes: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, &bind_group, &[]);
            pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
        }
        self.queue.submit([encoder.finish()]);
        Ok(Arc::new(GpuImage {
            _texture: output,
            view: output_view,
            encoded_srgb: true,
            width,
            height,
        }))
    }

    fn storage_input(&self, label: &str, texture: &Arc<TextureAsset>) -> Arc<wgpu::Buffer> {
        let key = Arc::as_ptr(texture) as usize;
        if let Some((_, buffer)) = self.input_cache.borrow().get(&key) {
            return Arc::clone(buffer);
        }
        let buffer = Arc::new(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some(label),
                    contents: &texture.pixels,
                    usage: wgpu::BufferUsages::STORAGE,
                }),
        );
        let mut cache = self.input_cache.borrow_mut();
        if cache.len() >= 64 {
            cache.clear();
        }
        cache.insert(key, (Arc::clone(texture), Arc::clone(&buffer)));
        buffer
    }
}

impl GraphExecutor for VulkanCompositor {
    type Image = Arc<GpuImage>;
    type Error = String;

    fn execute(&mut self, graph: &CompiledGraph) -> Result<Self::Image, Self::Error> {
        let mut images = HashMap::<usize, Arc<GpuImage>>::new();
        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor {
                label: Some("ZeroFPS compositor graph encoder"),
            });
        for node in &graph.nodes {
            if let GraphOperation::Source(source_index) = node.operation {
                let source = graph
                    .sources
                    .get(source_index)
                    .ok_or_else(|| format!("graph source {source_index} is missing"))?;
                let image = match source {
                    GraphSource::Texture(texture) => self.graph_texture(texture),
                    GraphSource::Constant(value) => self.constant_texture(*value),
                };
                images.insert(node.id, image);
                continue;
            }
            let inputs: [Arc<GpuImage>; 4] = std::array::from_fn(|index| {
                node.inputs[index]
                    .and_then(|id| images.get(&id).cloned())
                    .unwrap_or_else(|| Arc::clone(&self.graph_dummy))
            });
            let connected = node
                .inputs
                .iter()
                .enumerate()
                .fold(0u32, |mask, (index, input)| {
                    mask | (u32::from(input.is_some()) << index)
                });
            let (width, height) = node
                .inputs
                .iter()
                .flatten()
                .filter_map(|id| images.get(id))
                .fold((1u32, 1u32), |(width, height), image| {
                    (width.max(image.width), height.max(image.height))
                });
            let color_boundary = matches!(node.operation, GraphOperation::ClampColor);
            let output = if color_boundary {
                create_rgba8_image(&self.device, width, height, "graph color output")
            } else {
                create_rgba32f_image(&self.device, width, height, "graph float intermediate")
            };
            let parameters = graph_parameters(&node.operation, connected);
            let parameter_buffer =
                self.device
                    .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                        label: Some("graph operation parameters"),
                        contents: bytemuck::bytes_of(&parameters),
                        usage: wgpu::BufferUsages::UNIFORM,
                    });
            let bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                label: Some("graph operation resources"),
                layout: if color_boundary {
                    &self.graph_color_layout
                } else {
                    &self.graph_layout
                },
                entries: &[
                    texture_binding(0, &inputs[0].view),
                    texture_binding(1, &inputs[1].view),
                    texture_binding(2, &inputs[2].view),
                    texture_binding(3, &inputs[3].view),
                    texture_binding(4, &output.view),
                    binding(5, &parameter_buffer),
                ],
            });
            {
                let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                    label: Some("graph operation"),
                    timestamp_writes: None,
                });
                pass.set_pipeline(if color_boundary {
                    &self.graph_color_pipeline
                } else {
                    &self.graph_pipeline
                });
                pass.set_bind_group(0, &bind_group, &[]);
                pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
            }
            images.insert(node.id, output);
        }
        let output = images
            .get(&graph.output)
            .cloned()
            .ok_or_else(|| format!("graph output {} was not produced", graph.output))?;
        // Every operation is encoded first; there is exactly one graph submit
        // and no intermediate CPU readback.
        self.queue.submit([encoder.finish()]);
        Ok(output)
    }
}

impl VulkanCompositor {
    fn graph_texture(&self, texture: &Arc<TextureAsset>) -> Arc<GpuImage> {
        let key = Arc::as_ptr(texture) as usize;
        if let Some((_, image)) = self.graph_input_cache.borrow().get(&key) {
            return Arc::clone(image);
        }
        let image = create_rgba32f_image(
            &self.device,
            texture.width.max(1),
            texture.height.max(1),
            "graph source",
        );
        let pixels: Vec<f32> = texture
            .pixels
            .iter()
            .map(|value| *value as f32 / 255.0)
            .collect();
        self.queue.write_texture(
            image._texture.as_image_copy(),
            bytemuck::cast_slice(&pixels),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(texture.width.max(1) * 16),
                rows_per_image: Some(texture.height.max(1)),
            },
            wgpu::Extent3d {
                width: texture.width.max(1),
                height: texture.height.max(1),
                depth_or_array_layers: 1,
            },
        );
        let mut cache = self.graph_input_cache.borrow_mut();
        if cache.len() >= 64 {
            cache.clear();
        }
        cache.insert(key, (Arc::clone(texture), Arc::clone(&image)));
        image
    }

    fn constant_texture(&self, value: [f32; 4]) -> Arc<GpuImage> {
        let image = create_rgba32f_image(&self.device, 1, 1, "graph constant");
        self.queue.write_texture(
            image._texture.as_image_copy(),
            bytemuck::cast_slice(&value),
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(16),
                rows_per_image: Some(1),
            },
            wgpu::Extent3d {
                width: 1,
                height: 1,
                depth_or_array_layers: 1,
            },
        );
        image
    }
}

fn create_graph_pipeline(
    device: &wgpu::Device,
    output_format: wgpu::TextureFormat,
) -> (wgpu::BindGroupLayout, wgpu::ComputePipeline) {
    let shader_source = match output_format {
        wgpu::TextureFormat::Rgba8Unorm => {
            include_str!("vulkan_graph.wgsl").replace("rgba32float", "rgba8unorm")
        }
        _ => include_str!("vulkan_graph.wgsl").to_owned(),
    };
    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("ZeroFPS compositor graph"),
        source: wgpu::ShaderSource::Wgsl(shader_source.into()),
    });
    let layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("Vulkan graph bindings"),
        entries: &[
            sampled_texture_entry(0),
            sampled_texture_entry(1),
            sampled_texture_entry(2),
            sampled_texture_entry(3),
            wgpu::BindGroupLayoutEntry {
                binding: 4,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::StorageTexture {
                    access: wgpu::StorageTextureAccess::WriteOnly,
                    format: output_format,
                    view_dimension: wgpu::TextureViewDimension::D2,
                },
                count: None,
            },
            wgpu::BindGroupLayoutEntry {
                binding: 5,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("Vulkan graph pipeline layout"),
        bind_group_layouts: &[&layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Vulkan graph pipeline"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });
    (layout, pipeline)
}

fn graph_parameters(operation: &GraphOperation, connected: u32) -> GraphParameters {
    let mut result = GraphParameters {
        operation: 0,
        variant: 0,
        connected,
        point_count: 0,
        values: [0.0; 4],
        points: [[0.0; 4]; 32],
    };
    match operation {
        GraphOperation::Source(_) => {}
        GraphOperation::Remap { points, bezier } => {
            result.operation = 1;
            result.variant = u32::from(*bezier);
            result.point_count = points.len().min(8) as u32;
            for (target, source) in result.points.iter_mut().zip(points.iter().take(8)) {
                *target = [source[0], source[1], 0.0, 0.0];
            }
        }
        GraphOperation::Math {
            operation,
            constant,
        } => {
            result.operation = 2;
            result.variant = *operation as u32;
            result.values[0] = *constant;
        }
        GraphOperation::Algebra { program } => {
            result.operation = 12;
            result.point_count = program.len().min(32) as u32;
            for (target, instruction) in result.points.iter_mut().zip(program) {
                *target = match instruction {
                    AlgebraInstruction::Variable(index) => [*index as f32, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Constant(value) => [3.0, *value, 0.0, 0.0],
                    AlgebraInstruction::Add => [4.0, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Subtract => [5.0, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Multiply => [6.0, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Divide => [7.0, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Power => [8.0, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Negate => [9.0, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Sin => [10.0, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Cos => [11.0, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Abs => [12.0, 0.0, 0.0, 0.0],
                    AlgebraInstruction::Sqrt => [13.0, 0.0, 0.0, 0.0],
                };
            }
        }
        GraphOperation::SharpThreshold { threshold } => {
            result.operation = 3;
            result.values[0] = *threshold;
        }
        GraphOperation::SmoothThreshold { threshold, width } => {
            result.operation = 4;
            result.values = [*threshold, *width, 0.0, 0.0];
        }
        GraphOperation::ImageFilter { filter, radius } => {
            result.operation = 5;
            result.variant = *filter as u32;
            result.values[0] = *radius;
        }
        GraphOperation::Combine {
            mode,
            operation,
            alpha,
        } => {
            result.operation = 6;
            result.variant = if *mode == 1 { 8 } else { *operation as u32 };
            result.values[0] = *alpha;
        }
        GraphOperation::ColorSpace { from, to } => {
            result.operation = 7;
            result.variant = ((*from as u32) << 16) | *to as u32;
        }
        GraphOperation::ExtractChannel { channel } => {
            result.operation = 8;
            result.variant = *channel as u32;
        }
        GraphOperation::Grayscale { mode } => {
            result.operation = 9;
            result.variant = *mode as u32;
        }
        GraphOperation::JoinChannels => result.operation = 10,
        GraphOperation::ClampColor => result.operation = 11,
    }
    result
}

fn create_rgba8_image(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> Arc<GpuImage> {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8Unorm,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    let view = Arc::new(texture.create_view(&Default::default()));
    Arc::new(GpuImage {
        _texture: texture,
        view,
        encoded_srgb: true,
        width,
        height,
    })
}

fn create_rgba32f_image(
    device: &wgpu::Device,
    width: u32,
    height: u32,
    label: &str,
) -> Arc<GpuImage> {
    let texture = Arc::new(device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        },
        mip_level_count: 1,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba32Float,
        usage: wgpu::TextureUsages::STORAGE_BINDING
            | wgpu::TextureUsages::TEXTURE_BINDING
            | wgpu::TextureUsages::COPY_DST
            | wgpu::TextureUsages::COPY_SRC,
        view_formats: &[],
    }));
    let view = Arc::new(texture.create_view(&Default::default()));
    Arc::new(GpuImage {
        _texture: texture,
        view,
        encoded_srgb: false,
        width,
        height,
    })
}

fn sampled_texture_entry(binding: u32) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Texture {
            sample_type: wgpu::TextureSampleType::Float { filterable: false },
            view_dimension: wgpu::TextureViewDimension::D2,
            multisampled: false,
        },
        count: None,
    }
}

fn texture_binding(binding: u32, view: &wgpu::TextureView) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: wgpu::BindingResource::TextureView(view),
    }
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

#[cfg(test)]
mod graph_tests {
    use super::*;
    use crate::compositor_graph::{CpuGraphExecutor, GraphNode};

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn graph_worker_matches_cpu_reference() {
        let texture = Arc::new(TextureAsset {
            name: "source".into(),
            width: 2,
            height: 1,
            pixels: vec![64, 128, 255, 255, 200, 20, 100, 128],
        });
        let graph = Arc::new(CompiledGraph {
            generation: 1,
            sources: vec![
                GraphSource::Texture(texture),
                GraphSource::Constant([0.25, 0.25, 0.25, 0.25]),
            ],
            nodes: vec![
                GraphNode {
                    id: 0,
                    operation: GraphOperation::Source(0),
                    inputs: [None; 4],
                },
                GraphNode {
                    id: 1,
                    operation: GraphOperation::Math {
                        operation: 2,
                        constant: 0.5,
                    },
                    inputs: [Some(0), None, None, None],
                },
                GraphNode {
                    id: 2,
                    operation: GraphOperation::Source(1),
                    inputs: [None; 4],
                },
                GraphNode {
                    id: 3,
                    operation: GraphOperation::Combine {
                        mode: 1,
                        operation: 0,
                        alpha: 0.75,
                    },
                    inputs: [Some(1), Some(2), None, None],
                },
                GraphNode {
                    id: 4,
                    operation: GraphOperation::Math {
                        operation: 0,
                        constant: 1.0,
                    },
                    inputs: [Some(3), None, None, None],
                },
                GraphNode {
                    id: 5,
                    operation: GraphOperation::Math {
                        operation: 1,
                        constant: 1.0,
                    },
                    inputs: [Some(4), None, None, None],
                },
                GraphNode {
                    id: 6,
                    operation: GraphOperation::ColorSpace { from: 0, to: 1 },
                    inputs: [Some(5), None, None, None],
                },
                GraphNode {
                    id: 7,
                    operation: GraphOperation::Algebra {
                        program: crate::compositor_graph::compile_algebra_expression("x + y * 2")
                            .unwrap(),
                    },
                    inputs: [Some(6), Some(2), None, None],
                },
                GraphNode {
                    id: 8,
                    operation: GraphOperation::ClampColor,
                    inputs: [Some(7), None, None, None],
                },
            ],
            output: 8,
        });
        let expected = CpuGraphExecutor
            .execute(&graph)
            .expect("CPU graph reference")
            .to_texture_asset_clamped();
        let worker = VulkanGraphWorker::new().expect("Vulkan graph worker");
        worker.submit_latest(Arc::clone(&graph));
        let deadline = Instant::now() + Duration::from_secs(3);
        let result = loop {
            if let Some(result) = worker.try_result() {
                break result;
            }
            assert!(Instant::now() < deadline, "Vulkan graph worker timed out");
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(result.generation, graph.generation);
        let actual = result
            .texture
            .expect("GPU graph")
            .readback_rgba8()
            .expect("explicit parity readback");
        assert_eq!(actual.len(), expected.pixels.len());
        for (gpu, cpu) in actual.iter().zip(&expected.pixels) {
            assert!(gpu.abs_diff(*cpu) <= 1, "GPU {gpu}, CPU {cpu}");
        }
    }
}
