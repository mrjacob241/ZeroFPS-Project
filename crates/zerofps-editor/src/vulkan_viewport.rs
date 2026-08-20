use std::{
    collections::{HashMap, HashSet, VecDeque},
    sync::Arc,
    time::{Duration, Instant},
};

use egui::Vec2;
use wgpu::util::DeviceExt;
use zerofps_assets::TextureAsset;
use zerofps_core::{Transform, Vec3};

use crate::vulkan_runtime::GpuImage;
use crate::vulkan_runtime::shared_runtime;
use crate::{DirectionalShadowMap, MAX_VIEWPORT_LIGHTS, PointShadowAtlas, ViewportLight};

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
pub struct GpuVertex {
    pub position: [f32; 4],
    pub normal: [f32; 4],
    pub uv_color_rg: [f32; 4],
    pub color_ba_base_rg: [f32; 4],
    pub base_ba_material: [f32; 4],
    pub object_translation: [f32; 4],
    pub object_rotation: [f32; 4],
    pub object_scale: [f32; 4],
}

#[derive(Clone)]
pub struct GpuBatch {
    pub cache_key: u64,
    pub content_revision: u64,
    pub object_id: zerofps_core::NodeId,
    pub object_transform: Transform,
    pub local_bounds_min: Vec3,
    pub local_bounds_max: Vec3,
    pub casts_shadows: bool,
    pub texture_cache_key: u64,
    pub texture: Option<Arc<TextureAsset>>,
    pub gpu_texture: Option<Arc<GpuImage>>,
    pub transparent: bool,
    pub vertices: Vec<GpuVertex>,
}

#[derive(Clone, Copy, Default)]
pub struct PointShadowCpuTimings {
    pub setup: Duration,
    pub uniform_write: Duration,
    pub geometry_lookup: Duration,
    pub encoder_create: Duration,
    pub pass_and_finish: Duration,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    yaw: f32,
    pitch: f32,
    roll: f32,
    zoom: f32,
    grid_spacing: f32,
    camera_padding: [f32; 3],
    target: [f32; 4],
    viewport: [f32; 2],
    projection: u32,
    padding: u32,
    global_light_enabled: u32,
    point_light_count: u32,
    directional_shadow_enabled: u32,
    lighting_padding: u32,
    point_positions: [[f32; 4]; MAX_VIEWPORT_LIGHTS],
    point_colors: [[f32; 4]; MAX_VIEWPORT_LIGHTS],
    point_shadow_regions: [[f32; 4]; MAX_VIEWPORT_LIGHTS],
    shadow_origin: [f32; 4],
    shadow_right: [f32; 4],
    shadow_up: [f32; 4],
    shadow_forward: [f32; 4],
    shadow_parameters: [f32; 4],
    point_shadow_atlas_size: [f32; 4],
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct ShadowUniform {
    origin: [f32; 4],
    right: [f32; 4],
    up: [f32; 4],
    forward: [f32; 4],
    parameters: [f32; 4],
    atlas: [f32; 4],
}

const SHADOW_UNIFORM_STRIDE: u64 = 256;
const MAX_SHADOW_VIEWS: u64 = 1 + (crate::MAX_VIEWPORT_LIGHTS as u64 * 6);

fn camera_uniform_layout_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn shadow_uniform_layout_entry() -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding: 0,
        visibility: wgpu::ShaderStages::VERTEX,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Uniform,
            has_dynamic_offset: true,
            min_binding_size: None,
        },
        count: None,
    }
}

pub struct VulkanViewport {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    camera_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    transparent_pipeline: wgpu::RenderPipeline,
    shadow_uniform_buffer: wgpu::Buffer,
    point_shadow_view_buffers: Vec<wgpu::Buffer>,
    shadow_uniform_groups: Vec<wgpu::BindGroup>,
    shadow_frame_slot: usize,
    directional_shadow_pipeline: wgpu::RenderPipeline,
    point_shadow_pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    texture_cache: HashMap<usize, (Arc<TextureAsset>, Arc<wgpu::TextureView>)>,
    white: Arc<wgpu::TextureView>,
    targets: Option<CachedTargets>,
    camera_buffer: wgpu::Buffer,
    shadow_cache: Option<(usize, u32, u32, wgpu::Texture, Arc<wgpu::TextureView>)>,
    point_shadow_cache: Option<(usize, u32, u32, wgpu::Texture, Arc<wgpu::TextureView>)>,
    retired_shadow_targets: VecDeque<(u32, u32, bool, wgpu::Texture, Arc<wgpu::TextureView>, u64)>,
    shadow_depth_targets:
        HashMap<(u32, u32), (usize, Vec<(wgpu::Texture, Arc<wgpu::TextureView>)>)>,
    gpu_shadow_key: Option<u64>,
    gpu_point_shadow_key: Option<u64>,
    gpu_directional_metadata: Option<DirectionalShadowMap>,
    gpu_point_metadata: Option<PointShadowAtlas>,
    camera_group_cache: Option<(usize, usize, Arc<wgpu::BindGroup>)>,
    vertex_cache: HashMap<u64, (usize, u64, Arc<wgpu::Buffer>)>,
    shadow_vertex_buffer: Option<(Vec<(u64, u64, usize, usize)>, Arc<wgpu::Buffer>, u32)>,
    shadow_draw_indirect: wgpu::Buffer,
    timestamp_profiler: Option<GpuTimestampProfiler>,
    timestamp_poll_frame: u64,
    pending_shadow_commands: Vec<wgpu::CommandBuffer>,
    point_shadow_staging: wgpu::util::StagingBelt,
    timestamp_directional_written: bool,
    timestamp_point_written: bool,
    bind_group_cache: HashMap<usize, Arc<wgpu::BindGroup>>,
    last_shadow_encode_time: Duration,
    last_directional_shadow_time: Duration,
    last_point_shadow_time: Duration,
    point_shadow_cpu_timings: PointShadowCpuTimings,
    last_resource_upload_time: Duration,
    last_vertex_upload_time: Duration,
    last_texture_upload_time: Duration,
    last_viewport_target_allocation_time: Duration,
    last_shadow_target_allocation_time: Duration,
}

impl VulkanViewport {
    /// Complete queued GPU work before worker-owned Vulkan resources are dropped.
    pub fn wait_idle(&self) {
        let _ = self.device.poll(wgpu::Maintain::Wait);
    }
}

struct CachedTargets {
    size: [u32; 2],
    color: Arc<GpuImage>,
    _depth: wgpu::Texture,
    depth_view: Arc<wgpu::TextureView>,
}

struct TimestampReadback {
    buffer: wgpu::Buffer,
    pending: Option<std::sync::mpsc::Receiver<Result<(), wgpu::BufferAsyncError>>>,
    directional_written: bool,
    point_written: bool,
}

struct GpuTimestampProfiler {
    queries: wgpu::QuerySet,
    resolve: wgpu::Buffer,
    readbacks: Vec<TimestampReadback>,
    period_ns: f32,
    active_readback: Option<usize>,
    submitted_readback: Option<usize>,
    directional: Duration,
    point: Duration,
    viewport: Duration,
}

impl GpuTimestampProfiler {
    fn new(device: &wgpu::Device, queue: &wgpu::Queue) -> Option<Self> {
        if !device.features().contains(
            wgpu::Features::TIMESTAMP_QUERY | wgpu::Features::TIMESTAMP_QUERY_INSIDE_PASSES,
        ) {
            return None;
        }
        let queries = device.create_query_set(&wgpu::QuerySetDescriptor {
            label: Some("ZeroFPS GPU pass timestamps"),
            ty: wgpu::QueryType::Timestamp,
            count: 6,
        });
        let resolve = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("ZeroFPS GPU timestamp resolve"),
            size: wgpu::QUERY_RESOLVE_BUFFER_ALIGNMENT,
            usage: wgpu::BufferUsages::QUERY_RESOLVE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let readbacks = (0..3)
            .map(|_| TimestampReadback {
                buffer: device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("ZeroFPS GPU timestamp readback ring"),
                    size: 6 * 8,
                    usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ,
                    mapped_at_creation: false,
                }),
                pending: None,
                directional_written: false,
                point_written: false,
            })
            .collect();
        Some(Self {
            queries,
            resolve,
            readbacks,
            period_ns: queue.get_timestamp_period(),
            active_readback: None,
            submitted_readback: None,
            directional: Duration::ZERO,
            point: Duration::ZERO,
            viewport: Duration::ZERO,
        })
    }

    fn poll(&mut self, device: &wgpu::Device) {
        let _ = device.poll(wgpu::Maintain::Poll);
        for readback in &mut self.readbacks {
            let completed = readback
                .pending
                .as_ref()
                .is_some_and(|receiver| receiver.try_recv().is_ok_and(|result| result.is_ok()));
            if !completed {
                continue;
            }
            let mapped = readback.buffer.slice(..).get_mapped_range();
            let values = bytemuck::cast_slice::<u8, u64>(&mapped);
            let elapsed = |begin: usize, end: usize| {
                let ticks = values[end].saturating_sub(values[begin]);
                Duration::from_secs_f64(ticks as f64 * self.period_ns as f64 * 1.0e-9)
            };
            self.directional = if readback.directional_written {
                elapsed(0, 1)
            } else {
                Duration::ZERO
            };
            self.point = if readback.point_written {
                elapsed(2, 3)
            } else {
                Duration::ZERO
            };
            self.viewport = elapsed(4, 5);
            drop(mapped);
            readback.buffer.unmap();
            readback.pending = None;
        }
        self.active_readback = self
            .readbacks
            .iter()
            .position(|readback| readback.pending.is_none());
    }

    fn resolve(
        &mut self,
        encoder: &mut wgpu::CommandEncoder,
        directional_written: bool,
        point_written: bool,
    ) {
        let Some(index) = self.active_readback.take() else {
            return;
        };
        if directional_written {
            encoder.resolve_query_set(&self.queries, 0..2, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &self.resolve,
                0,
                &self.readbacks[index].buffer,
                0,
                2 * 8,
            );
        }
        if point_written {
            encoder.resolve_query_set(&self.queries, 2..4, &self.resolve, 0);
            encoder.copy_buffer_to_buffer(
                &self.resolve,
                0,
                &self.readbacks[index].buffer,
                2 * 8,
                2 * 8,
            );
        }
        encoder.resolve_query_set(&self.queries, 4..6, &self.resolve, 0);
        encoder.copy_buffer_to_buffer(
            &self.resolve,
            0,
            &self.readbacks[index].buffer,
            4 * 8,
            2 * 8,
        );
        self.readbacks[index].directional_written = directional_written;
        self.readbacks[index].point_written = point_written;
        self.submitted_readback = Some(index);
    }

    fn map_after_submit(&mut self) {
        let Some(index) = self.submitted_readback.take() else {
            return;
        };
        let readback = &mut self.readbacks[index];
        let (sender, receiver) = std::sync::mpsc::sync_channel(1);
        readback
            .buffer
            .slice(..)
            .map_async(wgpu::MapMode::Read, move |result| {
                let _ = sender.send(result);
            });
        readback.pending = Some(receiver);
    }
}

impl VulkanViewport {
    pub fn new() -> Result<Self, String> {
        let runtime = shared_runtime()?;
        let device = Arc::clone(&runtime.device);
        let queue = Arc::clone(&runtime.queue);
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport camera layout"),
            entries: &[
                camera_uniform_layout_entry(),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 2,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: false },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
            ],
        });
        let texture_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport texture layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Texture {
                        sample_type: wgpu::TextureSampleType::Float { filterable: true },
                        view_dimension: wgpu::TextureViewDimension::D2,
                        multisampled: false,
                    },
                    count: None,
                },
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Sampler(wgpu::SamplerBindingType::Filtering),
                    count: None,
                },
            ],
        });
        let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("viewport shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan_viewport.wgsl").into()),
        });
        let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
            label: Some("viewport pipeline layout"),
            bind_group_layouts: &[&camera_layout, &texture_layout],
            push_constant_ranges: &[],
        });
        let attributes = wgpu::vertex_attr_array![
            0 => Float32x4, 1 => Float32x4, 2 => Float32x4, 3 => Float32x4,
            4 => Float32x4, 5 => Float32x4, 6 => Float32x4, 7 => Float32x4
        ];
        let viewport_pipeline = |label, depth_write_enabled| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shader,
                    entry_point: Some("vs_main"),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attributes,
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shader,
                    entry_point: Some("fs_main"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::Rgba8UnormSrgb,
                        blend: Some(wgpu::BlendState::ALPHA_BLENDING),
                        write_mask: wgpu::ColorWrites::ALL,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let pipeline = viewport_pipeline("ZeroFPS Vulkan opaque viewport", true);
        let transparent_pipeline = viewport_pipeline("ZeroFPS Vulkan transparent viewport", false);
        let shadow_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("GPU shadow uniform layout"),
            entries: &[
                shadow_uniform_layout_entry(),
                wgpu::BindGroupLayoutEntry {
                    binding: 1,
                    visibility: wgpu::ShaderStages::VERTEX,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
            ],
        });
        let shadow_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: Some("GPU shadow shader"),
            source: wgpu::ShaderSource::Wgsl(include_str!("vulkan_shadow.wgsl").into()),
        });
        let shadow_pipeline_layout =
            device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("GPU shadow pipeline layout"),
                bind_group_layouts: &[&shadow_layout],
                push_constant_ranges: &[],
            });
        let shadow_pipeline = |label, entry_point| {
            device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
                label: Some(label),
                layout: Some(&shadow_pipeline_layout),
                vertex: wgpu::VertexState {
                    module: &shadow_shader,
                    entry_point: Some(entry_point),
                    buffers: &[wgpu::VertexBufferLayout {
                        array_stride: std::mem::size_of::<GpuVertex>() as u64,
                        step_mode: wgpu::VertexStepMode::Vertex,
                        attributes: &attributes,
                    }],
                    compilation_options: Default::default(),
                },
                fragment: Some(wgpu::FragmentState {
                    module: &shadow_shader,
                    entry_point: Some("fs_shadow"),
                    targets: &[Some(wgpu::ColorTargetState {
                        format: wgpu::TextureFormat::R32Float,
                        blend: None,
                        write_mask: wgpu::ColorWrites::RED,
                    })],
                    compilation_options: Default::default(),
                }),
                primitive: wgpu::PrimitiveState {
                    cull_mode: None,
                    ..Default::default()
                },
                depth_stencil: Some(wgpu::DepthStencilState {
                    format: wgpu::TextureFormat::Depth32Float,
                    depth_write_enabled: true,
                    depth_compare: wgpu::CompareFunction::Less,
                    stencil: Default::default(),
                    bias: Default::default(),
                }),
                multisample: Default::default(),
                multiview: None,
                cache: None,
            })
        };
        let directional_shadow_pipeline =
            shadow_pipeline("GPU directional shadow pipeline", "vs_shadow");
        let point_shadow_pipeline = shadow_pipeline("GPU point shadow pipeline", "vs_point_shadow");
        let shadow_uniform_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("resident shadow-view uniforms"),
            size: SHADOW_UNIFORM_STRIDE * MAX_SHADOW_VIEWS,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let point_shadow_view_buffers = (0..3)
            .map(|_| {
                device.create_buffer(&wgpu::BufferDescriptor {
                    label: Some("resident point-shadow view ring"),
                    size: std::mem::size_of::<ShadowUniform>() as u64 * MAX_SHADOW_VIEWS,
                    usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_DST,
                    mapped_at_creation: false,
                })
            })
            .collect::<Vec<_>>();
        let shadow_uniform_groups = point_shadow_view_buffers
            .iter()
            .map(|point_shadow_view_buffer| {
                device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("resident shadow-view uniform group"),
                    layout: &shadow_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                                buffer: &shadow_uniform_buffer,
                                offset: 0,
                                size: std::num::NonZeroU64::new(
                                    std::mem::size_of::<ShadowUniform>() as u64,
                                ),
                            }),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: point_shadow_view_buffer.as_entire_binding(),
                        },
                    ],
                })
            })
            .collect::<Vec<_>>();
        let shadow_draw_indirect = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("resident shadow indirect draw"),
            contents: bytemuck::cast_slice(&[0_u32, 1, 0, 0]),
            usage: wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::COPY_DST,
        });
        let timestamp_profiler = GpuTimestampProfiler::new(&device, &queue);
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
            mipmap_filter: wgpu::FilterMode::Linear,
            lod_min_clamp: 0.0,
            lod_max_clamp: 2.0,
            ..Default::default()
        });
        let white = Arc::new(upload_texture(
            &device,
            &queue,
            1,
            1,
            &[255; 4],
            "white texture",
        ));
        let camera_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("viewport camera"),
            size: std::mem::size_of::<CameraUniform>() as u64,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        Ok(Self {
            device,
            queue,
            camera_layout,
            texture_layout,
            pipeline,
            transparent_pipeline,
            shadow_uniform_buffer,
            point_shadow_view_buffers,
            shadow_uniform_groups,
            shadow_frame_slot: 0,
            directional_shadow_pipeline,
            point_shadow_pipeline,
            sampler,
            texture_cache: HashMap::new(),
            white,
            targets: None,
            camera_buffer,
            shadow_cache: None,
            point_shadow_cache: None,
            retired_shadow_targets: VecDeque::new(),
            shadow_depth_targets: HashMap::new(),
            gpu_shadow_key: None,
            gpu_point_shadow_key: None,
            gpu_directional_metadata: None,
            gpu_point_metadata: None,
            camera_group_cache: None,
            vertex_cache: HashMap::new(),
            shadow_vertex_buffer: None,
            shadow_draw_indirect,
            timestamp_profiler,
            timestamp_poll_frame: 0,
            pending_shadow_commands: Vec::with_capacity(2),
            point_shadow_staging: wgpu::util::StagingBelt::new(64 * 1024),
            timestamp_directional_written: false,
            timestamp_point_written: false,
            bind_group_cache: HashMap::new(),
            last_shadow_encode_time: Duration::ZERO,
            last_directional_shadow_time: Duration::ZERO,
            last_point_shadow_time: Duration::ZERO,
            point_shadow_cpu_timings: PointShadowCpuTimings::default(),
            last_resource_upload_time: Duration::ZERO,
            last_vertex_upload_time: Duration::ZERO,
            last_texture_upload_time: Duration::ZERO,
            last_viewport_target_allocation_time: Duration::ZERO,
            last_shadow_target_allocation_time: Duration::ZERO,
        })
    }

    pub fn render_resident(
        &mut self,
        size: Vec2,
        camera: (f32, f32, f32, f32, Vec3, f32, u32),
        batches: &[GpuBatch],
        global_light_enabled: bool,
        point_lights: &[ViewportLight],
        directional_shadow: Option<&DirectionalShadowMap>,
        point_shadows: Option<&PointShadowAtlas>,
        gpu_shadow_revision: Option<u64>,
        batch_revision: u64,
        global_shadow_resolution: u32,
        shadow_filter_radius: usize,
        max_vram_bytes: u64,
    ) -> Result<Arc<GpuImage>, String> {
        let width = size.x.round().max(1.0) as u32;
        let height = size.y.round().max(1.0) as u32;
        self.last_resource_upload_time = Duration::ZERO;
        self.last_vertex_upload_time = Duration::ZERO;
        self.last_texture_upload_time = Duration::ZERO;
        self.last_viewport_target_allocation_time = Duration::ZERO;
        self.last_shadow_target_allocation_time = Duration::ZERO;
        // Mapping timestamp readbacks asks wgpu to poll the device. Even a
        // nominally non-blocking poll can occasionally enter substantial
        // driver housekeeping when the compositor and viewport submit at the
        // same time. Telemetry does not need frame-rate sampling, so keep that
        // work off seven out of every eight interaction-critical frames.
        if self.timestamp_poll_frame % 8 == 0
            && let Some(profiler) = &mut self.timestamp_profiler
        {
            profiler.poll(&self.device);
        }
        self.timestamp_poll_frame = self.timestamp_poll_frame.wrapping_add(1);
        self.timestamp_directional_written = false;
        self.timestamp_point_written = false;
        self.shadow_frame_slot = (self.shadow_frame_slot + 1) % self.shadow_uniform_groups.len();
        self.pending_shadow_commands.clear();
        self.trim_retired_shadow_targets(max_vram_bytes);
        let target_changed = !self
            .targets
            .as_ref()
            .is_some_and(|targets| targets.size == [width, height]);
        let target_started = Instant::now();
        self.ensure_targets(width, height);
        if target_changed {
            let elapsed = target_started.elapsed();
            self.last_resource_upload_time += elapsed;
            self.last_viewport_target_allocation_time += elapsed;
        }
        let shadow_started = Instant::now();
        let (gpu_directional, gpu_points) = if let Some(revision) = gpu_shadow_revision {
            self.render_gpu_shadow_maps(
                batches,
                point_lights,
                global_light_enabled
                    .then_some(global_shadow_resolution)
                    .unwrap_or(0),
                shadow_filter_radius,
                revision,
                batch_revision,
            )?
        } else {
            (None, None)
        };
        self.last_shadow_encode_time = shadow_started.elapsed();
        let directional_shadow = gpu_directional.as_ref().or(directional_shadow);
        let point_shadows = gpu_points.as_ref().or(point_shadows);
        let mut point_positions = [[0.0; 4]; MAX_VIEWPORT_LIGHTS];
        let mut point_colors = [[0.0; 4]; MAX_VIEWPORT_LIGHTS];
        let mut point_shadow_regions = [[0.0; 4]; MAX_VIEWPORT_LIGHTS];
        for (index, light) in point_lights.iter().take(MAX_VIEWPORT_LIGHTS).enumerate() {
            point_positions[index] = [
                light.position.x,
                light.position.y,
                light.position.z,
                light.intensity.max(0.0),
            ];
            point_colors[index] = [
                light.color[0].max(0.0),
                light.color[1].max(0.0),
                light.color[2].max(0.0),
                light.radius.max(0.0),
            ];
        }
        let shadow_origin = directional_shadow.map_or([0.0; 4], |shadow| {
            [shadow.origin.x, shadow.origin.y, shadow.origin.z, 0.0]
        });
        let shadow_right = directional_shadow.map_or([0.0; 4], |shadow| {
            [shadow.right.x, shadow.right.y, shadow.right.z, 0.0]
        });
        let shadow_up = directional_shadow.map_or([0.0; 4], |shadow| {
            [shadow.up.x, shadow.up.y, shadow.up.z, 0.0]
        });
        let shadow_forward = directional_shadow.map_or([0.0; 4], |shadow| {
            [shadow.forward.x, shadow.forward.y, shadow.forward.z, 0.0]
        });
        let shadow_parameters = directional_shadow.map_or([0.0; 4], |shadow| {
            [
                shadow.extent,
                shadow.bias,
                shadow.resolution as f32,
                shadow.filter_radius as f32,
            ]
        });
        if let Some(atlas) = point_shadows {
            for (target, region) in point_shadow_regions.iter_mut().zip(atlas.regions) {
                *target = [
                    region.row as f32,
                    region.resolution as f32,
                    region.bias,
                    region.filter_radius as f32,
                ];
            }
        }
        let point_shadow_atlas_size = point_shadows.map_or([1.0, 1.0, 0.0, 0.0], |atlas| {
            [atlas.width as f32, atlas.height as f32, 0.0, 0.0]
        });
        let uniform = CameraUniform {
            yaw: camera.0,
            pitch: camera.1,
            roll: camera.2,
            zoom: camera.3,
            grid_spacing: camera.5,
            camera_padding: [0.0; 3],
            target: [camera.4.x, camera.4.y, camera.4.z, 0.0],
            viewport: [width as f32, height as f32],
            projection: camera.6,
            padding: 0,
            global_light_enabled: u32::from(global_light_enabled),
            point_light_count: point_lights.len().min(MAX_VIEWPORT_LIGHTS) as u32,
            directional_shadow_enabled: u32::from(directional_shadow.is_some()),
            lighting_padding: 0,
            point_positions,
            point_colors,
            point_shadow_regions,
            shadow_origin,
            shadow_right,
            shadow_up,
            shadow_forward,
            shadow_parameters,
            point_shadow_atlas_size,
        };
        self.queue
            .write_buffer(&self.camera_buffer, 0, bytemuck::bytes_of(&uniform));
        let color_target = Arc::clone(
            &self
                .targets
                .as_ref()
                .expect("viewport targets initialized")
                .color,
        );
        let depth_view = Arc::clone(
            &self
                .targets
                .as_ref()
                .expect("viewport targets initialized")
                .depth_view,
        );
        let shadow_view = if gpu_directional.is_some() {
            Arc::clone(&self.shadow_cache.as_ref().expect("GPU shadow target").4)
        } else {
            self.shadow_view(directional_shadow)
        };
        let point_shadow_view = if gpu_points.is_some() {
            Arc::clone(
                &self
                    .point_shadow_cache
                    .as_ref()
                    .expect("GPU point shadow target")
                    .4,
            )
        } else {
            self.point_shadow_view(point_shadows)
        };
        let shadow_key = Arc::as_ptr(&shadow_view) as usize;
        let point_shadow_key = Arc::as_ptr(&point_shadow_view) as usize;
        let camera_group = match self.camera_group_cache.as_ref() {
            Some((cached_shadow, cached_point, group))
                if *cached_shadow == shadow_key && *cached_point == point_shadow_key =>
            {
                Arc::clone(group)
            }
            _ => {
                let group = Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                    label: Some("viewport camera and shadow group"),
                    layout: &self.camera_layout,
                    entries: &[
                        wgpu::BindGroupEntry {
                            binding: 0,
                            resource: self.camera_buffer.as_entire_binding(),
                        },
                        wgpu::BindGroupEntry {
                            binding: 1,
                            resource: wgpu::BindingResource::TextureView(&shadow_view),
                        },
                        wgpu::BindGroupEntry {
                            binding: 2,
                            resource: wgpu::BindingResource::TextureView(&point_shadow_view),
                        },
                    ],
                }));
                self.camera_group_cache = Some((shadow_key, point_shadow_key, Arc::clone(&group)));
                group
            }
        };
        let viewport_timestamp_writes = self.timestamp_profiler.as_ref().and_then(|profiler| {
            profiler
                .active_readback
                .map(|_| wgpu::RenderPassTimestampWrites {
                    query_set: &profiler.queries,
                    beginning_of_pass_write_index: Some(4),
                    end_of_pass_write_index: Some(5),
                })
        });
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut live_bind_groups = HashSet::new();
        let mut live_vertex_buffers = HashSet::new();
        let mut viewport_batches = batches.iter().collect::<Vec<_>>();
        let camera_depth = |batch: &GpuBatch| {
            let local_center = (batch.local_bounds_min + batch.local_bounds_max) * 0.5;
            let world_center = batch
                .object_transform
                .rotation
                .rotate(batch.object_transform.scale.component_mul(local_center))
                + batch.object_transform.translation;
            let relative = world_center - camera.4;
            let forward = relative.x * camera.0.sin() + relative.y * camera.0.cos();
            relative.z * camera.1.sin() + forward * camera.1.cos()
        };
        viewport_batches.sort_by(|left, right| {
            left.transparent.cmp(&right.transparent).then_with(|| {
                if left.transparent {
                    camera_depth(right)
                        .partial_cmp(&camera_depth(left))
                        .unwrap_or(std::cmp::Ordering::Equal)
                } else {
                    std::cmp::Ordering::Equal
                }
            })
        });
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("viewport pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_target.view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color::TRANSPARENT),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth_view,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Store,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes: viewport_timestamp_writes,
                occlusion_query_set: None,
            });
            pass.set_bind_group(0, camera_group.as_ref(), &[]);
            for batch in viewport_batches {
                pass.set_pipeline(if batch.transparent {
                    &self.transparent_pipeline
                } else {
                    &self.pipeline
                });
                if batch.vertices.is_empty() {
                    continue;
                }
                let view = batch
                    .gpu_texture
                    .as_ref()
                    .map(|image| Arc::clone(&image.view))
                    .unwrap_or_else(|| {
                        self.texture_view(batch.texture_cache_key, batch.texture.as_ref())
                    });
                let view_key = Arc::as_ptr(&view) as usize;
                live_bind_groups.insert(view_key);
                let texture_group =
                    Arc::clone(self.bind_group_cache.entry(view_key).or_insert_with(|| {
                        Arc::new(self.device.create_bind_group(&wgpu::BindGroupDescriptor {
                            label: Some("viewport texture group"),
                            layout: &self.texture_layout,
                            entries: &[
                                wgpu::BindGroupEntry {
                                    binding: 0,
                                    resource: wgpu::BindingResource::TextureView(&view),
                                },
                                wgpu::BindGroupEntry {
                                    binding: 1,
                                    resource: wgpu::BindingResource::Sampler(&self.sampler),
                                },
                            ],
                        }))
                    }));
                let vertices = self.vertex_buffer(batch, batch_revision);
                live_vertex_buffers.insert(batch.cache_key);
                pass.set_bind_group(1, texture_group.as_ref(), &[]);
                pass.set_vertex_buffer(0, vertices.as_ref().slice(..));
                pass.draw(0..batch.vertices.len() as u32, 0..1);
            }
        }
        // Bind groups retain their texture views, and therefore the complete
        // compositor output allocation. Previous graph generations are not
        // useful after a new frame is encoded, so keep only this frame's live
        // resources instead of maintaining a historical cache.
        self.bind_group_cache
            .retain(|key, _| live_bind_groups.contains(key));
        // CPU compositor results and other procedural textures may receive a
        // fresh Arc identity every update. Retaining those pointer-keyed
        // uploads forever leaks one GPU texture per computed frame. Keep only
        // texture views referenced by the frame that was just encoded.
        self.texture_cache
            .retain(|_, (_, view)| live_bind_groups.contains(&(Arc::as_ptr(view) as usize)));
        self.vertex_cache
            .retain(|key, _| live_vertex_buffers.contains(key));
        if let Some(profiler) = &mut self.timestamp_profiler {
            profiler.resolve(
                &mut encoder,
                self.timestamp_directional_written,
                self.timestamp_point_written,
            );
        }
        let viewport_commands = encoder.finish();
        self.point_shadow_staging.finish();
        self.queue.submit(
            self.pending_shadow_commands
                .drain(..)
                .chain(std::iter::once(viewport_commands)),
        );
        self.point_shadow_staging.recall();
        if let Some(profiler) = &mut self.timestamp_profiler {
            profiler.map_after_submit();
        }
        Ok(color_target)
    }

    pub fn last_shadow_encode_time(&self) -> Duration {
        self.last_shadow_encode_time
    }

    pub fn last_directional_shadow_time(&self) -> Duration {
        self.last_directional_shadow_time
    }

    pub fn last_point_shadow_time(&self) -> Duration {
        self.last_point_shadow_time
    }

    pub fn point_shadow_cpu_timings(&self) -> PointShadowCpuTimings {
        self.point_shadow_cpu_timings
    }

    pub fn gpu_directional_shadow_time(&self) -> Duration {
        self.timestamp_profiler
            .as_ref()
            .map_or(Duration::ZERO, |profiler| profiler.directional)
    }

    pub fn gpu_point_shadow_time(&self) -> Duration {
        self.timestamp_profiler
            .as_ref()
            .map_or(Duration::ZERO, |profiler| profiler.point)
    }

    pub fn gpu_viewport_time(&self) -> Duration {
        self.timestamp_profiler
            .as_ref()
            .map_or(Duration::ZERO, |profiler| profiler.viewport)
    }

    pub fn last_resource_upload_time(&self) -> Duration {
        self.last_resource_upload_time
    }

    pub fn last_vertex_upload_time(&self) -> Duration {
        self.last_vertex_upload_time
    }

    pub fn last_texture_upload_time(&self) -> Duration {
        self.last_texture_upload_time
    }

    pub fn last_viewport_target_allocation_time(&self) -> Duration {
        self.last_viewport_target_allocation_time
    }

    pub fn last_shadow_target_allocation_time(&self) -> Duration {
        self.last_shadow_target_allocation_time
    }

    fn render_gpu_shadow_maps(
        &mut self,
        batches: &[GpuBatch],
        lights: &[ViewportLight],
        global_resolution: u32,
        filter_radius: usize,
        _revision: u64,
        batch_revision: u64,
    ) -> Result<(Option<DirectionalShadowMap>, Option<PointShadowAtlas>), String> {
        self.last_directional_shadow_time = Duration::ZERO;
        self.last_point_shadow_time = Duration::ZERO;
        self.point_shadow_cpu_timings = PointShadowCpuTimings::default();
        let geometry_key = shadow_geometry_key(batches);
        let directional = if global_resolution > 0 {
            let key = geometry_key
                ^ (global_resolution as u64).rotate_left(11)
                ^ (filter_radius as u64).rotate_left(23);
            if self.gpu_shadow_key != Some(key) {
                let started = Instant::now();
                let metadata = directional_metadata(batches, global_resolution, filter_radius);
                self.render_gpu_directional_shadow(batches, &metadata, key, batch_revision)?;
                self.last_directional_shadow_time = started.elapsed();
                self.gpu_directional_metadata = Some(metadata);
                self.gpu_shadow_key = Some(key);
            }
            self.gpu_directional_metadata.clone()
        } else {
            self.gpu_directional_metadata = None;
            self.gpu_shadow_key = None;
            None
        };
        let point_key = lights
            .iter()
            .fold(geometry_key.rotate_left(7), |key, light| {
                key.rotate_left(5)
                    ^ light.shadow_resolution as u64
                    ^ (light.position.x.to_bits() as u64).rotate_left(13)
                    ^ (light.position.y.to_bits() as u64).rotate_left(29)
                    ^ (light.position.z.to_bits() as u64).rotate_left(41)
                    ^ (light.radius.to_bits() as u64).rotate_left(17)
            });
        let points_required = lights.iter().any(|light| light.shadow_resolution > 0);
        if points_required {
            if self.gpu_point_shadow_key != Some(point_key) {
                let started = Instant::now();
                let points = point_metadata(lights, filter_radius);
                self.render_gpu_point_shadows(batches, lights, &points, point_key, batch_revision)?;
                self.last_point_shadow_time = started.elapsed();
                self.gpu_point_metadata = Some(points);
                self.gpu_point_shadow_key = Some(point_key);
            }
            Ok((directional, self.gpu_point_metadata.clone()))
        } else {
            self.gpu_point_metadata = None;
            self.gpu_point_shadow_key = None;
            Ok((directional, None))
        }
    }

    fn render_gpu_directional_shadow(
        &mut self,
        batches: &[GpuBatch],
        metadata: &DirectionalShadowMap,
        key: u64,
        _batch_revision: u64,
    ) -> Result<(), String> {
        let resolution = metadata.resolution as u32;
        self.ensure_shadow_target(resolution, resolution, key as usize, false);
        let uniform = ShadowUniform {
            origin: [metadata.origin.x, metadata.origin.y, metadata.origin.z, 0.0],
            right: [metadata.right.x, metadata.right.y, metadata.right.z, 0.0],
            up: [metadata.up.x, metadata.up.y, metadata.up.z, 0.0],
            forward: [
                metadata.forward.x,
                metadata.forward.y,
                metadata.forward.z,
                0.0,
            ],
            parameters: [metadata.extent, metadata.depth[0], metadata.depth[1], 0.0],
            atlas: [1.0, 1.0, 0.0, 0.0],
        };
        self.write_shadow_uniforms(0, &[uniform]);
        let color_view = Arc::clone(&self.shadow_cache.as_ref().unwrap().4);
        let depth = self.shadow_depth_target(resolution, resolution);
        let (vertex_count, shadow_vertices) = self.shadow_vertices(batches);
        self.queue.write_buffer(
            &self.shadow_draw_indirect,
            0,
            bytemuck::cast_slice(&[vertex_count, 1_u32, 0, 0]),
        );
        let timestamp_writes = self.timestamp_profiler.as_ref().and_then(|profiler| {
            profiler
                .active_readback
                .map(|_| wgpu::RenderPassTimestampWrites {
                    query_set: &profiler.queries,
                    beginning_of_pass_write_index: Some(0),
                    end_of_pass_write_index: Some(1),
                })
        });
        self.timestamp_directional_written = timestamp_writes.is_some();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GPU directional shadow pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::INFINITY,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.directional_shadow_pipeline);
            pass.set_bind_group(0, &self.shadow_uniform_groups[self.shadow_frame_slot], &[0]);
            pass.set_vertex_buffer(0, shadow_vertices.slice(..));
            pass.draw_indirect(&self.shadow_draw_indirect, 0);
        }
        self.pending_shadow_commands.push(encoder.finish());
        Ok(())
    }

    fn render_gpu_point_shadows(
        &mut self,
        batches: &[GpuBatch],
        lights: &[ViewportLight],
        metadata: &PointShadowAtlas,
        key: u64,
        _batch_revision: u64,
    ) -> Result<(), String> {
        let width = metadata.width as u32;
        let height = metadata.height as u32;
        let setup_started = Instant::now();
        self.ensure_shadow_target(width, height, key as usize, true);
        let color_view = Arc::clone(&self.point_shadow_cache.as_ref().unwrap().4);
        let depth = self.shadow_depth_target(width, height);
        let far_depth = point_shadow_far_depth(batches, lights);
        let mut uniforms = Vec::new();
        for (light_index, light) in lights.iter().take(MAX_VIEWPORT_LIGHTS).enumerate() {
            let region = metadata.regions[light_index];
            if region.resolution == 0 {
                continue;
            }
            for face in 0..6 {
                let (forward, right, up) = crate::cube_face_basis(face);
                let uniform = ShadowUniform {
                    origin: [light.position.x, light.position.y, light.position.z, 0.0],
                    right: [right.x, right.y, right.z, 0.0],
                    up: [up.x, up.y, up.z, 0.0],
                    forward: [forward.x, forward.y, forward.z, 0.0],
                    parameters: [far_depth, width as f32, height as f32, 0.0],
                    atlas: [
                        region.resolution as f32 / width.max(1) as f32,
                        region.resolution as f32 / height.max(1) as f32,
                        2.0 * ((face as f32 + 0.5) * region.resolution as f32)
                            / width.max(1) as f32
                            - 1.0,
                        1.0 - 2.0 * (region.row as f32 + 0.5 * region.resolution as f32)
                            / height.max(1) as f32,
                    ],
                };
                uniforms.push(uniform);
            }
        }
        self.point_shadow_cpu_timings.setup = setup_started.elapsed();
        let encoder_started = Instant::now();
        let mut encoder = self.device.create_command_encoder(&Default::default());
        self.point_shadow_cpu_timings.encoder_create = encoder_started.elapsed();
        let uniform_started = Instant::now();
        let uniform_bytes = bytemuck::cast_slice(&uniforms);
        if let Some(size) = wgpu::BufferSize::new(uniform_bytes.len() as u64) {
            self.point_shadow_staging
                .write_buffer(
                    &mut encoder,
                    &self.point_shadow_view_buffers[self.shadow_frame_slot],
                    0,
                    size,
                    &self.device,
                )
                .copy_from_slice(uniform_bytes);
        }
        self.point_shadow_cpu_timings.uniform_write = uniform_started.elapsed();
        let geometry_started = Instant::now();
        let (_vertex_count, shadow_vertices) = self.shadow_vertices(batches);
        if let Some((_, _, count)) = &self.shadow_vertex_buffer {
            self.queue.write_buffer(
                &self.shadow_draw_indirect,
                0,
                bytemuck::cast_slice(&[*count, uniforms.len() as u32, 0, 0]),
            );
        }
        self.point_shadow_cpu_timings.geometry_lookup = geometry_started.elapsed();
        let timestamp_writes = self.timestamp_profiler.as_ref().and_then(|profiler| {
            profiler
                .active_readback
                .map(|_| wgpu::RenderPassTimestampWrites {
                    query_set: &profiler.queries,
                    beginning_of_pass_write_index: Some(2),
                    end_of_pass_write_index: Some(3),
                })
        });
        self.timestamp_point_written = timestamp_writes.is_some();
        let pass_started = Instant::now();
        {
            let mut pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: Some("GPU point shadow atlas pass"),
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &color_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: f64::INFINITY,
                            g: 0.0,
                            b: 0.0,
                            a: 0.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                })],
                depth_stencil_attachment: Some(wgpu::RenderPassDepthStencilAttachment {
                    view: &depth,
                    depth_ops: Some(wgpu::Operations {
                        load: wgpu::LoadOp::Clear(1.0),
                        store: wgpu::StoreOp::Discard,
                    }),
                    stencil_ops: None,
                }),
                timestamp_writes,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.point_shadow_pipeline);
            pass.set_bind_group(0, &self.shadow_uniform_groups[self.shadow_frame_slot], &[0]);
            pass.set_vertex_buffer(0, shadow_vertices.slice(..));
            pass.draw_indirect(&self.shadow_draw_indirect, 0);
        }
        self.pending_shadow_commands.push(encoder.finish());
        self.point_shadow_cpu_timings.pass_and_finish = pass_started.elapsed();
        Ok(())
    }

    fn write_shadow_uniforms(&self, first_slot: u64, uniforms: &[ShadowUniform]) {
        if uniforms.is_empty() {
            return;
        }
        let stride = SHADOW_UNIFORM_STRIDE as usize;
        let mut bytes = vec![0_u8; stride * uniforms.len()];
        for (index, uniform) in uniforms.iter().enumerate() {
            let encoded = bytemuck::bytes_of(uniform);
            bytes[index * stride..index * stride + encoded.len()].copy_from_slice(encoded);
        }
        self.queue.write_buffer(
            &self.shadow_uniform_buffer,
            first_slot * SHADOW_UNIFORM_STRIDE,
            &bytes,
        );
    }

    fn shadow_vertices(&mut self, batches: &[GpuBatch]) -> (u32, Arc<wgpu::Buffer>) {
        let caster_batches = batches
            .iter()
            .filter(|batch| batch.casts_shadows && !batch.vertices.is_empty())
            .collect::<Vec<_>>();
        let mut offset = 0_usize;
        let signature = caster_batches
            .iter()
            .map(|batch| {
                let entry = (
                    batch.cache_key,
                    batch.content_revision,
                    offset,
                    batch.vertices.len(),
                );
                offset += batch.vertices.len();
                entry
            })
            .collect::<Vec<_>>();
        let structure_matches = self
            .shadow_vertex_buffer
            .as_ref()
            .is_some_and(|(cached, _, _)| {
                cached.len() == signature.len()
                    && cached
                        .iter()
                        .zip(&signature)
                        .all(|(old, new)| old.0 == new.0 && old.2 == new.2 && old.3 == new.3)
            });
        if structure_matches {
            let (cached, buffer, count) = self.shadow_vertex_buffer.as_mut().unwrap();
            let started = Instant::now();
            let mut uploaded = false;
            for ((old, new), batch) in cached.iter_mut().zip(&signature).zip(&caster_batches) {
                if old.1 != new.1 {
                    self.queue.write_buffer(
                        buffer,
                        (new.2 * std::mem::size_of::<GpuVertex>()) as u64,
                        bytemuck::cast_slice(&batch.vertices),
                    );
                    old.1 = new.1;
                    uploaded = true;
                }
            }
            if uploaded {
                let elapsed = started.elapsed();
                self.last_resource_upload_time += elapsed;
                self.last_vertex_upload_time += elapsed;
            }
            return (*count, Arc::clone(buffer));
        }

        let started = Instant::now();
        let vertices = caster_batches
            .iter()
            .flat_map(|batch| batch.vertices.iter().copied())
            .collect::<Vec<_>>();
        // A zero-sized wgpu buffer cannot be bound or sliced. Preserve the real
        // draw count, but allocate one unreachable vertex for an empty scene.
        let upload_vertices = if vertices.is_empty() {
            vec![<GpuVertex as bytemuck::Zeroable>::zeroed()]
        } else {
            vertices.clone()
        };
        let buffer = Arc::new(
            self.device
                .create_buffer_init(&wgpu::util::BufferInitDescriptor {
                    label: Some("resident consolidated shadow casters"),
                    contents: bytemuck::cast_slice(&upload_vertices),
                    usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                }),
        );
        let count = vertices.len() as u32;
        self.queue.write_buffer(
            &self.shadow_draw_indirect,
            0,
            bytemuck::cast_slice(&[count, 1_u32, 0, 0]),
        );
        self.shadow_vertex_buffer = Some((signature, Arc::clone(&buffer), count));
        let elapsed = started.elapsed();
        self.last_resource_upload_time += elapsed;
        self.last_vertex_upload_time += elapsed;
        (count, buffer)
    }

    fn vertex_buffer(&mut self, batch: &GpuBatch, _revision: u64) -> Arc<wgpu::Buffer> {
        let started = Instant::now();
        match self.vertex_cache.get(&batch.cache_key) {
            Some((count, uploaded_revision, buffer))
                if *count == batch.vertices.len()
                    && *uploaded_revision == batch.content_revision =>
            {
                Arc::clone(buffer)
            }
            Some((count, _, buffer)) if *count == batch.vertices.len() => {
                self.queue
                    .write_buffer(buffer, 0, bytemuck::cast_slice(&batch.vertices));
                let buffer = Arc::clone(buffer);
                self.vertex_cache.insert(
                    batch.cache_key,
                    (
                        batch.vertices.len(),
                        batch.content_revision,
                        Arc::clone(&buffer),
                    ),
                );
                let elapsed = started.elapsed();
                self.last_resource_upload_time += elapsed;
                self.last_vertex_upload_time += elapsed;
                buffer
            }
            _ => {
                let buffer = Arc::new(self.device.create_buffer_init(
                    &wgpu::util::BufferInitDescriptor {
                        label: Some("persistent viewport vertices"),
                        contents: bytemuck::cast_slice(&batch.vertices),
                        usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                    },
                ));
                self.vertex_cache.insert(
                    batch.cache_key,
                    (
                        batch.vertices.len(),
                        batch.content_revision,
                        Arc::clone(&buffer),
                    ),
                );
                let elapsed = started.elapsed();
                self.last_resource_upload_time += elapsed;
                self.last_vertex_upload_time += elapsed;
                buffer
            }
        }
    }

    fn ensure_shadow_target(&mut self, width: u32, height: u32, key: usize, point: bool) {
        let reusable = if point {
            self.point_shadow_cache.as_ref()
        } else {
            self.shadow_cache.as_ref()
        };
        if reusable
            .as_ref()
            .is_some_and(|(cached_key, cached_width, cached_height, _, _)| {
                *cached_key == key && *cached_width == width && *cached_height == height
            })
        {
            return;
        }
        let previous = if point {
            self.point_shadow_cache.take()
        } else {
            self.shadow_cache.take()
        };
        let started = Instant::now();
        if let Some(index) = self.retired_shadow_targets.iter().position(
            |(old_width, old_height, old_point, _, _, _)| {
                *old_width == width && *old_height == height && *old_point == point
            },
        ) {
            let (_, _, _, texture, view, _) = self
                .retired_shadow_targets
                .remove(index)
                .expect("matching shadow target disappeared");
            if let Some((_, old_width, old_height, old_texture, old_view)) = previous {
                let bytes = u64::from(old_width) * u64::from(old_height) * 4;
                self.retired_shadow_targets.push_back((
                    old_width,
                    old_height,
                    point,
                    old_texture,
                    old_view,
                    bytes,
                ));
            }
            if point {
                self.point_shadow_cache = Some((key, width, height, texture, view));
            } else {
                self.shadow_cache = Some((key, width, height, texture, view));
            }
            self.camera_group_cache = None;
            return;
        }
        if let Some((_, old_width, old_height, texture, view)) = previous {
            let bytes = u64::from(old_width) * u64::from(old_height) * 4;
            self.retired_shadow_targets
                .push_back((old_width, old_height, point, texture, view, bytes));
        }
        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some(if point {
                "resident GPU point shadow atlas"
            } else {
                "resident GPU directional shadow"
            }),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::R32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        });
        let view = Arc::new(texture.create_view(&Default::default()));
        if point {
            self.point_shadow_cache = Some((key, width, height, texture, view));
        } else {
            self.shadow_cache = Some((key, width, height, texture, view));
        }
        self.camera_group_cache = None;
        let elapsed = started.elapsed();
        self.last_resource_upload_time += elapsed;
        self.last_shadow_target_allocation_time += elapsed;
    }

    fn trim_retired_shadow_targets(&mut self, max_vram_bytes: u64) {
        // Reserve 1 GiB for meshes, textures, compositor images and viewport
        // targets. Shadow history may use the remainder of the 75% safety
        // budget, but three frames (directional + point atlas) are enough to
        // remove normal write-after-read stalls.
        let shadow_budget = max_vram_bytes.saturating_sub(1024 * 1024 * 1024);
        let mut retained_bytes = self
            .retired_shadow_targets
            .iter()
            .map(|(_, _, _, _, _, bytes)| *bytes)
            .sum::<u64>();
        while self.retired_shadow_targets.len() > 6 || retained_bytes > shadow_budget {
            let Some((_, _, _, _, _, bytes)) = self.retired_shadow_targets.pop_front() else {
                break;
            };
            retained_bytes = retained_bytes.saturating_sub(bytes);
        }
    }

    fn shadow_depth_target(&mut self, width: u32, height: u32) -> Arc<wgpu::TextureView> {
        let started = Instant::now();
        let allocated = !self.shadow_depth_targets.contains_key(&(width, height));
        let targets = self
            .shadow_depth_targets
            .entry((width, height))
            .or_insert_with(|| {
                let targets = (0..3)
                    .map(|_| {
                        let texture = self.device.create_texture(&wgpu::TextureDescriptor {
                            label: Some("resident GPU shadow depth ring"),
                            size: wgpu::Extent3d {
                                width: width.max(1),
                                height: height.max(1),
                                depth_or_array_layers: 1,
                            },
                            mip_level_count: 1,
                            sample_count: 1,
                            dimension: wgpu::TextureDimension::D2,
                            format: wgpu::TextureFormat::Depth32Float,
                            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
                            view_formats: &[],
                        });
                        let view = Arc::new(texture.create_view(&Default::default()));
                        (texture, view)
                    })
                    .collect();
                (0, targets)
            });
        let index = targets.0;
        targets.0 = (targets.0 + 1) % targets.1.len();
        let view = Arc::clone(&targets.1[index].1);
        if allocated {
            let elapsed = started.elapsed();
            self.last_resource_upload_time += elapsed;
            self.last_shadow_target_allocation_time += elapsed;
        }
        view
    }

    fn ensure_targets(&mut self, width: u32, height: u32) {
        if self
            .targets
            .as_ref()
            .is_some_and(|targets| targets.size == [width, height])
        {
            return;
        }
        let extent = wgpu::Extent3d {
            width,
            height,
            depth_or_array_layers: 1,
        };
        let color_texture = Arc::new(self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport resident color"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT
                | wgpu::TextureUsages::TEXTURE_BINDING
                | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        }));
        let color_view = Arc::new(color_texture.create_view(&Default::default()));
        let depth = self.device.create_texture(&wgpu::TextureDescriptor {
            label: Some("viewport resident depth"),
            size: extent,
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Depth32Float,
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT | wgpu::TextureUsages::COPY_SRC,
            view_formats: &[],
        });
        let depth_view = Arc::new(depth.create_view(&Default::default()));
        self.targets = Some(CachedTargets {
            size: [width, height],
            color: Arc::new(GpuImage {
                _texture: color_texture,
                view: color_view,
                encoded_srgb: false,
                width,
                height,
            }),
            _depth: depth,
            depth_view,
        });
    }

    fn texture_view(
        &mut self,
        stable_key: u64,
        texture: Option<&Arc<TextureAsset>>,
    ) -> Arc<wgpu::TextureView> {
        let Some(texture) = texture else {
            return Arc::clone(&self.white);
        };
        let key = if stable_key == 0 {
            Arc::as_ptr(texture) as usize
        } else {
            stable_key as usize
        };
        if !self.texture_cache.contains_key(&key) {
            let started = Instant::now();
            let view = upload_texture(
                &self.device,
                &self.queue,
                texture.width,
                texture.height,
                &texture.pixels,
                &texture.name,
            );
            self.texture_cache
                .insert(key, (Arc::clone(texture), Arc::new(view)));
            let elapsed = started.elapsed();
            self.last_resource_upload_time += elapsed;
            self.last_texture_upload_time += elapsed;
        }
        Arc::clone(&self.texture_cache[&key].1)
    }

    fn shadow_view(&mut self, shadow: Option<&DirectionalShadowMap>) -> Arc<wgpu::TextureView> {
        let key = shadow.map_or(0, |shadow| shadow.depth.as_ptr() as usize);
        if self
            .shadow_cache
            .as_ref()
            .is_some_and(|(cached_key, _, _, _, _)| *cached_key == key)
        {
            return Arc::clone(&self.shadow_cache.as_ref().unwrap().4);
        }
        let (resolution, depth) = shadow.map_or((1, &[f32::INFINITY][..]), |shadow| {
            (shadow.resolution as u32, shadow.depth.as_slice())
        });
        if let Some((cached_key, width, height, texture, view)) = self.shadow_cache.as_mut()
            && *width == resolution
            && *height == resolution
        {
            self.queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(depth),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(resolution * 4),
                    rows_per_image: Some(resolution),
                },
                wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: 1,
                },
            );
            *cached_key = key;
            return Arc::clone(view);
        }
        self.camera_group_cache = None;
        self.shadow_cache = None;
        let _ = self.device.poll(wgpu::Maintain::Wait);
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("directional shadow depth"),
                size: wgpu::Extent3d {
                    width: resolution,
                    height: resolution,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(depth),
        );
        let view = Arc::new(texture.create_view(&Default::default()));
        self.shadow_cache = Some((key, resolution, resolution, texture, Arc::clone(&view)));
        view
    }

    fn point_shadow_view(&mut self, atlas: Option<&PointShadowAtlas>) -> Arc<wgpu::TextureView> {
        let key = atlas.map_or(0, |atlas| atlas.depth.as_ptr() as usize);
        if self
            .point_shadow_cache
            .as_ref()
            .is_some_and(|(cached_key, _, _, _, _)| *cached_key == key)
        {
            return Arc::clone(&self.point_shadow_cache.as_ref().unwrap().4);
        }
        let (width, height, depth) = atlas.map_or((1, 1, &[f32::INFINITY][..]), |atlas| {
            (
                atlas.width as u32,
                atlas.height as u32,
                atlas.depth.as_slice(),
            )
        });
        if let Some((cached_key, cached_width, cached_height, texture, view)) =
            self.point_shadow_cache.as_mut()
            && *cached_width == width
            && *cached_height == height
        {
            self.queue.write_texture(
                texture.as_image_copy(),
                bytemuck::cast_slice(depth),
                wgpu::TexelCopyBufferLayout {
                    offset: 0,
                    bytes_per_row: Some(width * 4),
                    rows_per_image: Some(height),
                },
                wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
            );
            *cached_key = key;
            return Arc::clone(view);
        }
        self.camera_group_cache = None;
        self.point_shadow_cache = None;
        let _ = self.device.poll(wgpu::Maintain::Wait);
        let texture = self.device.create_texture_with_data(
            &self.queue,
            &wgpu::TextureDescriptor {
                label: Some("point light shadow atlas"),
                size: wgpu::Extent3d {
                    width,
                    height,
                    depth_or_array_layers: 1,
                },
                mip_level_count: 1,
                sample_count: 1,
                dimension: wgpu::TextureDimension::D2,
                format: wgpu::TextureFormat::R32Float,
                usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
                view_formats: &[],
            },
            wgpu::util::TextureDataOrder::LayerMajor,
            bytemuck::cast_slice(depth),
        );
        let view = Arc::new(texture.create_view(&Default::default()));
        self.point_shadow_cache = Some((key, width, height, texture, Arc::clone(&view)));
        view
    }
}

fn upload_texture(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    width: u32,
    height: u32,
    pixels: &[u8],
    label: &str,
) -> wgpu::TextureView {
    let source = TextureAsset {
        name: label.to_owned(),
        width,
        height,
        pixels: pixels.to_vec(),
        cached_mips: Vec::new(),
    };
    let levels = source.mip_chain(3);
    let texture = device.create_texture(&wgpu::TextureDescriptor {
        label: Some(label),
        size: wgpu::Extent3d {
            width: width.max(1),
            height: height.max(1),
            depth_or_array_layers: 1,
        },
        mip_level_count: levels.len() as u32,
        sample_count: 1,
        dimension: wgpu::TextureDimension::D2,
        format: wgpu::TextureFormat::Rgba8UnormSrgb,
        usage: wgpu::TextureUsages::TEXTURE_BINDING | wgpu::TextureUsages::COPY_DST,
        view_formats: &[],
    });
    for (mip_level, level) in levels.iter().enumerate() {
        queue.write_texture(
            wgpu::TexelCopyTextureInfo {
                texture: &texture,
                mip_level: mip_level as u32,
                origin: wgpu::Origin3d::ZERO,
                aspect: wgpu::TextureAspect::All,
            },
            &level.pixels,
            wgpu::TexelCopyBufferLayout {
                offset: 0,
                bytes_per_row: Some(level.width * 4),
                rows_per_image: Some(level.height),
            },
            wgpu::Extent3d {
                width: level.width,
                height: level.height,
                depth_or_array_layers: 1,
            },
        );
    }
    texture.create_view(&Default::default())
}

fn batch_world_bounds(batch: &GpuBatch) -> [Vec3; 8] {
    let minimum = batch.local_bounds_min;
    let maximum = batch.local_bounds_max;
    let transform = batch.object_transform;
    std::array::from_fn(|index| {
        let local = Vec3::new(
            if index & 1 == 0 { minimum.x } else { maximum.x },
            if index & 2 == 0 { minimum.y } else { maximum.y },
            if index & 4 == 0 { minimum.z } else { maximum.z },
        );
        let scaled = Vec3::new(
            local.x * transform.scale.x,
            local.y * transform.scale.y,
            local.z * transform.scale.z,
        );
        transform.rotation.rotate(scaled) + transform.translation
    })
}

fn shadow_bounds_points(batches: &[GpuBatch]) -> Vec<Vec3> {
    batches
        .iter()
        .filter(|batch| batch.casts_shadows && !batch.vertices.is_empty())
        .flat_map(batch_world_bounds)
        .collect()
}

fn shadow_geometry_key(batches: &[GpuBatch]) -> u64 {
    batches
        .iter()
        .filter(|batch| batch.casts_shadows && !batch.vertices.is_empty())
        .fold(0xcbf2_9ce4_8422_2325_u64, |hash, batch| {
            hash.rotate_left(9)
                ^ batch.cache_key
                ^ batch.content_revision.rotate_left(23)
                ^ (batch.vertices.len() as u64).rotate_left(41)
        })
}

fn directional_metadata(
    batches: &[GpuBatch],
    resolution: u32,
    filter_radius: usize,
) -> DirectionalShadowMap {
    let direction = Vec3::new(-0.35, 0.8, 0.45).normalized();
    let forward = direction * -1.0;
    let reference_up = if forward.z.abs() < 0.95 {
        Vec3::Z
    } else {
        Vec3::Y
    };
    let right = reference_up.cross(forward).normalized();
    let up = forward.cross(right).normalized();
    let points = shadow_bounds_points(batches);
    let mut minimum = Vec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut maximum = Vec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for &point in &points {
        minimum.x = minimum.x.min(point.x);
        minimum.y = minimum.y.min(point.y);
        minimum.z = minimum.z.min(point.z);
        maximum.x = maximum.x.max(point.x);
        maximum.y = maximum.y.max(point.y);
        maximum.z = maximum.z.max(point.z);
    }
    let origin = if minimum.x.is_finite() {
        (minimum + maximum) * 0.5
    } else {
        Vec3::ZERO
    };
    let mut extent = 1.0e-3_f32;
    for &point in &points {
        let relative = point - origin;
        extent = extent
            .max(relative.dot(right).abs())
            .max(relative.dot(up).abs());
    }
    extent = (extent * 1.05).max(1.0e-3);
    let mut depth_min = f32::INFINITY;
    let mut depth_max = f32::NEG_INFINITY;
    for &point in &points {
        let depth = (point - origin).dot(forward);
        depth_min = depth_min.min(depth);
        depth_max = depth_max.max(depth);
    }
    if !depth_min.is_finite() {
        depth_min = -1.0;
        depth_max = 1.0;
    }
    let padding = ((depth_max - depth_min) * 0.01).max(1.0e-4);
    DirectionalShadowMap {
        resolution: resolution.clamp(1, 2048) as usize,
        // In GPU metadata maps these two values hold the light-space depth
        // range. The CPU renderer never receives this descriptor.
        depth: vec![depth_min - padding, depth_max + padding],
        origin,
        right,
        up,
        forward,
        extent,
        bias: (extent * 2.0 / resolution.max(1) as f32).max(1.0e-5) * 1.5,
        filter_radius,
    }
}

fn point_metadata(lights: &[ViewportLight], filter_radius: usize) -> PointShadowAtlas {
    let max_resolution = lights
        .iter()
        .take(MAX_VIEWPORT_LIGHTS)
        .map(|light| light.shadow_resolution as usize)
        .max()
        .unwrap_or(0)
        .max(1);
    let height = lights
        .iter()
        .take(MAX_VIEWPORT_LIGHTS)
        .map(|light| light.shadow_resolution as usize)
        .sum::<usize>()
        .max(1);
    let mut regions = [crate::PointShadowRegion::default(); MAX_VIEWPORT_LIGHTS];
    let mut row = 0;
    for (index, light) in lights.iter().take(MAX_VIEWPORT_LIGHTS).enumerate() {
        let resolution = light.shadow_resolution as usize;
        if resolution > 0 {
            regions[index] = crate::PointShadowRegion {
                row,
                resolution,
                bias: (1.0 / resolution as f32).max(1.0e-5) * 2.0,
                filter_radius,
            };
            row += resolution;
        }
    }
    PointShadowAtlas {
        width: max_resolution * 6,
        height,
        depth: Vec::new(),
        regions,
    }
}

fn point_shadow_far_depth(batches: &[GpuBatch], lights: &[ViewportLight]) -> f32 {
    let mut far = 1.0_f32;
    for point in shadow_bounds_points(batches) {
        for light in lights.iter().take(MAX_VIEWPORT_LIGHTS) {
            far = far.max((point - light.position).length());
        }
    }
    far * 1.01
}

#[cfg(test)]
mod tests {
    use super::*;

    fn dynamic_offset(entry: wgpu::BindGroupLayoutEntry) -> bool {
        match entry.ty {
            wgpu::BindingType::Buffer {
                has_dynamic_offset, ..
            } => has_dynamic_offset,
            _ => panic!("expected a uniform-buffer layout entry"),
        }
    }

    #[test]
    fn only_shadow_view_uniforms_use_dynamic_offsets() {
        assert!(!dynamic_offset(camera_uniform_layout_entry()));
        assert!(dynamic_offset(shadow_uniform_layout_entry()));
        assert_eq!(SHADOW_UNIFORM_STRIDE % 256, 0);
    }

    #[test]
    fn point_shadow_atlas_rows_map_to_distinct_top_origin_charts() {
        let lights = [
            ViewportLight {
                position: Vec3::ZERO,
                color: [1.0; 3],
                intensity: 1.0,
                radius: 0.0,
                shadow_resolution: 64,
            },
            ViewportLight {
                position: Vec3::X,
                color: [1.0; 3],
                intensity: 1.0,
                radius: 0.0,
                shadow_resolution: 32,
            },
        ];
        let atlas = point_metadata(&lights, 1);
        assert_eq!((atlas.width, atlas.height), (384, 96));
        assert_eq!(atlas.regions[0].row, 0);
        assert_eq!(atlas.regions[1].row, 64);

        let chart_bounds = |region: crate::PointShadowRegion, face: usize| {
            let width = atlas.width as f32;
            let height = atlas.height as f32;
            let scale_x = region.resolution as f32 / width;
            let scale_y = region.resolution as f32 / height;
            let center_x = 2.0 * ((face as f32 + 0.5) * region.resolution as f32) / width - 1.0;
            let center_y =
                1.0 - 2.0 * (region.row as f32 + 0.5 * region.resolution as f32) / height;
            [
                (center_x - scale_x + 1.0) * 0.5 * width,
                (1.0 - center_y - scale_y) * 0.5 * height,
                (center_x + scale_x + 1.0) * 0.5 * width,
                (1.0 - center_y + scale_y) * 0.5 * height,
            ]
        };
        let assert_bounds = |actual: [f32; 4], expected: [f32; 4]| {
            for (actual, expected) in actual.into_iter().zip(expected) {
                assert!((actual - expected).abs() < 1.0e-3, "{actual} != {expected}");
            }
        };
        assert_bounds(chart_bounds(atlas.regions[0], 0), [0.0, 0.0, 64.0, 64.0]);
        assert_bounds(chart_bounds(atlas.regions[0], 5), [320.0, 0.0, 384.0, 64.0]);
        assert_bounds(chart_bounds(atlas.regions[1], 0), [0.0, 64.0, 32.0, 96.0]);
        assert_bounds(
            chart_bounds(atlas.regions[1], 5),
            [160.0, 64.0, 192.0, 96.0],
        );
    }

    #[test]
    fn shadow_geometry_key_ignores_material_handles_but_tracks_transforms() {
        let vertex = GpuVertex {
            position: [0.0, 0.0, 0.0, 1.0],
            normal: [0.0; 4],
            uv_color_rg: [0.0; 4],
            color_ba_base_rg: [0.0; 4],
            base_ba_material: [0.0; 4],
            object_translation: [0.0; 4],
            object_rotation: [0.0, 0.0, 0.0, 1.0],
            object_scale: [1.0, 1.0, 1.0, 0.0],
        };
        let mut batch = GpuBatch {
            cache_key: 7,
            content_revision: 11,
            object_id: zerofps_core::NodeId {
                slot: 1,
                generation: 0,
            },
            object_transform: Transform::IDENTITY,
            local_bounds_min: Vec3::ZERO,
            local_bounds_max: Vec3::ZERO,
            casts_shadows: true,
            texture_cache_key: 13,
            texture: None,
            gpu_texture: None,
            transparent: false,
            vertices: vec![vertex],
        };
        let original = shadow_geometry_key(std::slice::from_ref(&batch));
        batch.texture_cache_key = 99;
        assert_eq!(original, shadow_geometry_key(std::slice::from_ref(&batch)));
        batch.content_revision += 1;
        assert_ne!(original, shadow_geometry_key(std::slice::from_ref(&batch)));
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn renders_a_depth_tested_triangle() {
        let mut renderer = VulkanViewport::new().expect("Vulkan viewport should initialize");
        let vertex = |x, z, color: [f32; 4]| GpuVertex {
            position: [x, 0.0, z, 1.0],
            normal: [0.0, -1.0, 0.0, 0.0],
            uv_color_rg: [0.0, 0.0, color[0], color[1]],
            color_ba_base_rg: [color[2], color[3], 1.0, 1.0],
            base_ba_material: [1.0, 1.0, 0.0, 0.0],
            object_translation: [0.0, 0.0, 0.0, 0.0],
            object_rotation: [0.0, 0.0, 0.0, 1.0],
            object_scale: [1.0, 1.0, 1.0, 0.0],
        };
        let texture = Arc::new(TextureAsset {
            name: "mip smoke texture".into(),
            width: 4,
            height: 4,
            pixels: vec![255; 4 * 4 * 4],
            cached_mips: Vec::new(),
        });
        let batches = [GpuBatch {
            cache_key: 1,
            content_revision: 1,
            object_id: zerofps_core::NodeId {
                slot: 0,
                generation: 0,
            },
            object_transform: Transform::IDENTITY,
            local_bounds_min: Vec3::new(-1.0, 0.0, -1.0),
            local_bounds_max: Vec3::new(1.0, 0.0, 1.0),
            casts_shadows: true,
            texture_cache_key: 1,
            texture: Some(texture),
            gpu_texture: None,
            transparent: false,
            vertices: vec![
                vertex(-1.0, -1.0, [1.0; 4]),
                vertex(1.0, -1.0, [1.0; 4]),
                vertex(0.0, 1.0, [1.0; 4]),
            ],
        }];
        let lights = [ViewportLight {
            position: Vec3::new(0.0, -2.0, 2.0),
            color: [1.0; 3],
            intensity: 100.0,
            radius: 0.1,
            shadow_resolution: 32,
        }];
        let color = renderer
            .render_resident(
                Vec2::new(128.0, 128.0),
                (0.0, 0.0, 0.0, 1.0, Vec3::ZERO, 1.0, 1),
                &batches,
                true,
                &lights,
                None,
                None,
                Some(1),
                1,
                32,
                1,
                6 * 1024 * 1024 * 1024,
            )
            .unwrap();
        let pixels = color.readback_rgba8().unwrap();
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[3] > 0));
        let lit_luminance = pixels
            .chunks_exact(4)
            .filter(|pixel| pixel[3] > 0)
            .map(|pixel| u64::from(pixel[0]) + u64::from(pixel[1]) + u64::from(pixel[2]))
            .sum::<u64>();
        let opposite_lights = [ViewportLight {
            position: Vec3::new(0.0, 2.0, 2.0),
            ..lights[0]
        }];
        let opposite = renderer
            .render_resident(
                Vec2::new(128.0, 128.0),
                (0.0, 0.0, 0.0, 1.0, Vec3::ZERO, 1.0, 1),
                &batches,
                true,
                &opposite_lights,
                None,
                None,
                Some(3),
                1,
                32,
                1,
                6 * 1024 * 1024 * 1024,
            )
            .unwrap()
            .readback_rgba8()
            .unwrap();
        let opposite_luminance = opposite
            .chunks_exact(4)
            .filter(|pixel| pixel[3] > 0)
            .map(|pixel| u64::from(pixel[0]) + u64::from(pixel[1]) + u64::from(pixel[2]))
            .sum::<u64>();
        assert_eq!(
            renderer.last_vertex_upload_time(),
            Duration::ZERO,
            "changing only lighting must reuse resident geometry"
        );
        assert!(
            lit_luminance > opposite_luminance,
            "a point light on the normal-facing side should brighten the triangle"
        );
        renderer
            .render_resident(
                Vec2::new(128.0, 128.0),
                (0.0, 0.0, 0.0, 1.0, Vec3::ZERO, 1.0, 1),
                &batches,
                true,
                &lights,
                None,
                None,
                Some(2),
                1,
                32,
                1,
                6 * 1024 * 1024 * 1024,
            )
            .unwrap();
        // Timestamp readbacks are intentionally polled at telemetry cadence,
        // rather than on every latency-sensitive viewport frame.
        for revision in 4..10 {
            renderer
                .render_resident(
                    Vec2::new(128.0, 128.0),
                    (0.0, 0.0, 0.0, 1.0, Vec3::ZERO, 1.0, 1),
                    &batches,
                    true,
                    &lights,
                    None,
                    None,
                    Some(revision),
                    1,
                    32,
                    1,
                    6 * 1024 * 1024 * 1024,
                )
                .unwrap();
        }
        renderer
            .render_resident(
                Vec2::new(128.0, 128.0),
                (0.15, -0.05, 0.0, 1.0, Vec3::ZERO, 1.0, 1),
                &batches,
                true,
                &lights,
                None,
                None,
                Some(10_000),
                1,
                32,
                1,
                6 * 1024 * 1024 * 1024,
            )
            .unwrap();
        assert_eq!(
            renderer.last_point_shadow_time(),
            Duration::ZERO,
            "camera-only motion must reuse the resident point-shadow atlas"
        );
        assert_eq!(
            renderer.point_shadow_cpu_timings().uniform_write,
            Duration::ZERO,
            "camera-only motion must not rewrite point-shadow uniforms"
        );
        if renderer.timestamp_profiler.is_some() {
            assert!(renderer.gpu_directional_shadow_time() > Duration::ZERO);
            assert!(renderer.gpu_point_shadow_time() > Duration::ZERO);
            assert!(renderer.gpu_viewport_time() > Duration::ZERO);
        }
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn renders_scene_without_shadow_casters() {
        let mut renderer = VulkanViewport::new().expect("Vulkan viewport should initialize");
        let color = renderer
            .render_resident(
                Vec2::new(64.0, 64.0),
                (0.0, 0.0, 0.0, 1.0, Vec3::ZERO, 1.0, 1),
                &[],
                true,
                &[],
                None,
                None,
                Some(1),
                1,
                32,
                1,
                6 * 1024 * 1024 * 1024,
            )
            .expect("an empty scene must still produce a valid resident target");
        assert_eq!((color.width, color.height), (64, 64));
    }
}
