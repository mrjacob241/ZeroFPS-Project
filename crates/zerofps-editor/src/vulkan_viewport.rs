use std::{
    collections::{HashMap, HashSet},
    sync::Arc,
};

use egui::Vec2;
use wgpu::util::DeviceExt;
use zerofps_assets::TextureAsset;
use zerofps_core::Vec3;

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

pub struct GpuBatch {
    pub cache_key: u64,
    pub texture: Option<Arc<TextureAsset>>,
    pub gpu_texture: Option<Arc<GpuImage>>,
    pub vertices: Vec<GpuVertex>,
}

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct CameraUniform {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    grid_spacing: f32,
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

pub struct VulkanViewport {
    device: Arc<wgpu::Device>,
    queue: Arc<wgpu::Queue>,
    camera_layout: wgpu::BindGroupLayout,
    texture_layout: wgpu::BindGroupLayout,
    pipeline: wgpu::RenderPipeline,
    sampler: wgpu::Sampler,
    texture_cache: HashMap<usize, (Arc<TextureAsset>, Arc<wgpu::TextureView>)>,
    white: Arc<wgpu::TextureView>,
    targets: Option<CachedTargets>,
    camera_buffer: wgpu::Buffer,
    shadow_cache: Option<(usize, u32, u32, wgpu::Texture, Arc<wgpu::TextureView>)>,
    point_shadow_cache: Option<(usize, u32, u32, wgpu::Texture, Arc<wgpu::TextureView>)>,
    camera_group_cache: Option<(usize, usize, Arc<wgpu::BindGroup>)>,
    vertex_cache: HashMap<u64, (usize, Arc<wgpu::Buffer>)>,
    bind_group_cache: HashMap<usize, Arc<wgpu::BindGroup>>,
}

struct CachedTargets {
    size: [u32; 2],
    color: Arc<GpuImage>,
    _depth: wgpu::Texture,
    depth_view: Arc<wgpu::TextureView>,
}

impl VulkanViewport {
    pub fn new() -> Result<Self, String> {
        let runtime = shared_runtime()?;
        let device = Arc::clone(&runtime.device);
        let queue = Arc::clone(&runtime.queue);
        let camera_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
            label: Some("viewport camera layout"),
            entries: &[
                wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::VERTEX | wgpu::ShaderStages::FRAGMENT,
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Uniform,
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                },
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
        let pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: Some("ZeroFPS Vulkan viewport"),
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
                depth_write_enabled: true,
                depth_compare: wgpu::CompareFunction::Less,
                stencil: Default::default(),
                bias: Default::default(),
            }),
            multisample: Default::default(),
            multiview: None,
            cache: None,
        });
        let sampler = device.create_sampler(&wgpu::SamplerDescriptor {
            address_mode_u: wgpu::AddressMode::Repeat,
            address_mode_v: wgpu::AddressMode::Repeat,
            mag_filter: wgpu::FilterMode::Linear,
            min_filter: wgpu::FilterMode::Linear,
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
            sampler,
            texture_cache: HashMap::new(),
            white,
            targets: None,
            camera_buffer,
            shadow_cache: None,
            point_shadow_cache: None,
            camera_group_cache: None,
            vertex_cache: HashMap::new(),
            bind_group_cache: HashMap::new(),
        })
    }

    pub fn render_resident(
        &mut self,
        size: Vec2,
        camera: (f32, f32, f32, Vec3, f32, u32),
        batches: &[GpuBatch],
        global_light_enabled: bool,
        point_lights: &[ViewportLight],
        directional_shadow: Option<&DirectionalShadowMap>,
        point_shadows: Option<&PointShadowAtlas>,
    ) -> Result<Arc<GpuImage>, String> {
        let width = size.x.round().max(1.0) as u32;
        let height = size.y.round().max(1.0) as u32;
        self.ensure_targets(width, height);
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
            zoom: camera.2,
            grid_spacing: camera.4,
            target: [camera.3.x, camera.3.y, camera.3.z, 0.0],
            viewport: [width as f32, height as f32],
            projection: camera.5,
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
        let shadow_view = self.shadow_view(directional_shadow);
        let point_shadow_view = self.point_shadow_view(point_shadows);
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
        let mut encoder = self.device.create_command_encoder(&Default::default());
        let mut live_bind_groups = HashSet::new();
        let mut live_vertex_buffers = HashSet::new();
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
                timestamp_writes: None,
                occlusion_query_set: None,
            });
            pass.set_pipeline(&self.pipeline);
            pass.set_bind_group(0, camera_group.as_ref(), &[]);
            for batch in batches {
                if batch.vertices.is_empty() {
                    continue;
                }
                let view = batch
                    .gpu_texture
                    .as_ref()
                    .map(|image| Arc::clone(&image.view))
                    .unwrap_or_else(|| self.texture_view(batch.texture.as_ref()));
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
                let vertices = match self.vertex_cache.get(&batch.cache_key) {
                    Some((count, buffer)) if *count == batch.vertices.len() => {
                        self.queue
                            .write_buffer(buffer, 0, bytemuck::cast_slice(&batch.vertices));
                        Arc::clone(buffer)
                    }
                    _ => {
                        let buffer = Arc::new(self.device.create_buffer_init(
                            &wgpu::util::BufferInitDescriptor {
                                label: Some("persistent viewport vertices"),
                                contents: bytemuck::cast_slice(&batch.vertices),
                                usage: wgpu::BufferUsages::VERTEX | wgpu::BufferUsages::COPY_DST,
                            },
                        ));
                        self.vertex_cache
                            .insert(batch.cache_key, (batch.vertices.len(), Arc::clone(&buffer)));
                        buffer
                    }
                };
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
        self.queue.submit([encoder.finish()]);
        Ok(color_target)
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

    fn texture_view(&mut self, texture: Option<&Arc<TextureAsset>>) -> Arc<wgpu::TextureView> {
        let Some(texture) = texture else {
            return Arc::clone(&self.white);
        };
        let key = Arc::as_ptr(texture) as usize;
        self.texture_cache.entry(key).or_insert_with(|| {
            let view = upload_texture(
                &self.device,
                &self.queue,
                texture.width,
                texture.height,
                &texture.pixels,
                &texture.name,
            );
            (Arc::clone(texture), Arc::new(view))
        });
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
    let texture = device.create_texture_with_data(
        queue,
        &wgpu::TextureDescriptor {
            label: Some(label),
            size: wgpu::Extent3d {
                width: width.max(1),
                height: height.max(1),
                depth_or_array_layers: 1,
            },
            mip_level_count: 1,
            sample_count: 1,
            dimension: wgpu::TextureDimension::D2,
            format: wgpu::TextureFormat::Rgba8UnormSrgb,
            usage: wgpu::TextureUsages::TEXTURE_BINDING,
            view_formats: &[],
        },
        wgpu::util::TextureDataOrder::LayerMajor,
        pixels,
    );
    texture.create_view(&Default::default())
}

#[cfg(test)]
mod tests {
    use super::*;

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
        let batches = [GpuBatch {
            cache_key: 1,
            texture: None,
            gpu_texture: None,
            vertices: vec![
                vertex(-1.0, -1.0, [1.0; 4]),
                vertex(1.0, -1.0, [1.0; 4]),
                vertex(0.0, 1.0, [1.0; 4]),
            ],
        }];
        let color = renderer
            .render_resident(
                Vec2::new(128.0, 128.0),
                (0.0, 0.0, 1.0, Vec3::ZERO, 1.0, 1),
                &batches,
                true,
                &[],
                None,
                None,
            )
            .unwrap();
        let pixels = color.readback_rgba8().unwrap();
        assert!(pixels.chunks_exact(4).any(|pixel| pixel[3] > 0));
    }
}
