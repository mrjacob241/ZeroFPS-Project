//! ZeroFPS scene editor prototype.
//!
//! This executable intentionally keeps an editor-local authoring model.  The
//! model is a thin adapter which can later be replaced by `zerofps-core`
//! handles without coupling the UI to runtime ownership.

mod compositor_compile;
mod compositor_cpu;
mod compositor_graph;
mod dynamics;
mod vulkan_compositor;
mod vulkan_runtime;
mod vulkan_viewport;

use std::{
    collections::{BTreeMap, BTreeSet, HashMap, HashSet, VecDeque},
    path::PathBuf,
    sync::{Arc, Condvar, Mutex, mpsc},
    time::{Duration, Instant},
};

use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, FontId, Id, Key, Layout, Pos2, Rect, RichText, Sense,
    Stroke, TextureHandle, TextureId, TextureOptions, Vec2,
};
use zerofps_assets::{
    MeshAsset, MeshAutofixReport, Primitive, TextureAsset, Vertex, autofix_mesh, import_file,
    load_texture_mip_cache, mip_cache_path, prepare_texture_mips, save_texture_mip_cache,
};
use zerofps_core::{
    Attribute, AttributeDeclaration, AttributeKey, Component, GeometryTree, NodeId, Quat,
    ReparentMode, Transform, Vec3 as CoreVec3,
};
use zerofps_formats::{BundleAsset, ProjectFile, load_zfp, save_zfp};

use crate::compositor_graph::{CpuGraphExecutor, GraphExecutor};
use crate::dynamics::{ClothSettings, ClothState, MeshScalarField, PaintMode, WindField, heatmap};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
        renderer: eframe::Renderer::Wgpu,
        viewport: egui::ViewportBuilder::default()
            .with_title("ZeroFPS Project — Scene Editor")
            .with_inner_size([1440.0, 900.0])
            .with_min_inner_size([980.0, 640.0]),
        ..Default::default()
    };
    eframe::run_native(
        "ZeroFPS Project",
        options,
        Box::new(|cc| Ok(Box::new(EditorApp::new(cc)))),
    )
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ObjectKind {
    Model,
    Light,
    Camera,
    Empty,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum BuiltinPrimitive {
    Cube,
    Sphere,
    Floor,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ColliderShape {
    Sphere,
    Cylinder,
    Box,
    Flat,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
enum CylinderJoint {
    #[default]
    None,
    Wheel,
    Engine,
}

impl CylinderJoint {
    const ALL: [Self; 3] = [Self::None, Self::Wheel, Self::Engine];

    fn label(self) -> &'static str {
        match self {
            Self::None => "None",
            Self::Wheel => "Wheel joint",
            Self::Engine => "Engine joint",
        }
    }

    fn storage_name(self) -> &'static str {
        match self {
            Self::None => "none",
            Self::Wheel => "wheel",
            Self::Engine => "engine",
        }
    }

    fn from_storage(value: &str) -> Self {
        match value {
            "wheel" => Self::Wheel,
            "engine" => Self::Engine,
            _ => Self::None,
        }
    }
}

impl ColliderShape {
    const ALL: [Self; 4] = [Self::Sphere, Self::Cylinder, Self::Box, Self::Flat];

    fn label(self) -> &'static str {
        match self {
            Self::Sphere => "Sphere",
            Self::Cylinder => "Cylinder",
            Self::Box => "Box",
            Self::Flat => "Flat",
        }
    }

    fn storage_name(self) -> &'static str {
        match self {
            Self::Sphere => "sphere",
            Self::Cylinder => "cylinder",
            Self::Box => "box",
            Self::Flat => "flat",
        }
    }

    fn from_storage(value: &str) -> Self {
        match value {
            "sphere" => Self::Sphere,
            "cylinder" => Self::Cylinder,
            "flat" => Self::Flat,
            _ => Self::Box,
        }
    }
}

impl BuiltinPrimitive {
    fn label(self) -> &'static str {
        match self {
            Self::Cube => "Cube",
            Self::Sphere => "Sphere",
            Self::Floor => "Floor",
        }
    }

    fn asset_path(self) -> &'static str {
        match self {
            Self::Cube => "builtin:cube",
            Self::Sphere => "builtin:sphere",
            Self::Floor => "builtin:floor",
        }
    }
}

impl ObjectKind {
    fn icon(self) -> &'static str {
        match self {
            Self::Model => "◆",
            Self::Light => "☀",
            Self::Camera => "▣",
            Self::Empty => "◇",
        }
    }
}

struct EditorScene {
    tree: GeometryTree,
    selected: Option<NodeId>,
}

impl Default for EditorScene {
    fn default() -> Self {
        Self {
            tree: GeometryTree::new(),
            selected: None,
        }
    }
}

impl EditorScene {
    fn add(&mut self, name: &str, kind: ObjectKind, parent: Option<NodeId>) -> NodeId {
        let id = self
            .tree
            .create(name, parent)
            .expect("editor supplies valid parent");
        let component = match kind {
            ObjectKind::Model => Some(Component::Model {
                asset: "builtin:cube".into(),
                primitive: None,
            }),
            ObjectKind::Light => Some(Component::Light {
                intensity: 1.0,
                color: [1.0; 3],
                radius: 0.0,
                shadow_resolution: 256,
            }),
            ObjectKind::Camera => Some(Component::Camera {
                field_of_view_degrees: 60.0,
                projection: "perspective".into(),
                aspect_ratio: 16.0 / 9.0,
                near_clip: 0.1,
                far_clip: 1_000.0,
                orthographic_size: 10.0,
            }),
            ObjectKind::Empty => None,
        };
        if let Some(component) = component {
            self.tree.add_component(id, component).expect("fresh node");
        }
        id
    }

    fn kind(&self, id: NodeId) -> ObjectKind {
        self.tree
            .node(id)
            .ok()
            .and_then(|node| {
                node.components
                    .iter()
                    .find_map(|component| match component {
                        Component::Model { .. } => Some(ObjectKind::Model),
                        Component::Light { .. } => Some(ObjectKind::Light),
                        Component::Camera { .. } => Some(ObjectKind::Camera),
                        _ => None,
                    })
            })
            .unwrap_or(ObjectKind::Empty)
    }

    fn visible(&self, id: NodeId) -> bool {
        matches!(
            self.tree
                .resolve_attribute(id, &AttributeKey::Visible)
                .ok()
                .map(|v| v.value),
            Some(Attribute::Bool(true))
        )
    }

    fn participates_in_physics(&self, id: NodeId) -> bool {
        self.kind(id) != ObjectKind::Camera
    }

    fn remove_selected(&mut self) {
        let Some(id) = self.selected.take() else {
            return;
        };
        let _ = self.tree.remove(id);
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Tool {
    Select,
    Move,
    Rotate,
    Scale,
    FieldPaint,
    TexturePaint,
}

#[derive(Clone)]
struct PaintedMask {
    width: u32,
    height: u32,
    pixels: Vec<u8>,
    revision: u64,
}

impl PaintedMask {
    fn new(size: u32) -> Self {
        Self {
            width: size,
            height: size,
            pixels: vec![0; (size * size) as usize],
            revision: 1,
        }
    }

    fn uniform(width: u32, height: u32, value: u8) -> Self {
        Self {
            width,
            height,
            pixels: vec![value; (width * height) as usize],
            revision: 1,
        }
    }

    fn texture(&self, heatmap_preview: bool) -> TextureAsset {
        let mut rgba = Vec::with_capacity(self.pixels.len() * 4);
        for &value in &self.pixels {
            if heatmap_preview {
                let color = heatmap(value as f32 / 255.0, 0.0, 1.0);
                rgba.extend(color.map(|channel| (channel.clamp(0.0, 1.0) * 255.0) as u8));
            } else {
                rgba.extend([value, value, value, 255]);
            }
        }
        TextureAsset {
            name: "painted-mask".into(),
            width: self.width,
            height: self.height,
            pixels: rgba,
            cached_mips: Vec::new(),
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum PlayState {
    Editing,
    Building,
    Running,
    Paused,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum BottomTab {
    Assets,
    Scripts,
    Console,
    Telemetry,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum InspectorTab {
    Inspector,
    Inputs,
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum WorkspaceTab {
    Scene,
    Compositing,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum RenderDevice {
    Vulkan,
    Cpu,
}

impl RenderDevice {
    fn label(self) -> &'static str {
        match self {
            Self::Vulkan => "Vulkan",
            Self::Cpu => "CPU",
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
enum MaterialTab {
    Shader,
    Surface,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewportMode {
    Shaded,
    ShadedEdges,
    Wireframe,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ShaderMode {
    Toon,
    Diffuse,
}

impl ShaderMode {
    fn label(self) -> &'static str {
        match self {
            Self::Toon => "Toon",
            Self::Diffuse => "Diffuse",
        }
    }

    fn attribute_value(self) -> &'static str {
        match self {
            Self::Toon => "toon",
            Self::Diffuse => "diffuse",
        }
    }

    fn from_attribute(value: &str) -> Option<Self> {
        match value {
            "toon" => Some(Self::Toon),
            "diffuse" => Some(Self::Diffuse),
            _ => None,
        }
    }
}

fn material_shader_key() -> AttributeKey {
    AttributeKey::MaterialShader
}

fn material_smooth_normals_key() -> AttributeKey {
    AttributeKey::MaterialSmoothNormals
}

fn material_use_imported_optics_key() -> AttributeKey {
    AttributeKey::MaterialUseImportedOptics
}

fn material_tint_key() -> AttributeKey {
    AttributeKey::Tint
}

fn mesh_autofix_key() -> AttributeKey {
    AttributeKey::Custom("mesh.autofix".into())
}

impl ViewportMode {
    fn label(self) -> &'static str {
        match self {
            Self::Shaded => "Matte",
            Self::ShadedEdges => "Matte + Edges",
            Self::Wireframe => "Wireframe",
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ProjectionMode {
    Perspective,
    Orthographic,
}

const PERSPECTIVE_CAMERA_DISTANCE: f32 = 20.0;
const PERSPECTIVE_NEAR: f32 = 0.05;

impl ProjectionMode {
    fn label(self) -> &'static str {
        match self {
            Self::Perspective => "Perspective",
            Self::Orthographic => "Orthographic",
        }
    }
}

struct LogEntry {
    level: &'static str,
    color: Color32,
    message: String,
}

struct ImportedAsset {
    path: String,
    mesh: MeshAsset,
    autofixed_mesh: MeshAsset,
    autofix_report: MeshAutofixReport,
    bounds: ([f32; 3], [f32; 3]),
}

#[derive(Clone)]
enum NodeSettings {
    ObjectTexture {
        object_index: usize,
        channel: usize,
    },
    ImageAsset {
        path: String,
    },
    ConstantValue {
        value: f32,
        color: [f32; 3],
    },
    Remap {
        points: Vec<[f32; 2]>,
        mode: usize,
        selected: Option<usize>,
    },
    TextureMath {
        operation: usize,
        constant: f32,
    },
    Algebra {
        expression: String,
    },
    SharpThreshold {
        threshold: f32,
    },
    SmoothThreshold {
        threshold: f32,
        width: f32,
    },
    ImageFilter {
        filter: usize,
        radius: f32,
    },
    Output {
        object_index: usize,
        channel: usize,
    },
    TextureCombine {
        mode: usize,
        operation: usize,
        alpha: f32,
    },
    ColorSpaceConvert {
        from: usize,
        to: usize,
    },
    ColorDecoder,
    Grayscale {
        mode: usize,
    },
    ColorEncoder,
    ObjectHandle {
        object_index: usize,
        label: String,
        control: usize,
        value: f32,
        minimum: f32,
        maximum: f32,
        source_handle: Option<usize>,
    },
    Time {
        scale: f32,
        modulus: f32,
        live_update: bool,
    },
    Debug,
    PaintedMask {
        object_index: usize,
    },
    MassDensity {
        object_index: usize,
        base_density: f32,
        scale: f32,
        minimum_mass: f32,
        normalize: bool,
        total_mass: f32,
    },
    SpringMesh {
        object_index: usize,
        stiffness: f32,
        bend_stiffness: f32,
        damping: f32,
        iterations: usize,
    },
    ForceField {
        object_index: usize,
        formulas: [String; 3],
        scale: f32,
    },
    VelocityField {
        object_index: usize,
        formulas: [String; 3],
        scale: f32,
        blend: f32,
        mode: usize,
    },
    Simulator {
        object_index: usize,
        gravity: bool,
        time_scale: f32,
    },
    Position {
        values: [f32; 3],
    },
    Rotation {
        degrees: [f32; 3],
    },
    ObjectSimulator {
        object_index: usize,
        mass: f32,
        gravity: bool,
        linear_velocity: [f32; 3],
        angular_velocity: [f32; 3],
        linear_damping: f32,
        angular_damping: f32,
    },
    ForceOutput {
        object_index: usize,
        force: [f32; 3],
    },
    Engine {
        object_index: usize,
        throttle: f32,
        torque: f32,
        reverse: bool,
        auto_braking: bool,
        max_rpm: f32,
    },
    ObjectTransform {
        object_index: usize,
    },
    ObjectMesh {
        object_index: usize,
    },
}

impl NodeSettings {
    fn object_index(&self) -> Option<usize> {
        match self {
            Self::ObjectTexture { object_index, .. }
            | Self::Output { object_index, .. }
            | Self::ObjectHandle { object_index, .. }
            | Self::PaintedMask { object_index }
            | Self::MassDensity { object_index, .. }
            | Self::SpringMesh { object_index, .. }
            | Self::ForceField { object_index, .. }
            | Self::VelocityField { object_index, .. }
            | Self::Simulator { object_index, .. }
            | Self::ObjectSimulator { object_index, .. }
            | Self::ForceOutput { object_index, .. }
            | Self::Engine { object_index, .. }
            | Self::ObjectTransform { object_index }
            | Self::ObjectMesh { object_index } => Some(*object_index),
            _ => None,
        }
    }

    fn set_object_index(&mut self, value: usize) {
        match self {
            Self::ObjectTexture { object_index, .. }
            | Self::Output { object_index, .. }
            | Self::ObjectHandle { object_index, .. }
            | Self::PaintedMask { object_index }
            | Self::MassDensity { object_index, .. }
            | Self::SpringMesh { object_index, .. }
            | Self::ForceField { object_index, .. }
            | Self::VelocityField { object_index, .. }
            | Self::Simulator { object_index, .. }
            | Self::ObjectSimulator { object_index, .. }
            | Self::ForceOutput { object_index, .. }
            | Self::Engine { object_index, .. }
            | Self::ObjectTransform { object_index }
            | Self::ObjectMesh { object_index } => *object_index = value,
            _ => {}
        }
    }

    fn kind(&self) -> usize {
        match self {
            Self::ObjectTexture { .. } => 0,
            Self::ImageAsset { .. } => 1,
            Self::ConstantValue { .. } => 2,
            Self::Remap { .. } => 3,
            Self::TextureMath { .. } => 4,
            Self::SharpThreshold { .. } => 5,
            Self::SmoothThreshold { .. } => 6,
            Self::ImageFilter { .. } => 7,
            Self::Output { .. } => 8,
            Self::TextureCombine { .. } => 9,
            Self::ColorSpaceConvert { .. } => 10,
            Self::ColorDecoder => 11,
            Self::Grayscale { .. } => 12,
            Self::ColorEncoder => 13,
            Self::ObjectHandle { .. } => 14,
            Self::Time { .. } => 15,
            Self::Debug => 16,
            Self::ObjectTransform { .. } => 17,
            Self::ObjectMesh { .. } => 18,
            Self::PaintedMask { .. } => 19,
            Self::MassDensity { .. } => 20,
            Self::SpringMesh { .. } => 21,
            Self::ForceField { .. } => 22,
            Self::VelocityField { .. } => 23,
            Self::Simulator { .. } => 24,
            Self::Position { .. } => 25,
            Self::Rotation { .. } => 26,
            Self::ObjectSimulator { .. } => 27,
            Self::ForceOutput { .. } => 28,
            Self::Engine { .. } => 29,
            Self::Algebra { .. } => 30,
        }
    }

    fn default_for_kind(kind: usize) -> Option<Self> {
        Some(match kind {
            0 => Self::ObjectTexture {
                object_index: 0,
                channel: 0,
            },
            1 => Self::ImageAsset {
                path: String::new(),
            },
            2 => Self::ConstantValue {
                value: 0.5,
                color: [0.5, 0.5, 0.5],
            },
            3 => Self::Remap {
                points: vec![[0.0, 0.0], [0.33, 0.33], [0.67, 0.67], [1.0, 1.0]],
                mode: 0,
                selected: None,
            },
            4 => Self::TextureMath {
                operation: 2,
                constant: 0.5,
            },
            5 => Self::SharpThreshold { threshold: 0.5 },
            6 => Self::SmoothThreshold {
                threshold: 0.5,
                width: 0.1,
            },
            7 => Self::ImageFilter {
                filter: 0,
                radius: 3.0,
            },
            8 => Self::Output {
                object_index: 0,
                channel: 0,
            },
            9 => Self::TextureCombine {
                mode: 0,
                operation: 2,
                alpha: 0.5,
            },
            10 => Self::ColorSpaceConvert { from: 0, to: 1 },
            11 => Self::ColorDecoder,
            12 => Self::Grayscale { mode: 0 },
            13 => Self::ColorEncoder,
            14 => Self::ObjectHandle {
                object_index: 0,
                label: "Compositor value".into(),
                control: 0,
                value: 0.5,
                minimum: 0.0,
                maximum: 1.0,
                source_handle: None,
            },
            15 => Self::Time {
                scale: 1.0,
                modulus: 0.0,
                live_update: true,
            },
            16 => Self::Debug,
            17 | 18 => return None,
            19 => Self::PaintedMask { object_index: 0 },
            20 => Self::MassDensity {
                object_index: 0,
                base_density: 0.15,
                scale: 1.0,
                minimum_mass: 1.0e-6,
                normalize: false,
                total_mass: 1.0,
            },
            21 => Self::SpringMesh {
                object_index: 0,
                stiffness: 0.85,
                bend_stiffness: 0.25,
                damping: 0.025,
                iterations: 7,
            },
            22 => Self::ForceField {
                object_index: 0,
                formulas: ["0".into(), "sin(t * 2)".into(), "0".into()],
                scale: 1.0,
            },
            23 => Self::VelocityField {
                object_index: 0,
                formulas: ["0".into(), "0".into(), "0".into()],
                scale: 1.0,
                blend: 1.0,
                mode: 1,
            },
            24 => Self::Simulator {
                object_index: 0,
                gravity: true,
                time_scale: 1.0,
            },
            25 => Self::Position { values: [0.0; 3] },
            26 => Self::Rotation { degrees: [0.0; 3] },
            27 => Self::ObjectSimulator {
                object_index: 0,
                mass: 1.0,
                gravity: true,
                linear_velocity: [0.0; 3],
                angular_velocity: [0.0; 3],
                linear_damping: 0.05,
                angular_damping: 0.05,
            },
            28 => return None,
            29 => Self::Engine {
                object_index: 0,
                throttle: 0.0,
                torque: 100.0,
                reverse: false,
                auto_braking: false,
                max_rpm: 6_000.0,
            },
            30 => Self::Algebra {
                expression: "x + y * z".into(),
            },
            _ => return None,
        })
    }
}

#[derive(Clone)]
struct ObjectSimulationState {
    initial_position: CoreVec3,
    initial_rotation_degrees: CoreVec3,
    position: CoreVec3,
    rotation_degrees: CoreVec3,
    linear_velocity: CoreVec3,
    angular_velocity: CoreVec3,
}

#[derive(Clone, Copy)]
struct JointSimulationState {
    base_rotation: Quat,
    angle_radians: f32,
    angular_velocity: f32,
}

struct CompositorNode {
    id: usize,
    object_index: usize,
    object_name: String,
    settings_object_name: Option<String>,
    settings: NodeSettings,
    position: Vec2,
}

enum CompositorProbeValue {
    Image(Arc<compositor_graph::FloatImage>),
    Number(f32),
    Triple([f32; 3]),
}

struct PendingCompositorGraph {
    graph: Arc<compositor_graph::CompiledGraph>,
    target: NodeId,
}

#[derive(Clone, Copy)]
struct PreviewVertex {
    source_index: usize,
    local_position: [f32; 3],
    local_normal: [f32; 3],
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

#[derive(Clone)]
struct PreviewTriangle {
    object_id: NodeId,
    object_transform: Transform,
    vertices: [PreviewVertex; 3],
    base_color: [f32; 4],
    source_base_color: [f32; 4],
    texture: Option<Arc<TextureAsset>>,
    texture_cache_key: u64,
    gpu_texture: Option<Arc<vulkan_runtime::GpuImage>>,
    source_texture: Option<Arc<TextureAsset>>,
    shader: ShaderMode,
    smooth_normals: bool,
    transmission: f32,
    ior: f32,
    casts_shadows: bool,
}

#[derive(Clone)]
struct SceneCameraPreview {
    transform: Transform,
    perspective: bool,
    field_of_view_degrees: f32,
    aspect_ratio: f32,
    orthographic_size: f32,
}

const MAX_VIEWPORT_LIGHTS: usize = 8;

#[derive(Clone, Copy, Debug)]
struct ViewportLight {
    position: CoreVec3,
    color: [f32; 3],
    intensity: f32,
    radius: f32,
    shadow_resolution: u32,
}

#[derive(Clone, Debug)]
struct ViewportLighting {
    global_enabled: bool,
    global_shadow_resolution: u32,
    shadow_filter_radius: usize,
    shadow_blur_radius: usize,
    points: Vec<ViewportLight>,
    directional_shadow: Option<Arc<DirectionalShadowMap>>,
    point_shadows: Option<Arc<PointShadowAtlas>>,
}

#[derive(Clone, Debug)]
struct DirectionalShadowMap {
    resolution: usize,
    depth: Vec<f32>,
    origin: CoreVec3,
    right: CoreVec3,
    up: CoreVec3,
    forward: CoreVec3,
    extent: f32,
    bias: f32,
    filter_radius: usize,
}

#[derive(Clone, Copy, Debug, Default)]
struct PointShadowRegion {
    row: usize,
    resolution: usize,
    bias: f32,
    filter_radius: usize,
}

#[derive(Clone, Debug)]
struct PointShadowAtlas {
    width: usize,
    height: usize,
    depth: Vec<f32>,
    regions: [PointShadowRegion; MAX_VIEWPORT_LIGHTS],
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct DirectionalShadowCacheKey {
    scene_revision: u64,
    resolution: u32,
    filter_radius: usize,
    blur_radius: usize,
}

#[derive(Clone)]
enum TextureOverride {
    Cpu(Arc<TextureAsset>),
    Gpu(Arc<vulkan_runtime::GpuImage>),
}

struct DepthFrame {
    color: FrameColor,
    linear_depth: Vec<f32>,
}

enum FrameColor {
    Cpu(ColorImage),
    Vulkan(Arc<vulkan_runtime::GpuImage>),
}

struct RenderJob {
    key: DepthCacheKey,
    viewport_size: Vec2,
    triangles: Arc<Vec<PreviewTriangle>>,
    camera: (f32, f32, f32, f32, CoreVec3, f32, ProjectionMode),
    lighting: ViewportLighting,
    show_grid: bool,
    mode: ViewportMode,
    tool: Tool,
    reusable_depth: Vec<f32>,
    device: RenderDevice,
    queued_at: Instant,
}

struct RenderResult {
    key: DepthCacheKey,
    frame: DepthFrame,
    camera: (f32, f32, f32, f32, CoreVec3, f32, ProjectionMode),
    triangles: Arc<Vec<PreviewTriangle>>,
    show_grid: bool,
    mode: ViewportMode,
    tool: Tool,
    render_time: Duration,
    prepare_time: Duration,
    shadow_prepare_time: Duration,
    resource_upload_time: Duration,
    vertex_upload_time: Duration,
    texture_upload_time: Duration,
    viewport_target_allocation_time: Duration,
    shadow_target_allocation_time: Duration,
    renderer_initialization_time: Duration,
    device: RenderDevice,
    queue_wait: Duration,
}

#[derive(Clone, Copy, Default)]
struct TimingMetric {
    latest_ms: f64,
    average_ms: f64,
    maximum_ms: f64,
    samples: u64,
}

impl TimingMetric {
    fn record(&mut self, duration: Duration) {
        let milliseconds = duration.as_secs_f64() * 1_000.0;
        self.latest_ms = milliseconds;
        self.maximum_ms = self.maximum_ms.max(milliseconds);
        self.samples = self.samples.saturating_add(1);
        self.average_ms = if self.samples == 1 {
            milliseconds
        } else {
            self.average_ms * 0.9 + milliseconds * 0.1
        };
    }
}

#[derive(Default)]
struct EditorPerformanceTelemetry {
    viewport_cpu: TimingMetric,
    viewport_vulkan: TimingMetric,
    viewport_prepare: TimingMetric,
    viewport_present: TimingMetric,
    compositor_vulkan_submit: TimingMetric,
    viewport_queue_wait: TimingMetric,
    control_to_graph_apply: TimingMetric,
    control_to_composite_ready: TimingMetric,
    control_to_present: TimingMetric,
    graph_compile: TimingMetric,
    graph_evaluation: TimingMetric,
    shadow_prepare: TimingMetric,
    viewport_resource_upload: TimingMetric,
    viewport_vertex_upload: TimingMetric,
    viewport_texture_upload: TimingMetric,
    viewport_target_allocation: TimingMetric,
    shadow_target_allocation: TimingMetric,
    viewport_initialization: TimingMetric,
}

struct DisplayWorker {
    pending: Arc<(Mutex<Option<RenderJob>>, Condvar)>,
    results: mpsc::Receiver<RenderResult>,
}

struct PresentedView {
    camera: (f32, f32, f32, f32, CoreVec3, f32, ProjectionMode),
    triangles: Arc<Vec<PreviewTriangle>>,
    show_grid: bool,
    mode: ViewportMode,
    tool: Tool,
}

#[derive(Clone, Copy, Default)]
struct InputSample {
    orbit: Vec2,
    pan: Vec2,
    zoom_log: f32,
    viewport_extent: f32,
}

#[derive(Clone, Copy, Default)]
struct RuntimeInputSnapshot {
    orbit: Vec2,
    pan: Vec2,
    zoom_log: f32,
    viewport_extent: f32,
    sequence: u64,
    captured_at: Option<Instant>,
}

struct InputWorker {
    samples: mpsc::Sender<InputSample>,
    latest: Arc<Mutex<RuntimeInputSnapshot>>,
}

struct ImportRequest {
    path: String,
    add_to_scene: bool,
    save_cache_in_file: bool,
}

struct PreparedImport {
    asset: MeshAsset,
    autofixed_mesh: MeshAsset,
    autofix_report: MeshAutofixReport,
    bounds: ([f32; 3], [f32; 3]),
    inferred_grid_spacing: Option<f32>,
}

struct ImportResult {
    path: String,
    add_to_scene: bool,
    asset: Result<PreparedImport, zerofps_assets::ImportError>,
    cache_message: Option<Result<String, String>>,
}

struct ImportProgress {
    path: String,
    phase: &'static str,
}

struct AssetImportWorker {
    requests: mpsc::Sender<ImportRequest>,
    results: mpsc::Receiver<ImportResult>,
    progress: mpsc::Receiver<ImportProgress>,
}

impl AssetImportWorker {
    fn new(ctx: egui::Context) -> Self {
        let (request_sender, request_receiver) = mpsc::channel::<ImportRequest>();
        let (result_sender, results) = mpsc::channel();
        let (progress_sender, progress) = mpsc::channel();
        std::thread::Builder::new()
            .name("zerofps-asset-import".into())
            .spawn(move || {
                while let Ok(request) = request_receiver.recv() {
                    let report_progress = |phase| {
                        let _ = progress_sender.send(ImportProgress {
                            path: request.path.clone(),
                            phase,
                        });
                        ctx.request_repaint();
                    };
                    report_progress("Decoding model and source textures");
                    let mut asset = import_file(&request.path);
                    let cache_message = if request.save_cache_in_file {
                        report_progress("Loading processed texture cache");
                        asset.as_mut().ok().map(|asset| {
                            match load_texture_mip_cache(&request.path, asset) {
                                Ok(true) => Ok(format!(
                                    "Loaded texture cache `{}`",
                                    mip_cache_path(&request.path).display()
                                )),
                                Ok(false) => {
                                    prepare_texture_mips(asset);
                                    save_texture_mip_cache(&request.path, asset)
                                        .map(|path| {
                                            format!("Created texture cache `{}`", path.display())
                                        })
                                        .map_err(|error| error.to_string())
                                }
                                Err(error) => {
                                    prepare_texture_mips(asset);
                                    save_texture_mip_cache(&request.path, asset)
                                        .map(|path| {
                                            format!(
                                                "Rebuilt texture cache `{}` after {error}",
                                                path.display()
                                            )
                                        })
                                        .map_err(|write_error| {
                                            format!(
                                                "could not read cache ({error}) or rebuild it ({write_error})"
                                            )
                                        })
                                }
                            }
                        })
                    } else {
                        None
                    };
                    if asset.is_ok() {
                        report_progress("Repairing and validating mesh");
                    }
                    let asset = asset.map(|asset| {
                        let bounds = mesh_bounds(&asset);
                        let inferred_grid_spacing = infer_grid_spacing(&asset);
                        let (autofixed_mesh, autofix_report) = autofix_mesh(&asset);
                        PreparedImport {
                            asset,
                            autofixed_mesh,
                            autofix_report,
                            bounds,
                            inferred_grid_spacing,
                        }
                    });
                    report_progress("Finalizing scene resources");
                    let result = ImportResult {
                        asset,
                        path: request.path,
                        add_to_scene: request.add_to_scene,
                        cache_message,
                    };
                    if result_sender.send(result).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            })
            .expect("asset import worker thread should start");
        Self {
            requests: request_sender,
            results,
            progress,
        }
    }

    fn submit(&self, path: String, add_to_scene: bool, save_cache_in_file: bool) {
        let _ = self.requests.send(ImportRequest {
            path,
            add_to_scene,
            save_cache_in_file,
        });
    }
}

impl InputWorker {
    fn new(_ctx: egui::Context) -> Self {
        let (sample_sender, sample_receiver) = mpsc::channel::<InputSample>();
        let latest = Arc::new(Mutex::new(RuntimeInputSnapshot::default()));
        let worker_latest = Arc::clone(&latest);
        std::thread::Builder::new()
            .name("zerofps-input".into())
            .spawn(move || {
                while let Ok(first) = sample_receiver.recv() {
                    let mut processed = RuntimeInputSnapshot {
                        orbit: first.orbit,
                        pan: first.pan,
                        zoom_log: first.zoom_log,
                        viewport_extent: first.viewport_extent,
                        sequence: 0,
                        captured_at: Some(Instant::now()),
                    };
                    for sample in sample_receiver.try_iter() {
                        processed.orbit += sample.orbit;
                        processed.pan += sample.pan;
                        processed.zoom_log += sample.zoom_log;
                        processed.viewport_extent = sample.viewport_extent;
                        processed.captured_at = Some(Instant::now());
                    }
                    let mut latest = worker_latest
                        .lock()
                        .unwrap_or_else(|poisoned| poisoned.into_inner());
                    processed.sequence = latest.sequence.wrapping_add(1);
                    *latest = processed;
                }
            })
            .expect("input worker thread should start");
        Self {
            samples: sample_sender,
            latest,
        }
    }

    fn submit(&self, sample: InputSample) {
        let _ = self.samples.send(sample);
    }

    /// Samples the latest coalesced input without consuming it. The compiled-game
    /// runtime will read this once per simulation tick.
    #[allow(dead_code)]
    fn snapshot(&self) -> RuntimeInputSnapshot {
        *self
            .latest
            .lock()
            .unwrap_or_else(|poisoned| poisoned.into_inner())
    }
}

impl DisplayWorker {
    fn new(ctx: egui::Context) -> Self {
        let pending = Arc::new((Mutex::new(None::<RenderJob>), Condvar::new()));
        let worker_pending = Arc::clone(&pending);
        let (result_sender, results) = mpsc::channel();
        std::thread::Builder::new()
            .name("zerofps-display".into())
            .spawn(move || {
                let mut workspace = RasterWorkspace::default();
                let mut vulkan: Option<vulkan_viewport::VulkanViewport> = None;
                let mut vulkan_unavailable = false;
                let mut directional_shadow_cache: Option<(
                    DirectionalShadowCacheKey,
                    Arc<DirectionalShadowMap>,
                )> = None;
                let mut point_shadow_cache: Option<(u64, Arc<PointShadowAtlas>)> = None;
                let mut vulkan_batch_cache: Option<(u64, Arc<Vec<vulkan_viewport::GpuBatch>>)> =
                    None;
                loop {
                    let job = {
                        let (lock, ready) = &*worker_pending;
                        let guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        let mut guard = ready
                            .wait_while(guard, |job| job.is_none())
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        guard.take().expect("display job became available")
                    };
                    let queue_wait = job.queued_at.elapsed();
                    let mut job = job;
                    let shadow_prepare_started = Instant::now();
                    // Rendering policy: preserve temporal continuity. People tolerate reduced
                    // spatial detail much better than flicker, uneven frame pacing, or shadow
                    // updates that visibly lag behind motion. Reuse an exact cached map, but
                    // never skip a changed map merely to meet a frame budget.
                    let gpu_shadow_pipeline =
                        job.device == RenderDevice::Vulkan && !vulkan_unavailable;
                    if !gpu_shadow_pipeline
                        && job.lighting.global_enabled
                        && job.lighting.global_shadow_resolution > 0
                    {
                        let key = DirectionalShadowCacheKey {
                            scene_revision: job.key.scene_revision,
                            resolution: job.lighting.global_shadow_resolution,
                            filter_radius: job.lighting.shadow_filter_radius,
                            blur_radius: job.lighting.shadow_blur_radius,
                        };
                        let shadow = match directional_shadow_cache.as_ref() {
                            Some((cached_key, shadow)) if *cached_key == key => Arc::clone(shadow),
                            _ => {
                                let shadow = Arc::new(build_directional_shadow_map_with_blur(
                                    &job.triangles,
                                    key.resolution as usize,
                                    key.filter_radius,
                                    key.blur_radius,
                                ));
                                directional_shadow_cache = Some((key, Arc::clone(&shadow)));
                                shadow
                            }
                        };
                        job.lighting.directional_shadow = Some(shadow);
                    } else {
                        job.lighting.directional_shadow = None;
                    }
                    let point_shadow_key = job.lighting.points.iter().fold(
                        job.key.scene_revision
                            ^ (job.lighting.shadow_filter_radius as u64).rotate_left(17),
                        |key, light| {
                            key.rotate_left(7)
                                ^ light.shadow_resolution as u64
                                ^ (light.position.x.to_bits() as u64).rotate_left(11)
                                ^ (light.position.y.to_bits() as u64).rotate_left(23)
                                ^ (light.position.z.to_bits() as u64).rotate_left(37)
                        },
                    );
                    let point_shadow_key =
                        point_shadow_key ^ (job.lighting.shadow_blur_radius as u64).rotate_left(29);
                    if !gpu_shadow_pipeline
                        && job
                            .lighting
                            .points
                            .iter()
                            .any(|light| light.shadow_resolution > 0)
                    {
                        let atlas = match point_shadow_cache.as_ref() {
                            Some((cached_key, atlas)) if *cached_key == point_shadow_key => {
                                Arc::clone(atlas)
                            }
                            _ => {
                                let atlas = Arc::new(build_point_shadow_atlas_with_blur(
                                    &job.triangles,
                                    &job.lighting.points,
                                    job.lighting.shadow_filter_radius,
                                    job.lighting.shadow_blur_radius,
                                ));
                                point_shadow_cache = Some((point_shadow_key, Arc::clone(&atlas)));
                                atlas
                            }
                        };
                        job.lighting.point_shadows = Some(atlas);
                    } else {
                        job.lighting.point_shadows = None;
                    }
                    let mut shadow_prepare_time = shadow_prepare_started.elapsed();
                    let mut resource_upload_time = Duration::ZERO;
                    let mut vertex_upload_time = Duration::ZERO;
                    let mut texture_upload_time = Duration::ZERO;
                    let mut viewport_target_allocation_time = Duration::ZERO;
                    let mut shadow_target_allocation_time = Duration::ZERO;
                    let mut renderer_initialization_time = Duration::ZERO;
                    let render_started = Instant::now();
                    let mut prepare_time = Duration::ZERO;
                    let frame = if job.mode == ViewportMode::Wireframe {
                        let width = job.viewport_size.x.round().max(1.0) as usize;
                        let height = job.viewport_size.y.round().max(1.0) as usize;
                        let mut linear_depth = job.reusable_depth;
                        linear_depth.resize(width * height, f32::INFINITY);
                        linear_depth.fill(f32::INFINITY);
                        DepthFrame {
                            color: FrameColor::Cpu(ColorImage::new(
                                [width, height],
                                Color32::TRANSPARENT,
                            )),
                            linear_depth,
                        }
                    } else if job.device == RenderDevice::Vulkan && !vulkan_unavailable {
                        if vulkan.is_none() {
                            let initialization_started = Instant::now();
                            match vulkan_viewport::VulkanViewport::new() {
                                Ok(renderer) => vulkan = Some(renderer),
                                Err(_) => vulkan_unavailable = true,
                            }
                            renderer_initialization_time = initialization_started.elapsed();
                        }
                        let rendered = vulkan.as_mut().and_then(|renderer| {
                            let prepare_started = Instant::now();
                            // A compositor result can change the packed material
                            // vertices without changing geometry/scene state.
                            // Repack when scene or texture contents change, but
                            // keep stable per-batch GPU buffer identities.
                            let batch_revision =
                                job.key.scene_revision ^ job.key.texture_revision.rotate_left(29);
                            let batches = match vulkan_batch_cache.as_ref() {
                                Some((cached_revision, batches))
                                    if *cached_revision == batch_revision =>
                                {
                                    Arc::clone(batches)
                                }
                                _ => {
                                    let batches = Arc::new(build_vulkan_batches(&job.triangles));
                                    vulkan_batch_cache =
                                        Some((batch_revision, Arc::clone(&batches)));
                                    batches
                                }
                            };
                            prepare_time = prepare_started.elapsed();
                            let projection = match job.camera.6 {
                                ProjectionMode::Perspective => 0,
                                ProjectionMode::Orthographic => 1,
                            };
                            let rendered = renderer
                                .render_resident(
                                    job.viewport_size,
                                    (
                                        job.camera.0,
                                        job.camera.1,
                                        job.camera.2,
                                        job.camera.3,
                                        job.camera.4,
                                        job.camera.5,
                                        projection,
                                    ),
                                    &batches,
                                    job.lighting.global_enabled,
                                    &job.lighting.points,
                                    job.lighting.directional_shadow.as_deref(),
                                    job.lighting.point_shadows.as_deref(),
                                    Some(job.key.scene_revision),
                                    batch_revision,
                                    job.lighting.global_shadow_resolution,
                                    job.lighting.shadow_filter_radius,
                                )
                                .ok();
                            shadow_prepare_time += renderer.last_shadow_encode_time();
                            resource_upload_time += renderer.last_resource_upload_time();
                            vertex_upload_time += renderer.last_vertex_upload_time();
                            texture_upload_time += renderer.last_texture_upload_time();
                            viewport_target_allocation_time +=
                                renderer.last_viewport_target_allocation_time();
                            shadow_target_allocation_time +=
                                renderer.last_shadow_target_allocation_time();
                            rendered
                        });
                        if let Some(color) = rendered {
                            DepthFrame {
                                color: FrameColor::Vulkan(color),
                                linear_depth: Vec::new(),
                            }
                        } else {
                            vulkan_unavailable = true;
                            rasterize_depth_frame(
                                job.viewport_size,
                                &job.triangles,
                                job.camera,
                                &job.lighting,
                                job.reusable_depth,
                                &mut workspace,
                            )
                        }
                    } else {
                        rasterize_depth_frame(
                            job.viewport_size,
                            &job.triangles,
                            job.camera,
                            &job.lighting,
                            job.reusable_depth,
                            &mut workspace,
                        )
                    };
                    if result_sender
                        .send(RenderResult {
                            key: job.key,
                            frame,
                            camera: job.camera,
                            triangles: job.triangles,
                            show_grid: job.show_grid,
                            mode: job.mode,
                            tool: job.tool,
                            render_time: render_started.elapsed(),
                            prepare_time,
                            shadow_prepare_time,
                            resource_upload_time,
                            vertex_upload_time,
                            texture_upload_time,
                            viewport_target_allocation_time,
                            shadow_target_allocation_time,
                            renderer_initialization_time,
                            device: job.device,
                            queue_wait,
                        })
                        .is_err()
                    {
                        break;
                    }
                    ctx.request_repaint();
                }
            })
            .expect("display worker thread should start");
        Self { pending, results }
    }

    fn submit_latest(&self, job: RenderJob) {
        let (lock, ready) = &*self.pending;
        *lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner()) = Some(job);
        ready.notify_one();
    }
}

fn build_vulkan_batches(triangles: &[PreviewTriangle]) -> Vec<vulkan_viewport::GpuBatch> {
    let mut groups: HashMap<(usize, bool), vulkan_viewport::GpuBatch> = HashMap::new();
    for triangle in triangles {
        let key = triangle
            .gpu_texture
            .as_ref()
            .map(|texture| Arc::as_ptr(texture) as usize)
            .or_else(|| {
                triangle
                    .texture
                    .as_ref()
                    .map(|texture| Arc::as_ptr(texture) as usize)
            })
            .unwrap_or(0);
        let batch = groups
            .entry((key, triangle.casts_shadows))
            .or_insert_with(|| vulkan_viewport::GpuBatch {
                cache_key: triangle
                    .source_texture
                    .as_ref()
                    .map(|texture| Arc::as_ptr(texture) as u64)
                    .unwrap_or(0)
                    ^ 0x9e37_79b9_7f4a_7c15,
                casts_shadows: triangle.casts_shadows,
                texture_cache_key: triangle.texture_cache_key,
                texture: triangle.texture.clone(),
                gpu_texture: triangle.gpu_texture.clone(),
                vertices: Vec::new(),
            });
        batch.cache_key = batch.cache_key.rotate_left(7)
            ^ triangle.object_id.slot as u64
            ^ ((triangle.object_id.generation as u64) << 32)
            ^ ((triangle.vertices[0].uv[0].to_bits() as u64) << 1)
            ^ ((triangle.vertices[0].uv[1].to_bits() as u64) << 33);
        let local_positions = triangle.vertices.map(|vertex| {
            CoreVec3::new(
                vertex.local_position[0],
                vertex.local_position[1],
                vertex.local_position[2],
            )
        });
        let face_normal = (local_positions[1] - local_positions[0])
            .cross(local_positions[2] - local_positions[0])
            .normalized();
        for vertex in triangle.vertices {
            let normal = if triangle.smooth_normals {
                vertex.local_normal
            } else {
                [face_normal.x, face_normal.y, face_normal.z]
            };
            let transform = triangle.object_transform;
            batch.vertices.push(vulkan_viewport::GpuVertex {
                position: [
                    vertex.local_position[0],
                    vertex.local_position[1],
                    vertex.local_position[2],
                    1.0,
                ],
                normal: [normal[0], normal[1], normal[2], 0.0],
                uv_color_rg: [vertex.uv[0], vertex.uv[1], vertex.color[0], vertex.color[1]],
                color_ba_base_rg: [
                    vertex.color[2],
                    vertex.color[3],
                    triangle.base_color[0],
                    triangle.base_color[1],
                ],
                base_ba_material: [
                    triangle.base_color[2],
                    triangle.base_color[3],
                    f32::from(triangle.shader == ShaderMode::Toon),
                    triangle.transmission,
                ],
                object_translation: [
                    transform.translation.x,
                    transform.translation.y,
                    transform.translation.z,
                    triangle.object_id.slot as f32 + 1.0,
                ],
                object_rotation: [
                    transform.rotation.x,
                    transform.rotation.y,
                    transform.rotation.z,
                    transform.rotation.w,
                ],
                object_scale: [
                    transform.scale.x,
                    transform.scale.y,
                    transform.scale.z,
                    f32::from(
                        triangle
                            .gpu_texture
                            .as_ref()
                            .is_some_and(|image| image.encoded_srgb),
                    ),
                ],
            });
        }
    }
    groups.into_values().collect()
}

#[derive(Default)]
struct RasterWorkspace {
    prepared: Vec<PreparedRasterTriangle>,
    bands: Vec<Vec<PreparedRasterTriangle>>,
    clipped: Vec<PreviewVertex>,
    projected: Vec<RasterVertex>,
    texture_mips: HashMap<usize, (Arc<TextureAsset>, Arc<Vec<TextureAsset>>)>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DepthCacheKey {
    size: [usize; 2],
    yaw: f32,
    pitch: f32,
    roll: f32,
    zoom: f32,
    target: CoreVec3,
    grid_spacing: f32,
    projection: ProjectionMode,
    scene_revision: u64,
    texture_revision: u64,
    global_light_enabled: bool,
    global_shadow_resolution: u32,
    shadow_quality: usize,
    shadow_blur_radius: usize,
    show_grid: bool,
    mode: ViewportMode,
    tool: Tool,
    device: RenderDevice,
}

struct EditorApp {
    scene: EditorScene,
    workspace_tab: WorkspaceTab,
    active_tool: Tool,
    play_state: PlayState,
    build_started: Option<Instant>,
    bottom_tab: BottomTab,
    inspector_tab: InspectorTab,
    material_tab: MaterialTab,
    compositor_pan: Vec2,
    compositor_zoom: f32,
    compositor_selected_node: usize,
    compositor_nodes: Vec<CompositorNode>,
    compositor_next_id: usize,
    compositor_image_dialog_target: Option<usize>,
    compositor_links: Vec<(usize, usize, usize, usize)>,
    compositor_dragging_node: Option<(usize, Vec2)>,
    compositor_pending_output: Option<(usize, usize)>,
    compositor_pending_spawn: Option<usize>,
    compositor_texture_overrides: Vec<(NodeId, TextureOverride)>,
    compositor_eval_cache: HashMap<(usize, usize, u32), Arc<TextureAsset>>,
    compositor_gpu_cache: HashMap<(usize, usize, u32), Arc<vulkan_runtime::GpuImage>>,
    compositor_image_cache: HashMap<String, Arc<TextureAsset>>,
    compositor_source_cache: HashMap<String, Arc<TextureAsset>>,
    compositor_debug_textures: HashMap<usize, TextureHandle>,
    compositor_apply_due: Option<Instant>,
    compositor_control_started: Option<Instant>,
    compositor_present_revision: Option<(u64, Instant)>,
    compositor_lod_max_dimension: u32,
    render_device: RenderDevice,
    performance: EditorPerformanceTelemetry,
    cpu_compositor: compositor_cpu::CpuGraphWorker,
    vulkan_compositor: Option<vulkan_compositor::VulkanGraphWorker>,
    vulkan_compositor_attempted: bool,
    compositor_next_generation: u64,
    vulkan_latest_generation: u64,
    vulkan_waiting_generation: Option<u64>,
    compositor_pending_target: Option<NodeId>,
    compositor_graph_queue: VecDeque<PendingCompositorGraph>,
    compositor_clock_started: Instant,
    compositor_next_time_tick: Instant,
    dynamics_fields: HashMap<NodeId, MeshScalarField>,
    dynamics_cloth: HashMap<NodeId, ClothState>,
    object_simulation_states: HashMap<usize, ObjectSimulationState>,
    joint_simulation_states: HashMap<NodeId, JointSimulationState>,
    dynamics_enabled: BTreeSet<NodeId>,
    dynamics_running: bool,
    dynamics_single_step: bool,
    dynamics_time: f32,
    dynamics_accumulator: f32,
    dynamics_last_tick: Instant,
    dynamics_wind: WindField,
    dynamics_settings: ClothSettings,
    dynamics_pan: Vec2,
    dynamics_zoom: f32,
    dynamics_node_positions: [Vec2; 6],
    dynamics_dragging_node: Option<(usize, Vec2)>,
    paint_mode: PaintMode,
    paint_value: f32,
    paint_strength: f32,
    paint_radius_pixels: f32,
    field_undo: Vec<(NodeId, Vec<f32>)>,
    field_redo: Vec<(NodeId, Vec<f32>)>,
    field_stroke_before: Option<(NodeId, Vec<f32>)>,
    painted_masks: HashMap<NodeId, PaintedMask>,
    texture_paint_undo: Vec<(NodeId, Vec<u8>)>,
    texture_paint_redo: Vec<(NodeId, Vec<u8>)>,
    texture_paint_stroke_before: Option<(NodeId, Vec<u8>)>,
    texture_paint_last_uv: Option<[f32; 2]>,
    texture_paint_heatmap: bool,
    advanced: bool,
    global_light_enabled: bool,
    camera_preview_visible: bool,
    global_shadow_resolution: u32,
    shadow_quality: usize,
    shadow_blur_radius: usize,
    target_fps: u32,
    next_viewport_frame: Instant,
    save_cache_in_file: bool,
    show_grid: bool,
    grid_spacing: f32,
    snap: bool,
    viewport_focused: bool,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_zoom: f32,
    camera_target: CoreVec3,
    hierarchy_filter: String,
    hierarchy_drag_candidate: Option<(NodeId, Instant)>,
    logs: Vec<LogEntry>,
    project_path: PathBuf,
    project_has_destination: bool,
    project_dirty: bool,
    project_error_dialog: Option<(String, String)>,
    undo_stack: Vec<GeometryTree>,
    redo_stack: Vec<GeometryTree>,
    asset_import_path: String,
    asset_loading: BTreeMap<String, &'static str>,
    asset_loading_present_revision: BTreeMap<String, u64>,
    imported_assets: Vec<ImportedAsset>,
    viewport_mode: ViewportMode,
    projection_mode: ProjectionMode,
    viewport_color: Option<TextureHandle>,
    viewport_native_texture: Option<TextureId>,
    viewport_native_view: Option<usize>,
    wgpu_render_state: Option<egui_wgpu::RenderState>,
    viewport_depth: Vec<f32>,
    viewport_depth_key: Option<DepthCacheKey>,
    scene_revision: u64,
    texture_revision: u64,
    cached_preview_revision: u64,
    cached_preview_texture_revision: u64,
    cached_preview: Arc<Vec<PreviewTriangle>>,
    display_worker: DisplayWorker,
    input_worker: InputWorker,
    asset_import_worker: AssetImportWorker,
    viewport_requested_key: Option<DepthCacheKey>,
    viewport_render_in_flight: bool,
    presented_view: Option<PresentedView>,
    dialog_result: Option<mpsc::Receiver<Option<PathBuf>>>,
    save_dialog_result: Option<mpsc::Receiver<Option<PathBuf>>>,
    load_dialog_result: Option<mpsc::Receiver<Option<PathBuf>>>,
    compositor_image_dialog_result: Option<mpsc::Receiver<Option<PathBuf>>>,
}

impl EditorApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        if let Some(render_state) = &cc.wgpu_render_state {
            let _ = vulkan_runtime::install_runtime(
                render_state.device.clone(),
                render_state.queue.clone(),
                render_state.adapter.get_info().name,
            );
        }
        Self {
            scene: EditorScene::default(),
            workspace_tab: WorkspaceTab::Scene,
            active_tool: Tool::Move,
            play_state: PlayState::Editing,
            build_started: None,
            bottom_tab: BottomTab::Assets,
            inspector_tab: InspectorTab::Inspector,
            material_tab: MaterialTab::Shader,
            compositor_pan: Vec2::ZERO,
            compositor_zoom: 1.0,
            compositor_selected_node: 0,
            compositor_nodes: Vec::new(),
            compositor_next_id: 0,
            compositor_image_dialog_target: None,
            compositor_links: Vec::new(),
            compositor_dragging_node: None,
            compositor_pending_output: None,
            compositor_pending_spawn: None,
            compositor_texture_overrides: Vec::new(),
            compositor_eval_cache: HashMap::new(),
            compositor_gpu_cache: HashMap::new(),
            compositor_image_cache: HashMap::new(),
            compositor_source_cache: HashMap::new(),
            compositor_debug_textures: HashMap::new(),
            compositor_apply_due: None,
            compositor_control_started: None,
            compositor_present_revision: None,
            compositor_lod_max_dimension: u32::MAX,
            render_device: RenderDevice::Vulkan,
            performance: EditorPerformanceTelemetry::default(),
            cpu_compositor: compositor_cpu::CpuGraphWorker::new()
                .expect("CPU compositor worker should start"),
            vulkan_compositor: None,
            vulkan_compositor_attempted: false,
            compositor_next_generation: 0,
            vulkan_latest_generation: 0,
            vulkan_waiting_generation: None,
            compositor_pending_target: None,
            compositor_graph_queue: VecDeque::new(),
            compositor_clock_started: Instant::now(),
            compositor_next_time_tick: Instant::now(),
            dynamics_fields: HashMap::new(),
            dynamics_cloth: HashMap::new(),
            object_simulation_states: HashMap::new(),
            joint_simulation_states: HashMap::new(),
            dynamics_enabled: BTreeSet::new(),
            dynamics_running: false,
            dynamics_single_step: false,
            dynamics_time: 0.0,
            dynamics_accumulator: 0.0,
            dynamics_last_tick: Instant::now(),
            dynamics_wind: WindField::default(),
            dynamics_settings: ClothSettings::default(),
            dynamics_pan: Vec2::ZERO,
            dynamics_zoom: 1.0,
            dynamics_node_positions: [
                Vec2::new(20.0, 80.0),
                Vec2::new(20.0, 300.0),
                Vec2::new(300.0, 80.0),
                Vec2::new(300.0, 300.0),
                Vec2::new(590.0, 170.0),
                Vec2::new(880.0, 170.0),
            ],
            dynamics_dragging_node: None,
            paint_mode: PaintMode::Replace,
            paint_value: 1.0,
            paint_strength: 0.5,
            paint_radius_pixels: 45.0,
            field_undo: Vec::new(),
            field_redo: Vec::new(),
            field_stroke_before: None,
            painted_masks: HashMap::new(),
            texture_paint_undo: Vec::new(),
            texture_paint_redo: Vec::new(),
            texture_paint_stroke_before: None,
            texture_paint_last_uv: None,
            texture_paint_heatmap: true,
            advanced: false,
            global_light_enabled: true,
            camera_preview_visible: false,
            global_shadow_resolution: 512,
            shadow_quality: 3,
            shadow_blur_radius: 1,
            target_fps: 60,
            next_viewport_frame: Instant::now(),
            save_cache_in_file: false,
            show_grid: true,
            grid_spacing: 1.0,
            snap: false,
            viewport_focused: false,
            camera_yaw: -0.55,
            camera_pitch: 0.42,
            camera_zoom: 1.0,
            camera_target: CoreVec3::ZERO,
            hierarchy_filter: String::new(),
            hierarchy_drag_candidate: None,
            logs: vec![
                LogEntry {
                    level: "INFO",
                    color: Color32::from_rgb(103, 191, 255),
                    message: "Scene ready. Editor renderer initialized.".into(),
                },
                LogEntry {
                    level: "TIP",
                    color: Color32::from_rgb(112, 210, 156),
                    message: "Press Play to compile and launch a real game process.".into(),
                },
            ],
            project_path: PathBuf::from("Unnamed.zfp"),
            project_has_destination: false,
            project_dirty: true,
            project_error_dialog: None,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            asset_import_path: String::new(),
            asset_loading: BTreeMap::new(),
            asset_loading_present_revision: BTreeMap::new(),
            imported_assets: builtin_imported_assets(),
            viewport_mode: ViewportMode::Shaded,
            projection_mode: ProjectionMode::Orthographic,
            viewport_color: None,
            viewport_native_texture: None,
            viewport_native_view: None,
            wgpu_render_state: cc.wgpu_render_state.clone(),
            viewport_depth: Vec::new(),
            viewport_depth_key: None,
            scene_revision: 0,
            texture_revision: 0,
            cached_preview_revision: u64::MAX,
            cached_preview_texture_revision: u64::MAX,
            cached_preview: Arc::new(Vec::new()),
            display_worker: DisplayWorker::new(cc.egui_ctx.clone()),
            input_worker: InputWorker::new(cc.egui_ctx.clone()),
            asset_import_worker: AssetImportWorker::new(cc.egui_ctx.clone()),
            viewport_requested_key: None,
            viewport_render_in_flight: false,
            presented_view: None,
            dialog_result: None,
            save_dialog_result: None,
            load_dialog_result: None,
            compositor_image_dialog_result: None,
        }
    }

    fn import_asset(&mut self, add_to_scene: bool) {
        let path = self.asset_import_path.trim().to_owned();
        if path.is_empty() {
            return;
        }
        self.asset_loading.insert(path.clone(), "Queued");
        self.asset_import_worker
            .submit(path, add_to_scene, self.save_cache_in_file);
        self.asset_import_path.clear();
    }

    fn poll_asset_imports(&mut self) {
        while let Ok(progress) = self.asset_import_worker.progress.try_recv() {
            self.asset_loading.insert(progress.path, progress.phase);
        }
        while let Ok(result) = self.asset_import_worker.results.try_recv() {
            if let Some(cache_message) = result.cache_message.as_ref() {
                match cache_message {
                    Ok(message) => self.logs.push(LogEntry {
                        level: "CACHE",
                        color: Color32::from_rgb(112, 174, 220),
                        message: message.clone(),
                    }),
                    Err(message) => self.logs.push(LogEntry {
                        level: "WARN",
                        color: Color32::from_rgb(235, 167, 88),
                        message: format!("Texture cache unavailable: {message}"),
                    }),
                }
            }
            let path = result.path;
            match result.asset {
                Ok(prepared) => {
                    let PreparedImport {
                        asset,
                        autofixed_mesh,
                        autofix_report,
                        bounds,
                        inferred_grid_spacing,
                    } = prepared;
                    let completed_path = path.clone();
                    let triangle_count = asset.triangle_count();
                    let inferred_grid_spacing = self
                        .imported_assets
                        .is_empty()
                        .then_some(inferred_grid_spacing)
                        .flatten();
                    if let Some(spacing) = inferred_grid_spacing {
                        self.grid_spacing = spacing;
                    }
                    self.logs.push(LogEntry {
                        level: "IMPORT",
                        color: Color32::from_rgb(112, 210, 156),
                        message: format!(
                            "Imported {}: {} vertices, {} triangles{}",
                            &path,
                            asset.vertices.len(),
                            triangle_count,
                            inferred_grid_spacing
                                .map(|spacing| {
                                    format!(" · grid inferred at {}", format_grid_spacing(spacing))
                                })
                                .unwrap_or_default()
                        ),
                    });
                    let asset_index = if let Some((index, existing)) = self
                        .imported_assets
                        .iter_mut()
                        .enumerate()
                        .find(|(_, existing)| existing.path == path)
                    {
                        existing.mesh = asset;
                        existing.autofixed_mesh = autofixed_mesh;
                        existing.autofix_report = autofix_report;
                        existing.bounds = bounds;
                        index
                    } else {
                        self.imported_assets.push(ImportedAsset {
                            path,
                            mesh: asset,
                            autofixed_mesh,
                            autofix_report,
                            bounds,
                        });
                        self.imported_assets.len() - 1
                    };
                    self.scene_revision = self.scene_revision.wrapping_add(1);
                    if result.add_to_scene {
                        self.add_asset_to_scene(asset_index);
                    }
                    let rebuild: Vec<NodeId> = self
                        .dynamics_enabled
                        .iter()
                        .copied()
                        .filter(|id| {
                            self.scene.tree.node(*id).is_ok_and(|node| {
                                node.components.iter().any(|component| {
                                    matches!(
                                        component,
                                        Component::Model { asset, .. } if asset == &completed_path
                                    )
                                })
                            })
                        })
                        .collect();
                    for id in rebuild {
                        let _ = self.enable_dynamics_for(id);
                    }
                    self.asset_loading
                        .insert(completed_path.clone(), "Uploading renderer resources");
                    self.asset_loading_present_revision
                        .insert(completed_path, self.scene_revision);
                }
                Err(error) => {
                    self.asset_loading.remove(&path);
                    self.logs.push(LogEntry {
                        level: "ERROR",
                        color: Color32::from_rgb(235, 91, 91),
                        message: format!("Asset import failed: {error}"),
                    });
                }
            }
        }
    }

    fn add_asset_to_scene(&mut self, asset_index: usize) {
        let Some(asset) = self.imported_assets.get(asset_index) else {
            return;
        };
        let asset_path = asset.path.clone();
        let asset_name = asset.mesh.name.clone();
        let asset_nodes = asset.mesh.nodes.clone();
        let primitive_names = asset
            .mesh
            .primitives
            .iter()
            .map(|primitive| primitive.name.clone())
            .collect::<Vec<_>>();
        let previous = self.scene.tree.clone();
        let object_name = std::path::Path::new(&asset_path)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or(&asset_name)
            .to_owned();
        if asset_nodes.is_empty() {
            let id = self.scene.add(&object_name, ObjectKind::Empty, None);
            let _ = self.scene.tree.add_component(
                id,
                Component::Model {
                    asset: asset_path,
                    primitive: None,
                },
            );
            let _ = self.scene.tree.set_attribute(
                id,
                mesh_autofix_key(),
                AttributeDeclaration::Value(Attribute::Bool(true)),
            );
            self.scene.selected = Some(id);
        } else {
            let roots = asset_nodes
                .iter()
                .enumerate()
                .filter_map(|(index, node)| node.parent.is_none().then_some(index))
                .collect::<Vec<_>>();
            let wrapper =
                (roots.len() > 1).then(|| self.scene.add(&object_name, ObjectKind::Empty, None));
            let mut scene_nodes = Vec::with_capacity(asset_nodes.len());
            for (node_index, node) in asset_nodes.iter().enumerate() {
                let parent = node
                    .parent
                    .and_then(|parent| scene_nodes.get(parent).copied())
                    .or(wrapper);
                let id = self.scene.add(&node.name, ObjectKind::Empty, parent);
                let local = node.local;
                let _ = self.scene.tree.set_local_transform(
                    id,
                    Transform {
                        translation: CoreVec3::new(
                            local.translation[0],
                            local.translation[1],
                            local.translation[2],
                        ),
                        rotation: Quat {
                            w: local.rotation[0],
                            x: local.rotation[1],
                            y: local.rotation[2],
                            z: local.rotation[3],
                        }
                        .normalized(),
                        scale: CoreVec3::new(local.scale[0], local.scale[1], local.scale[2]),
                    },
                );
                let _ = self.scene.tree.set_attribute(
                    id,
                    AttributeKey::Custom("gltf.node_index".into()),
                    AttributeDeclaration::Value(Attribute::Integer(node_index as i64)),
                );
                for (ordinal, primitive) in node.primitives.iter().copied().enumerate() {
                    let model_id = if ordinal == 0 {
                        id
                    } else {
                        let name = primitive_names
                            .get(primitive)
                            .map(String::as_str)
                            .unwrap_or("Primitive");
                        self.scene.add(name, ObjectKind::Empty, Some(id))
                    };
                    let _ = self.scene.tree.add_component(
                        model_id,
                        Component::Model {
                            asset: asset_path.clone(),
                            primitive: Some(primitive),
                        },
                    );
                    let _ = self.scene.tree.set_attribute(
                        model_id,
                        mesh_autofix_key(),
                        AttributeDeclaration::Value(Attribute::Bool(true)),
                    );
                }
                scene_nodes.push(id);
            }
            self.scene.selected = wrapper.or_else(|| roots.first().map(|root| scene_nodes[*root]));
        }
        self.record_undo(previous);
        self.logs.push(LogEntry {
            level: "SCENE",
            color: Color32::from_rgb(103, 191, 255),
            message: format!(
                "Added `{object_name}` to the scene{}",
                (!asset_nodes.is_empty())
                    .then(|| format!(" as {} linked glTF objects", asset_nodes.len()))
                    .unwrap_or_default()
            ),
        });
    }

    fn add_builtin_primitive(&mut self, primitive: BuiltinPrimitive) {
        let path = primitive.asset_path();
        if !self.imported_assets.iter().any(|asset| asset.path == path) {
            let mesh = builtin_primitive_mesh(primitive);
            let bounds = mesh_bounds(&mesh);
            self.imported_assets.push(ImportedAsset {
                path: path.into(),
                autofixed_mesh: mesh.clone(),
                mesh,
                autofix_report: MeshAutofixReport::default(),
                bounds,
            });
        }

        let previous = self.scene.tree.clone();
        let id = self.scene.add(primitive.label(), ObjectKind::Empty, None);
        let primitive_scale = self.grid_spacing.max(f32::EPSILON);
        let _ = self.scene.tree.set_local_transform(
            id,
            Transform {
                scale: CoreVec3::new(primitive_scale, primitive_scale, primitive_scale),
                ..Transform::IDENTITY
            },
        );
        self.scene
            .tree
            .add_component(
                id,
                Component::Model {
                    asset: path.into(),
                    primitive: None,
                },
            )
            .expect("fresh primitive object");
        let _ = self.scene.tree.set_attribute(
            id,
            mesh_autofix_key(),
            AttributeDeclaration::Value(Attribute::Bool(false)),
        );
        self.scene.selected = Some(id);
        self.record_undo(previous);
        self.scene_revision = self.scene_revision.wrapping_add(1);
        self.project_dirty = true;
        self.logs.push(LogEntry {
            level: "SCENE",
            color: Color32::from_rgb(103, 191, 255),
            message: format!("Added {} primitive", primitive.label()),
        });
    }

    fn best_fit_collider(&self, root: NodeId, shape: ColliderShape) -> Component {
        let bounds = self
            .collider_subtree_bounds(root)
            .unwrap_or(([-0.5; 3], [0.5; 3]));
        fitted_collider_from_bounds(bounds, shape)
    }

    fn collider_subtree_bounds(&self, root: NodeId) -> Option<([f32; 3], [f32; 3])> {
        let root_global = self.scene.tree.node(root).ok()?.global_transform();
        let mut stack = vec![root];
        let mut minimum = [f32::INFINITY; 3];
        let mut maximum = [f32::NEG_INFINITY; 3];
        let mut found = false;
        while let Some(id) = stack.pop() {
            let node = self.scene.tree.node(id).ok()?;
            stack.extend(node.children().iter().copied());
            let Ok(relative) = root_global.relative_to(node.global_transform()) else {
                continue;
            };
            for component in &node.components {
                let Component::Model { asset, primitive } = component else {
                    continue;
                };
                let Some(mesh) = self
                    .imported_assets
                    .iter()
                    .find(|candidate| candidate.path == *asset)
                    .map(|asset| &asset.mesh)
                else {
                    continue;
                };
                let primitive_indices = primitive
                    .map(|index| index..index.saturating_add(1))
                    .unwrap_or(0..mesh.primitives.len());
                for primitive_index in primitive_indices {
                    let Some(primitive) = mesh.primitives.get(primitive_index) else {
                        continue;
                    };
                    for index in &primitive.indices {
                        let Some(vertex) = mesh.vertices.get(*index as usize) else {
                            continue;
                        };
                        let point =
                            relative
                                .rotation
                                .rotate(relative.scale.component_mul(CoreVec3::new(
                                    vertex.position[0],
                                    vertex.position[1],
                                    vertex.position[2],
                                )))
                                + relative.translation;
                        for (axis, value) in [point.x, point.y, point.z].into_iter().enumerate() {
                            minimum[axis] = minimum[axis].min(value);
                            maximum[axis] = maximum[axis].max(value);
                        }
                        found = true;
                    }
                }
            }
        }
        found.then_some((minimum, maximum))
    }

    fn build_preview_triangles(&self) -> Vec<PreviewTriangle> {
        let total_scene_triangles = self
            .scene
            .tree
            .iter()
            .filter_map(|(_, node)| {
                let (path, primitive) =
                    node.components
                        .iter()
                        .find_map(|component| match component {
                            Component::Model { asset, primitive } => {
                                Some((asset.as_str(), *primitive))
                            }
                            _ => None,
                        })?;
                self.imported_assets
                    .iter()
                    .find(|asset| asset.path == path)
                    .map(|asset| {
                        primitive
                            .and_then(|index| asset.mesh.primitives.get(index))
                            .map_or_else(
                                || asset.mesh.triangle_count(),
                                |primitive| primitive.indices.len() / 3,
                            )
                    })
            })
            .sum::<usize>();
        let mut output = Vec::with_capacity(total_scene_triangles);
        for (id, node) in self.scene.tree.iter() {
            let Some((path, primitive_filter)) =
                node.components
                    .iter()
                    .find_map(|component| match component {
                        Component::Model { asset, primitive } => Some((asset.as_str(), *primitive)),
                        _ => None,
                    })
            else {
                continue;
            };
            let mesh_autofix = self
                .scene
                .tree
                .resolve_attribute(id, &mesh_autofix_key())
                .ok()
                .and_then(|attribute| match attribute.value {
                    Attribute::Bool(value) => Some(value),
                    _ => None,
                })
                .unwrap_or(true);
            let Some(asset) = self
                .imported_assets
                .iter()
                .find(|asset| asset.path == path)
                .map(|asset| {
                    if mesh_autofix {
                        &asset.autofixed_mesh
                    } else {
                        &asset.mesh
                    }
                })
            else {
                continue;
            };
            let transform = node.global_transform();
            let shader = self
                .scene
                .tree
                .resolve_attribute(id, &material_shader_key())
                .ok()
                .and_then(|resolved| match resolved.value {
                    Attribute::Text(value) => ShaderMode::from_attribute(&value),
                    _ => None,
                })
                .unwrap_or(ShaderMode::Diffuse);
            let smooth_normals = self
                .scene
                .tree
                .resolve_attribute(id, &material_smooth_normals_key())
                .ok()
                .and_then(|resolved| match resolved.value {
                    Attribute::Bool(value) => Some(value),
                    _ => None,
                })
                .unwrap_or(true);
            let use_imported_optics = self
                .scene
                .tree
                .resolve_attribute(id, &material_use_imported_optics_key())
                .ok()
                .and_then(|resolved| match resolved.value {
                    Attribute::Bool(value) => Some(value),
                    _ => None,
                })
                .unwrap_or(true);
            let casts_shadows = self
                .scene
                .tree
                .resolve_attribute(id, &AttributeKey::CastShadows)
                .ok()
                .and_then(|resolved| match resolved.value {
                    Attribute::Bool(value) => Some(value),
                    _ => None,
                })
                .unwrap_or(true);
            let material_tint = self
                .scene
                .tree
                .resolve_attribute(id, &material_tint_key())
                .ok()
                .and_then(|resolved| match resolved.value {
                    Attribute::Color(value) => Some(value),
                    _ => None,
                })
                .unwrap_or([1.0; 4]);
            let compositor_override = self
                .compositor_texture_overrides
                .iter()
                .find(|(target, _)| *target == id)
                .map(|(_, texture)| texture.clone());
            let deformation = self.dynamics_cloth.get(&id).map(|cloth| &cloth.snapshot);
            let field = self.dynamics_fields.get(&id);
            for (primitive_index, primitive) in asset.primitives.iter().enumerate() {
                if primitive_filter.is_some_and(|selected| selected != primitive_index) {
                    continue;
                }
                let material = primitive
                    .material
                    .as_ref()
                    .and_then(|name| asset.materials.get(name));
                let source_base_color = material
                    .map(|material| material.base_color)
                    .unwrap_or([0.42, 0.64, 0.78, 1.0]);
                let tinted_base_color = std::array::from_fn(|channel| {
                    source_base_color[channel] * material_tint[channel]
                });
                let field_preview = self.active_tool == Tool::FieldPaint && field.is_some();
                let texture_paint_preview =
                    self.active_tool == Tool::TexturePaint && self.painted_masks.contains_key(&id);
                let base_color =
                    if compositor_override.is_some() || field_preview || texture_paint_preview {
                        [1.0; 4]
                    } else {
                        tinted_base_color
                    };
                let source_texture_asset = material
                    .and_then(|material| material.base_color_texture.as_ref())
                    .and_then(|name| asset.textures.get(name));
                let source_texture_cache_key = source_texture_asset
                    .map(|texture| texture as *const TextureAsset as usize as u64)
                    .unwrap_or(0);
                let source_texture = source_texture_asset.cloned().map(Arc::new);
                let painted_preview = texture_paint_preview
                    .then(|| Arc::new(self.painted_masks[&id].texture(self.texture_paint_heatmap)));
                let texture = match (&compositor_override, field_preview, painted_preview) {
                    (_, true, _) => None,
                    (_, false, Some(texture)) => Some(texture),
                    (Some(TextureOverride::Cpu(texture)), false, None) => Some(Arc::clone(texture)),
                    _ => source_texture.clone(),
                };
                let gpu_texture =
                    match (&compositor_override, field_preview || texture_paint_preview) {
                        (_, true) => None,
                        (Some(TextureOverride::Gpu(texture)), false) => Some(Arc::clone(texture)),
                        _ => None,
                    };
                let texture_cache_key = match (
                    &compositor_override,
                    field_preview,
                    texture_paint_preview,
                    &texture,
                ) {
                    (Some(TextureOverride::Cpu(texture)), false, false, _) => {
                        Arc::as_ptr(texture) as usize as u64
                    }
                    (_, false, false, Some(_)) => source_texture_cache_key,
                    (_, _, _, Some(texture)) => Arc::as_ptr(texture) as usize as u64,
                    _ => 0,
                };
                let transmission = if use_imported_optics {
                    material
                        .and_then(|material| material.transmission)
                        .unwrap_or(0.0)
                        .clamp(0.0, 1.0)
                } else {
                    0.0
                };
                let ior = if use_imported_optics {
                    material
                        .and_then(|material| material.ior)
                        .unwrap_or(1.5)
                        .clamp(1.0, 2.5)
                } else {
                    1.5
                };
                for triangle in primitive.indices.chunks_exact(3) {
                    let mut vertices = [PreviewVertex {
                        source_index: 0,
                        local_position: [0.0; 3],
                        local_normal: [0.0, 0.0, 1.0],
                        position: [0.0; 3],
                        normal: [0.0, 0.0, 1.0],
                        uv: [0.0; 2],
                        color: [1.0; 4],
                    }; 3];
                    for (destination, index) in vertices.iter_mut().zip(triangle) {
                        let source_index = *index as usize;
                        let source_vertex = asset.vertices[source_index];
                        let source = deformation
                            .and_then(|snapshot| snapshot.positions.get(source_index))
                            .copied()
                            .unwrap_or(source_vertex.position);
                        let local = CoreVec3::new(source[0], source[1], source[2]);
                        let world = transform
                            .rotation
                            .rotate(transform.scale.component_mul(local))
                            + transform.translation;
                        let source_normal = deformation
                            .and_then(|snapshot| snapshot.normals.get(source_index))
                            .copied()
                            .unwrap_or(source_vertex.normal);
                        let local_normal =
                            CoreVec3::new(source_normal[0], source_normal[1], source_normal[2]);
                        let world_normal =
                            transform_normal(local_normal, transform.scale, transform.rotation);
                        *destination = PreviewVertex {
                            source_index,
                            local_position: source,
                            local_normal: source_normal,
                            position: [world.x, world.y, world.z],
                            normal: [world_normal.x, world_normal.y, world_normal.z],
                            uv: source_vertex.uv,
                            color: if self.active_tool == Tool::FieldPaint {
                                field
                                    .and_then(|field| {
                                        field.values.get(source_index).map(|value| {
                                            heatmap(*value, field.display_min, field.display_max)
                                        })
                                    })
                                    .unwrap_or(source_vertex.color)
                            } else {
                                source_vertex.color
                            },
                        };
                    }
                    output.push(PreviewTriangle {
                        object_id: id,
                        object_transform: transform,
                        vertices,
                        base_color,
                        source_base_color: tinted_base_color,
                        texture: texture.clone(),
                        texture_cache_key,
                        gpu_texture: gpu_texture.clone(),
                        source_texture: source_texture.clone(),
                        shader,
                        smooth_normals,
                        transmission,
                        ior,
                        casts_shadows,
                    });
                }
            }
        }
        output
    }

    fn refresh_preview_cache(&mut self) {
        if self.cached_preview_revision != self.scene_revision {
            self.cached_preview = Arc::new(self.build_preview_triangles());
            self.cached_preview_revision = self.scene_revision;
            self.cached_preview_texture_revision = self.texture_revision;
        } else if self.cached_preview_texture_revision != self.texture_revision {
            let triangles = Arc::make_mut(&mut self.cached_preview);
            for triangle in triangles {
                if let Some((_, texture)) = self
                    .compositor_texture_overrides
                    .iter()
                    .find(|(target, _)| *target == triangle.object_id)
                {
                    match texture {
                        TextureOverride::Cpu(texture) => {
                            triangle.texture = Some(Arc::clone(texture));
                            triangle.texture_cache_key = Arc::as_ptr(texture) as usize as u64;
                            triangle.gpu_texture = None;
                        }
                        TextureOverride::Gpu(texture) => {
                            triangle.texture = None;
                            triangle.texture_cache_key = 0;
                            triangle.gpu_texture = Some(Arc::clone(texture));
                        }
                    }
                    triangle.base_color = [1.0; 4];
                } else {
                    triangle.texture = triangle.source_texture.clone();
                    triangle.gpu_texture = None;
                    triangle.base_color = triangle.source_base_color;
                }
            }
            self.cached_preview_texture_revision = self.texture_revision;
        }
    }

    fn record_undo(&mut self, previous: GeometryTree) {
        self.undo_stack.push(previous);
        if self.undo_stack.len() > 100 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
        self.project_dirty = true;
        self.scene_revision = self.scene_revision.wrapping_add(1);
    }

    fn record_camera_only_undo(&mut self, previous: GeometryTree) {
        self.undo_stack.push(previous);
        if self.undo_stack.len() > 100 {
            self.undo_stack.remove(0);
        }
        self.redo_stack.clear();
        self.project_dirty = true;
        self.viewport_requested_key = None;
    }

    fn subtree_affects_render_or_shadows(&self, id: NodeId) -> bool {
        let Ok(node) = self.scene.tree.node(id) else {
            return false;
        };
        node.components
            .iter()
            .any(|component| matches!(component, Component::Model { .. } | Component::Light { .. }))
            || node
                .children()
                .iter()
                .copied()
                .any(|child| self.subtree_affects_render_or_shadows(child))
    }

    fn undo(&mut self) {
        if let Some(previous) = self.undo_stack.pop() {
            self.redo_stack.push(self.scene.tree.clone());
            self.scene.tree = previous;
            self.scene.selected = self
                .scene
                .selected
                .filter(|id| self.scene.tree.contains(*id))
                .or_else(|| self.scene.tree.roots().first().copied());
            self.project_dirty = true;
            self.scene_revision = self.scene_revision.wrapping_add(1);
        }
    }

    fn redo(&mut self) {
        if let Some(next) = self.redo_stack.pop() {
            self.undo_stack.push(self.scene.tree.clone());
            self.scene.tree = next;
            self.scene.selected = self
                .scene
                .selected
                .filter(|id| self.scene.tree.contains(*id))
                .or_else(|| self.scene.tree.roots().first().copied());
            self.project_dirty = true;
            self.scene_revision = self.scene_revision.wrapping_add(1);
        }
    }

    fn save_project(&mut self) {
        let (project, files) = self.project_bundle();
        match project.and_then(|project| {
            save_zfp(&self.project_path, &project, &files).map_err(|error| error.to_string())
        }) {
            Ok(()) => {
                self.project_dirty = false;
                self.scene_revision = self.scene_revision.wrapping_add(1);
                self.logs.push(LogEntry {
                    level: "SAVE",
                    color: Color32::from_rgb(112, 210, 156),
                    message: format!("Saved {}", self.project_path.display()),
                });
            }
            Err(error) => {
                let message = format!(
                    "Could not save `{}`.\n\n{}\n\nThe previous project file was preserved.",
                    self.project_path.display(),
                    error
                );
                self.project_error_dialog = Some(("Project Save Failed".into(), message));
                self.logs.push(LogEntry {
                    level: "ERROR",
                    color: Color32::from_rgb(235, 91, 91),
                    message: format!("Could not save project: {error}"),
                });
            }
        }
    }

    fn save_or_save_as(&mut self, ctx: &egui::Context) {
        if self.project_has_destination {
            self.save_project();
        } else {
            self.start_save_as(ctx);
        }
    }

    fn project_display_name(&self) -> String {
        let filename = self
            .project_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("Unnamed.zfp");
        if self.project_dirty {
            format!("*{filename}")
        } else {
            filename.to_owned()
        }
    }

    fn start_save_as(&mut self, ctx: &egui::Context) {
        if self.save_dialog_result.is_some() {
            return;
        }
        let directory = self
            .project_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .map(std::path::Path::to_path_buf)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        let filename = self
            .project_path
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or("zerofps-project.zfp")
            .to_owned();
        let repaint = ctx.clone();
        let (sender, receiver) = mpsc::channel();
        self.save_dialog_result = Some(receiver);
        std::thread::Builder::new()
            .name("zerofps-save-dialog".into())
            .spawn(move || {
                let path = rfd::FileDialog::new()
                    .set_title("Save ZeroFPS Project As")
                    .set_directory(directory)
                    .set_file_name(filename)
                    .add_filter("ZeroFPS Project", &["zfp"])
                    .save_file();
                let _ = sender.send(path);
                repaint.request_repaint();
            })
            .expect("save-dialog worker thread should start");
    }

    fn poll_save_as(&mut self) {
        let Some(receiver) = &self.save_dialog_result else {
            return;
        };
        match receiver.try_recv() {
            Ok(path) => {
                self.save_dialog_result = None;
                if let Some(mut path) = path {
                    if !path
                        .extension()
                        .and_then(|extension| extension.to_str())
                        .is_some_and(|extension| extension.eq_ignore_ascii_case("zfp"))
                    {
                        path.set_extension("zfp");
                    }
                    self.project_path = path;
                    self.project_has_destination = true;
                    self.save_project();
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => self.save_dialog_result = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    fn start_load_project(&mut self, ctx: &egui::Context) {
        if self.load_dialog_result.is_some() {
            return;
        }
        let directory = self
            .project_path
            .parent()
            .filter(|path| !path.as_os_str().is_empty())
            .map(std::path::Path::to_path_buf)
            .or_else(|| std::env::current_dir().ok())
            .unwrap_or_else(|| PathBuf::from("."));
        let repaint = ctx.clone();
        let (sender, receiver) = mpsc::channel();
        self.load_dialog_result = Some(receiver);
        std::thread::Builder::new()
            .name("zerofps-project-dialog".into())
            .spawn(move || {
                let path = rfd::FileDialog::new()
                    .set_title("Load ZeroFPS Project")
                    .set_directory(directory)
                    .add_filter("ZeroFPS Project", &["zfp"])
                    .pick_file();
                let _ = sender.send(path);
                repaint.request_repaint();
            })
            .expect("project-dialog worker thread should start");
    }

    fn poll_load_project(&mut self) {
        let Some(receiver) = &self.load_dialog_result else {
            return;
        };
        match receiver.try_recv() {
            Ok(path) => {
                self.load_dialog_result = None;
                if let Some(path) = path {
                    self.load_project_from(path);
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => self.load_dialog_result = None,
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    fn start_compositor_image_import(&mut self, ctx: &egui::Context) {
        if self.compositor_image_dialog_result.is_some() {
            return;
        }
        let directory = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
        let repaint = ctx.clone();
        let (sender, receiver) = mpsc::channel();
        self.compositor_image_dialog_result = Some(receiver);
        std::thread::Builder::new()
            .name("zerofps-compositor-image-dialog".into())
            .spawn(move || {
                let path = rfd::FileDialog::new()
                    .set_title("Import Compositor Image")
                    .set_directory(directory)
                    .add_filter("Images", &["png", "jpg", "jpeg"])
                    .add_filter("PNG", &["png"])
                    .add_filter("JPEG", &["jpg", "jpeg"])
                    .pick_file();
                let _ = sender.send(path);
                repaint.request_repaint();
            })
            .expect("compositor-image dialog worker should start");
    }

    fn poll_compositor_image_import(&mut self) {
        let Some(receiver) = &self.compositor_image_dialog_result else {
            return;
        };
        match receiver.try_recv() {
            Ok(path) => {
                self.compositor_image_dialog_result = None;
                if let Some(path) = path {
                    let path_str = path.to_string_lossy().into_owned();
                    if let Some(target_id) = self.compositor_image_dialog_target.take() {
                        if let Some(node) =
                            self.compositor_nodes.iter_mut().find(|n| n.id == target_id)
                        {
                            if let NodeSettings::ImageAsset { path } = &mut node.settings {
                                *path = path_str.clone();
                                self.project_dirty = true;
                                self.logs.push(LogEntry {
                                    level: "IMPORT",
                                    color: Color32::from_rgb(112, 210, 156),
                                    message: format!("Registered compositor image `{path_str}`"),
                                });
                            }
                        }
                    }
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => {
                self.compositor_image_dialog_result = None;
            }
            Err(mpsc::TryRecvError::Empty) => {}
        }
    }

    fn load_project_from(&mut self, path: PathBuf) {
        let cache_name = path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("project");
        let cache_root = PathBuf::from("target/zfp-cache").join(cache_name);
        match load_zfp(&path, &cache_root) {
            Ok(mut bundle) => {
                rewrite_asset_paths(&mut bundle.project, &bundle.extracted_files);
                rewrite_compositor_image_paths(&mut bundle.project, &bundle.extracted_files);
                let project = bundle.project;
                self.apply_project_properties(&project);
                self.scene.tree = project.scene.geometry;
                self.compositor_texture_overrides.clear();
                self.scene.selected = self.scene.tree.roots().first().copied();
                let referenced_assets: BTreeSet<String> = self
                    .scene
                    .tree
                    .iter()
                    .flat_map(|(_, node)| node.components.iter())
                    .filter_map(|component| match component {
                        Component::Model { asset, .. } => Some(asset.clone()),
                        _ => None,
                    })
                    .collect();
                for path in referenced_assets {
                    if !self.imported_assets.iter().any(|asset| asset.path == path) {
                        self.asset_import_path = path;
                        self.import_asset(false);
                    }
                }
                for id in self.dynamics_enabled.clone() {
                    let _ = self.enable_dynamics_for(id);
                }
                self.project_dirty = false;
                self.project_has_destination = true;
                self.project_path = path;
                self.logs.push(LogEntry {
                    level: "LOAD",
                    color: Color32::from_rgb(103, 191, 255),
                    message: format!("Loaded {}", self.project_path.display()),
                });
            }
            Err(error) => {
                let message = format!("Could not open `{}`.\n\n{}", path.display(), error);
                self.project_error_dialog = Some(("Project Load Failed".into(), message));
                self.logs.push(LogEntry {
                    level: "ERROR",
                    color: Color32::from_rgb(235, 91, 91),
                    message: format!("Could not load project: {error}"),
                });
            }
        }
    }

    fn project_error_popup(&mut self, ctx: &egui::Context) {
        let Some((title, message)) = self.project_error_dialog.clone() else {
            return;
        };
        let mut dismiss = false;
        egui::Window::new(title)
            .id(Id::new("project_error_popup"))
            .anchor(Align2::CENTER_CENTER, Vec2::ZERO)
            .collapsible(false)
            .resizable(false)
            .movable(false)
            .show(ctx, |ui| {
                ui.set_max_width(520.0);
                ui.colored_label(
                    Color32::from_rgb(235, 91, 91),
                    "The project operation failed.",
                );
                ui.add_space(6.0);
                ui.label(message);
                ui.add_space(12.0);
                ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                    dismiss = ui.button("OK").clicked();
                });
            });
        if dismiss || ctx.input(|input| input.key_pressed(Key::Escape)) {
            self.project_error_dialog = None;
        }
    }

    fn project_bundle(&self) -> (Result<ProjectFile, String>, Vec<BundleAsset>) {
        let mut project = ProjectFile::new("ZeroFPS Project", "Scene 01", self.scene.tree.clone());
        for (key, value) in [
            ("editor.grid_spacing", self.grid_spacing.to_string()),
            ("editor.show_grid", self.show_grid.to_string()),
            (
                "editor.global_light_enabled",
                self.global_light_enabled.to_string(),
            ),
            (
                "editor.global_shadow_resolution",
                self.global_shadow_resolution.to_string(),
            ),
            ("editor.shadow_quality", self.shadow_quality.to_string()),
            (
                "editor.shadow_blur_radius",
                self.shadow_blur_radius.to_string(),
            ),
            ("editor.target_fps", self.target_fps.to_string()),
            (
                "editor.save_cache_in_file",
                self.save_cache_in_file.to_string(),
            ),
            ("editor.camera_yaw", self.camera_yaw.to_string()),
            ("editor.camera_pitch", self.camera_pitch.to_string()),
            ("editor.camera_zoom", self.camera_zoom.to_string()),
            ("editor.camera_target_x", self.camera_target.x.to_string()),
            ("editor.camera_target_y", self.camera_target.y.to_string()),
            ("editor.camera_target_z", self.camera_target.z.to_string()),
            (
                "editor.projection",
                self.projection_mode.label().to_ascii_lowercase(),
            ),
            (
                "editor.viewport_mode",
                self.viewport_mode.label().to_ascii_lowercase(),
            ),
            (
                "editor.workspace",
                match self.workspace_tab {
                    WorkspaceTab::Scene => "scene",
                    WorkspaceTab::Compositing => "nodes",
                }
                .into(),
            ),
            (
                "editor.device",
                self.render_device.label().to_ascii_lowercase(),
            ),
        ] {
            project.project.properties.insert(key.into(), value);
        }
        // Save compositor next_id and selected
        project.project.properties.insert(
            "compositor.next_id".into(),
            self.compositor_next_id.to_string(),
        );
        project.project.properties.insert(
            "compositor.selected".into(),
            self.compositor_selected_node.to_string(),
        );
        project.project.properties.insert(
            "dynamics.enabled".into(),
            self.dynamics_enabled
                .iter()
                .map(|id| format!("{}:{}", id.slot, id.generation))
                .collect::<Vec<_>>()
                .join(","),
        );
        for (key, value) in [
            (
                "dynamics.wind.enabled",
                self.dynamics_wind.enabled.to_string(),
            ),
            ("dynamics.wind.x", self.dynamics_wind.velocity.x.to_string()),
            ("dynamics.wind.y", self.dynamics_wind.velocity.y.to_string()),
            ("dynamics.wind.z", self.dynamics_wind.velocity.z.to_string()),
            (
                "dynamics.wind.gust_strength",
                self.dynamics_wind.gust_strength.to_string(),
            ),
            (
                "dynamics.wind.gust_frequency",
                self.dynamics_wind.gust_frequency.to_string(),
            ),
            (
                "dynamics.wind.turbulence",
                self.dynamics_wind.turbulence.to_string(),
            ),
            (
                "dynamics.cloth.mass",
                self.dynamics_settings.particle_mass.to_string(),
            ),
            (
                "dynamics.cloth.stretch",
                self.dynamics_settings.stretch_compliance.to_string(),
            ),
            (
                "dynamics.cloth.bend",
                self.dynamics_settings.bend_compliance.to_string(),
            ),
            (
                "dynamics.cloth.damping",
                self.dynamics_settings.damping.to_string(),
            ),
            (
                "dynamics.cloth.iterations",
                self.dynamics_settings.iterations.to_string(),
            ),
        ] {
            project.project.properties.insert(key.into(), value);
        }
        for (id, field) in &self.dynamics_fields {
            let value = serde_json::json!({
                "name": field.name,
                "values": field.values,
                "default": field.default,
                "display_min": field.display_min,
                "display_max": field.display_max,
            });
            project.project.properties.insert(
                format!("dynamics.field.{}.{}", id.slot, id.generation),
                value.to_string(),
            );
        }
        for (id, mask) in &self.painted_masks {
            let prefix = format!("texture_paint.{}.{}", id.slot, id.generation);
            project
                .project
                .properties
                .insert(format!("{prefix}.width"), mask.width.to_string());
            project
                .project
                .properties
                .insert(format!("{prefix}.height"), mask.height.to_string());
            project
                .project
                .properties
                .insert(format!("{prefix}.pixels_r8_hex"), encode_hex(&mask.pixels));
        }
        let referenced: BTreeSet<String> = project
            .scene
            .geometry
            .iter()
            .flat_map(|(_, node)| node.components.iter())
            .filter_map(|component| match component {
                Component::Model { asset, .. } if !asset.starts_with("builtin:") => {
                    Some(asset.clone())
                }
                _ => None,
            })
            .collect();
        let mut mapping = BTreeMap::new();
        let mut files = Vec::new();
        for (index, source) in referenced.into_iter().enumerate() {
            let source_path = PathBuf::from(&source);
            if !source_path.is_file() {
                return (
                    Err(format!("referenced asset does not exist: {source}")),
                    Vec::new(),
                );
            }
            let filename = source_path
                .file_name()
                .and_then(|name| name.to_str())
                .map(safe_bundle_filename)
                .unwrap_or_else(|| "asset.bin".into());
            let archive_path = format!("assets/{index:04}/{filename}");
            mapping.insert(source, archive_path.clone());
            files.push(BundleAsset {
                source: source_path.clone(),
                archive_path: archive_path.clone(),
            });
            if self.save_cache_in_file {
                let cache_source = mip_cache_path(&source_path);
                if cache_source.is_file() {
                    files.push(BundleAsset {
                        source: cache_source,
                        archive_path: format!("{archive_path}.zfp-cache"),
                    });
                }
            }
        }
        rewrite_asset_paths(&mut project, &mapping);
        // Serialize compositor nodes
        for node in &self.compositor_nodes {
            let id = node.id;
            let kind = node.settings.kind();
            project
                .project
                .properties
                .insert(format!("compositor.node.{id}.kind"), kind.to_string());
            project.project.properties.insert(
                format!("compositor.node.{id}.x"),
                node.position.x.to_string(),
            );
            project.project.properties.insert(
                format!("compositor.node.{id}.graph_object"),
                node.object_index.to_string(),
            );
            project.project.properties.insert(
                format!("compositor.node.{id}.graph_object_name"),
                node.object_name.clone(),
            );
            if let Some(name) = &node.settings_object_name {
                project.project.properties.insert(
                    format!("compositor.node.{id}.settings_object_name"),
                    name.clone(),
                );
            }
            project.project.properties.insert(
                format!("compositor.node.{id}.y"),
                node.position.y.to_string(),
            );
            match &node.settings {
                NodeSettings::ObjectTexture {
                    object_index,
                    channel,
                } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.object_index"),
                        object_index.to_string(),
                    );
                    project
                        .project
                        .properties
                        .insert(format!("compositor.node.{id}.channel"), channel.to_string());
                }
                NodeSettings::ImageAsset { path } => {
                    if !path.is_empty() {
                        let source = PathBuf::from(path);
                        if source.is_file() {
                            let filename = source
                                .file_name()
                                .and_then(|n| n.to_str())
                                .map(safe_bundle_filename)
                                .unwrap_or_else(|| "image.png".into());
                            let archive_path = format!("assets/compositor/{id:04}-{filename}");
                            project.project.properties.insert(
                                format!("compositor.node.{id}.image_archive"),
                                archive_path.clone(),
                            );
                            files.push(BundleAsset {
                                source,
                                archive_path,
                            });
                        } else {
                            project
                                .project
                                .properties
                                .insert(format!("compositor.node.{id}.image_path"), path.clone());
                        }
                    }
                }
                NodeSettings::ConstantValue { value, color } => {
                    project
                        .project
                        .properties
                        .insert(format!("compositor.node.{id}.constant"), value.to_string());
                    project.project.properties.insert(
                        format!("compositor.node.{id}.color"),
                        format!("{},{},{}", color[0], color[1], color[2]),
                    );
                }
                NodeSettings::Remap { points, mode, .. } => {
                    let bezier_str = points
                        .iter()
                        .map(|p| format!("{},{}", p[0], p[1]))
                        .collect::<Vec<_>>()
                        .join(";");
                    project
                        .project
                        .properties
                        .insert(format!("compositor.node.{id}.bezier"), bezier_str);
                    project
                        .project
                        .properties
                        .insert(format!("compositor.node.{id}.remap_mode"), mode.to_string());
                }
                NodeSettings::TextureMath {
                    operation,
                    constant,
                } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.math_operation"),
                        operation.to_string(),
                    );
                    project.project.properties.insert(
                        format!("compositor.node.{id}.math_constant"),
                        constant.to_string(),
                    );
                }
                NodeSettings::Algebra { expression } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.algebra_expression"),
                        expression.clone(),
                    );
                }
                NodeSettings::SharpThreshold { threshold } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.threshold"),
                        threshold.to_string(),
                    );
                }
                NodeSettings::SmoothThreshold { threshold, width } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.threshold"),
                        threshold.to_string(),
                    );
                    project.project.properties.insert(
                        format!("compositor.node.{id}.threshold_width"),
                        width.to_string(),
                    );
                }
                NodeSettings::ImageFilter { filter, radius } => {
                    project
                        .project
                        .properties
                        .insert(format!("compositor.node.{id}.filter"), filter.to_string());
                    project.project.properties.insert(
                        format!("compositor.node.{id}.filter_radius"),
                        radius.to_string(),
                    );
                }
                NodeSettings::Output {
                    object_index,
                    channel,
                } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.object_index"),
                        object_index.to_string(),
                    );
                    project
                        .project
                        .properties
                        .insert(format!("compositor.node.{id}.channel"), channel.to_string());
                }
                NodeSettings::TextureCombine {
                    mode,
                    operation,
                    alpha,
                } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.combine_mode"),
                        mode.to_string(),
                    );
                    project.project.properties.insert(
                        format!("compositor.node.{id}.combine_operation"),
                        operation.to_string(),
                    );
                    project.project.properties.insert(
                        format!("compositor.node.{id}.combine_alpha"),
                        alpha.to_string(),
                    );
                }
                NodeSettings::ColorSpaceConvert { from, to } => {
                    project
                        .project
                        .properties
                        .insert(format!("compositor.node.{id}.cs_from"), from.to_string());
                    project
                        .project
                        .properties
                        .insert(format!("compositor.node.{id}.cs_to"), to.to_string());
                }
                NodeSettings::ColorDecoder => {}
                NodeSettings::Grayscale { mode } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.grayscale_mode"),
                        mode.to_string(),
                    );
                }
                NodeSettings::ColorEncoder => {}
                NodeSettings::ObjectHandle {
                    object_index,
                    label,
                    control,
                    value,
                    minimum,
                    maximum,
                    source_handle,
                } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.object_index"),
                        object_index.to_string(),
                    );
                    project
                        .project
                        .properties
                        .insert(format!("compositor.node.{id}.handle_label"), label.clone());
                    project.project.properties.insert(
                        format!("compositor.node.{id}.handle_control"),
                        control.to_string(),
                    );
                    project.project.properties.insert(
                        format!("compositor.node.{id}.handle_value"),
                        value.to_string(),
                    );
                    project.project.properties.insert(
                        format!("compositor.node.{id}.handle_minimum"),
                        minimum.to_string(),
                    );
                    project.project.properties.insert(
                        format!("compositor.node.{id}.handle_maximum"),
                        maximum.to_string(),
                    );
                    if let Some(source) = source_handle {
                        project.project.properties.insert(
                            format!("compositor.node.{id}.handle_source"),
                            source.to_string(),
                        );
                    }
                }
                NodeSettings::Time {
                    scale,
                    modulus,
                    live_update,
                } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.time_scale"),
                        scale.to_string(),
                    );
                    project.project.properties.insert(
                        format!("compositor.node.{id}.time_modulus"),
                        modulus.to_string(),
                    );
                    project.project.properties.insert(
                        format!("compositor.node.{id}.time_live_update"),
                        live_update.to_string(),
                    );
                }
                NodeSettings::Position { values } => {
                    for (axis, value) in ["x", "y", "z"].into_iter().zip(values) {
                        project.project.properties.insert(
                            format!("compositor.node.{id}.position_{axis}"),
                            value.to_string(),
                        );
                    }
                }
                NodeSettings::Rotation { degrees } => {
                    for (axis, value) in ["x", "y", "z"].into_iter().zip(degrees) {
                        project.project.properties.insert(
                            format!("compositor.node.{id}.rotation_{axis}"),
                            value.to_string(),
                        );
                    }
                }
                NodeSettings::PaintedMask { object_index }
                | NodeSettings::ObjectTransform { object_index }
                | NodeSettings::ObjectMesh { object_index } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.object_index"),
                        object_index.to_string(),
                    );
                }
                NodeSettings::MassDensity {
                    object_index,
                    base_density,
                    scale,
                    minimum_mass,
                    normalize,
                    total_mass,
                } => {
                    for (suffix, value) in [
                        ("object_index", object_index.to_string()),
                        ("base_density", base_density.to_string()),
                        ("scale", scale.to_string()),
                        ("minimum_mass", minimum_mass.to_string()),
                        ("normalize", normalize.to_string()),
                        ("total_mass", total_mass.to_string()),
                    ] {
                        project
                            .project
                            .properties
                            .insert(format!("compositor.node.{id}.{suffix}"), value);
                    }
                }
                NodeSettings::SpringMesh {
                    object_index,
                    stiffness,
                    bend_stiffness,
                    damping,
                    iterations,
                } => {
                    for (suffix, value) in [
                        ("object_index", object_index.to_string()),
                        ("stiffness", stiffness.to_string()),
                        ("bend_stiffness", bend_stiffness.to_string()),
                        ("damping", damping.to_string()),
                        ("iterations", iterations.to_string()),
                    ] {
                        project
                            .project
                            .properties
                            .insert(format!("compositor.node.{id}.{suffix}"), value);
                    }
                }
                NodeSettings::ForceField {
                    object_index,
                    formulas,
                    scale,
                } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.object_index"),
                        object_index.to_string(),
                    );
                    for (axis, formula) in ["x", "y", "z"].into_iter().zip(formulas) {
                        project.project.properties.insert(
                            format!("compositor.node.{id}.force_{axis}"),
                            formula.clone(),
                        );
                    }
                    project.project.properties.insert(
                        format!("compositor.node.{id}.force_scale"),
                        scale.to_string(),
                    );
                }
                NodeSettings::VelocityField {
                    object_index,
                    formulas,
                    scale,
                    blend,
                    mode,
                } => {
                    for (suffix, value) in [
                        ("object_index", object_index.to_string()),
                        ("velocity_scale", scale.to_string()),
                        ("velocity_blend", blend.to_string()),
                        ("velocity_mode", mode.to_string()),
                    ] {
                        project
                            .project
                            .properties
                            .insert(format!("compositor.node.{id}.{suffix}"), value);
                    }
                    for (axis, formula) in ["x", "y", "z"].into_iter().zip(formulas) {
                        project.project.properties.insert(
                            format!("compositor.node.{id}.velocity_{axis}"),
                            formula.clone(),
                        );
                    }
                }
                NodeSettings::Simulator {
                    object_index,
                    gravity,
                    time_scale,
                } => {
                    for (suffix, value) in [
                        ("object_index", object_index.to_string()),
                        ("simulator_gravity", gravity.to_string()),
                        ("simulator_time_scale", time_scale.to_string()),
                    ] {
                        project
                            .project
                            .properties
                            .insert(format!("compositor.node.{id}.{suffix}"), value);
                    }
                }
                NodeSettings::ObjectSimulator {
                    object_index,
                    mass,
                    gravity,
                    linear_velocity,
                    angular_velocity,
                    linear_damping,
                    angular_damping,
                } => {
                    for (suffix, value) in [
                        ("object_index", object_index.to_string()),
                        ("object_sim_mass", mass.to_string()),
                        ("object_sim_gravity", gravity.to_string()),
                        ("object_sim_linear_damping", linear_damping.to_string()),
                        ("object_sim_angular_damping", angular_damping.to_string()),
                    ] {
                        project
                            .project
                            .properties
                            .insert(format!("compositor.node.{id}.{suffix}"), value);
                    }
                    for (axis, value) in ["x", "y", "z"].into_iter().zip(linear_velocity) {
                        project.project.properties.insert(
                            format!("compositor.node.{id}.object_sim_velocity_{axis}"),
                            value.to_string(),
                        );
                    }
                    for (axis, value) in ["x", "y", "z"].into_iter().zip(angular_velocity) {
                        project.project.properties.insert(
                            format!("compositor.node.{id}.object_sim_angular_{axis}"),
                            value.to_string(),
                        );
                    }
                }
                NodeSettings::ForceOutput {
                    object_index,
                    force,
                } => {
                    project.project.properties.insert(
                        format!("compositor.node.{id}.object_index"),
                        object_index.to_string(),
                    );
                    for (axis, value) in ["x", "y", "z"].into_iter().zip(force) {
                        project.project.properties.insert(
                            format!("compositor.node.{id}.force_output_{axis}"),
                            value.to_string(),
                        );
                    }
                }
                NodeSettings::Engine {
                    object_index,
                    throttle,
                    torque,
                    reverse,
                    auto_braking,
                    max_rpm,
                } => {
                    for (suffix, value) in [
                        ("object_index", object_index.to_string()),
                        ("engine_throttle", throttle.to_string()),
                        ("engine_torque", torque.to_string()),
                        ("engine_reverse", reverse.to_string()),
                        ("engine_auto_braking", auto_braking.to_string()),
                        ("engine_max_rpm", max_rpm.to_string()),
                    ] {
                        project
                            .project
                            .properties
                            .insert(format!("compositor.node.{id}.{suffix}"), value);
                    }
                }
                NodeSettings::Debug => {}
            }
        }
        for (index, (from_id, from_output, to_id, to_input)) in
            self.compositor_links.iter().enumerate()
        {
            project.project.properties.insert(
                format!("compositor.link.{index:04}"),
                format!("{from_id},{from_output},{to_id},{to_input}"),
            );
        }
        (Ok(project), files)
    }

    fn apply_project_properties(&mut self, project: &ProjectFile) {
        let properties = &project.project.properties;
        let number = |key: &str| {
            properties
                .get(key)
                .and_then(|value| value.parse::<f32>().ok())
        };
        if let Some(value) = number("editor.grid_spacing").filter(|value| *value > 0.0) {
            self.grid_spacing = value;
        }
        if let Some(value) = properties
            .get("editor.show_grid")
            .and_then(|value| value.parse::<bool>().ok())
        {
            self.show_grid = value;
        }
        if let Some(value) = properties
            .get("editor.global_light_enabled")
            .and_then(|value| value.parse::<bool>().ok())
        {
            self.global_light_enabled = value;
        }
        if let Some(value) = properties
            .get("editor.global_shadow_resolution")
            .and_then(|value| value.parse::<u32>().ok())
        {
            self.global_shadow_resolution = sanitize_shadow_resolution(value);
        }
        self.shadow_quality = properties
            .get("editor.shadow_quality")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(self.shadow_quality)
            .min(4);
        self.shadow_blur_radius = properties
            .get("editor.shadow_blur_radius")
            .and_then(|value| value.parse::<usize>().ok())
            .unwrap_or(self.shadow_blur_radius)
            .min(4);
        self.target_fps = properties
            .get("editor.target_fps")
            .and_then(|value| value.parse::<u32>().ok())
            .map(sanitize_target_fps)
            .unwrap_or(self.target_fps);
        self.next_viewport_frame = Instant::now();
        self.save_cache_in_file = properties
            .get("editor.save_cache_in_file")
            .and_then(|value| value.parse::<bool>().ok())
            .unwrap_or(self.save_cache_in_file);
        self.camera_yaw = number("editor.camera_yaw").unwrap_or(self.camera_yaw);
        self.camera_pitch = number("editor.camera_pitch").unwrap_or(self.camera_pitch);
        self.camera_zoom = number("editor.camera_zoom")
            .filter(|value| *value > 0.0)
            .unwrap_or(self.camera_zoom);
        self.camera_target = CoreVec3::new(
            number("editor.camera_target_x").unwrap_or(self.camera_target.x),
            number("editor.camera_target_y").unwrap_or(self.camera_target.y),
            number("editor.camera_target_z").unwrap_or(self.camera_target.z),
        );
        self.projection_mode = match properties.get("editor.projection").map(String::as_str) {
            Some("perspective") => ProjectionMode::Perspective,
            Some("orthographic") => ProjectionMode::Orthographic,
            _ => self.projection_mode,
        };
        self.viewport_mode = match properties.get("editor.viewport_mode").map(String::as_str) {
            Some("matte") => ViewportMode::Shaded,
            Some("matte + edges") => ViewportMode::ShadedEdges,
            Some("wireframe") => ViewportMode::Wireframe,
            _ => self.viewport_mode,
        };
        self.workspace_tab = match properties.get("editor.workspace").map(String::as_str) {
            Some("scene") => WorkspaceTab::Scene,
            Some("nodes" | "compositing") => WorkspaceTab::Compositing,
            Some("dynamics") => WorkspaceTab::Compositing,
            _ => self.workspace_tab,
        };
        self.render_device = match properties
            .get("editor.device")
            .or_else(|| properties.get("compositor.backend"))
            .map(String::as_str)
        {
            Some("cpu") => RenderDevice::Cpu,
            Some("vulkan") => RenderDevice::Vulkan,
            _ => self.render_device,
        };
        self.dynamics_wind.enabled = properties
            .get("dynamics.wind.enabled")
            .and_then(|value| value.parse().ok())
            .unwrap_or(self.dynamics_wind.enabled);
        self.dynamics_wind.velocity = CoreVec3::new(
            number("dynamics.wind.x").unwrap_or(self.dynamics_wind.velocity.x),
            number("dynamics.wind.y").unwrap_or(self.dynamics_wind.velocity.y),
            number("dynamics.wind.z").unwrap_or(self.dynamics_wind.velocity.z),
        );
        self.dynamics_wind.gust_strength =
            number("dynamics.wind.gust_strength").unwrap_or(self.dynamics_wind.gust_strength);
        self.dynamics_wind.gust_frequency =
            number("dynamics.wind.gust_frequency").unwrap_or(self.dynamics_wind.gust_frequency);
        self.dynamics_wind.turbulence =
            number("dynamics.wind.turbulence").unwrap_or(self.dynamics_wind.turbulence);
        self.dynamics_settings.particle_mass =
            number("dynamics.cloth.mass").unwrap_or(self.dynamics_settings.particle_mass);
        self.dynamics_settings.stretch_compliance =
            number("dynamics.cloth.stretch").unwrap_or(self.dynamics_settings.stretch_compliance);
        self.dynamics_settings.bend_compliance =
            number("dynamics.cloth.bend").unwrap_or(self.dynamics_settings.bend_compliance);
        self.dynamics_settings.damping =
            number("dynamics.cloth.damping").unwrap_or(self.dynamics_settings.damping);
        self.dynamics_settings.iterations = properties
            .get("dynamics.cloth.iterations")
            .and_then(|value| value.parse().ok())
            .unwrap_or(self.dynamics_settings.iterations);
        self.dynamics_fields.clear();
        for (key, encoded) in properties {
            let Some(identity) = key.strip_prefix("dynamics.field.") else {
                continue;
            };
            let Some((slot, generation)) = identity.split_once('.') else {
                continue;
            };
            let (Ok(slot), Ok(generation)) = (slot.parse(), generation.parse()) else {
                continue;
            };
            let Ok(value) = serde_json::from_str::<serde_json::Value>(encoded) else {
                continue;
            };
            let Some(values) = value.get("values").and_then(|values| values.as_array()) else {
                continue;
            };
            let values = values
                .iter()
                .filter_map(|value| value.as_f64().map(|value| value as f32))
                .collect::<Vec<_>>();
            if values.is_empty() || values.iter().any(|value| !value.is_finite()) {
                continue;
            }
            self.dynamics_fields.insert(
                NodeId { slot, generation },
                MeshScalarField {
                    name: value
                        .get("name")
                        .and_then(|name| name.as_str())
                        .unwrap_or("dynamics.mobility")
                        .into(),
                    values,
                    default: value
                        .get("default")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0) as f32,
                    display_min: value
                        .get("display_min")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(0.0) as f32,
                    display_max: value
                        .get("display_max")
                        .and_then(|value| value.as_f64())
                        .unwrap_or(1.0) as f32,
                    revision: 1,
                },
            );
        }
        self.dynamics_enabled = properties
            .get("dynamics.enabled")
            .into_iter()
            .flat_map(|value| value.split(','))
            .filter_map(|identity| {
                let (slot, generation) = identity.split_once(':')?;
                Some(NodeId {
                    slot: slot.parse().ok()?,
                    generation: generation.parse().ok()?,
                })
            })
            .collect();
        self.painted_masks.clear();
        for (key, encoded) in properties {
            let Some(identity) = key
                .strip_prefix("texture_paint.")
                .and_then(|value| value.strip_suffix(".pixels_r8_hex"))
            else {
                continue;
            };
            let Some((slot, generation)) = identity.split_once('.') else {
                continue;
            };
            let (Ok(slot), Ok(generation)) = (slot.parse(), generation.parse()) else {
                continue;
            };
            let prefix = format!("texture_paint.{slot}.{generation}");
            let Some(width) = properties
                .get(&format!("{prefix}.width"))
                .and_then(|value| value.parse::<u32>().ok())
                .filter(|value| (1..=4096).contains(value))
            else {
                continue;
            };
            let Some(height) = properties
                .get(&format!("{prefix}.height"))
                .and_then(|value| value.parse::<u32>().ok())
                .filter(|value| (1..=4096).contains(value))
            else {
                continue;
            };
            let Some(pixels) =
                decode_hex(encoded).filter(|pixels| pixels.len() == (width * height) as usize)
            else {
                continue;
            };
            self.painted_masks.insert(
                NodeId { slot, generation },
                PaintedMask {
                    width,
                    height,
                    pixels,
                    revision: 1,
                },
            );
        }
        self.dynamics_cloth.clear();
        let unsigned = |key: &str| {
            properties
                .get(key)
                .and_then(|value| value.parse::<usize>().ok())
        };
        // Load compositor nodes from new per-instance format
        // Collect all node IDs that exist
        let mut node_ids: BTreeSet<usize> = BTreeSet::new();
        for key in properties.keys() {
            if let Some(rest) = key.strip_prefix("compositor.node.") {
                if let Some(dot_pos) = rest.find('.') {
                    if let Ok(id) = rest[..dot_pos].parse::<usize>() {
                        node_ids.insert(id);
                    }
                }
            }
        }
        if !node_ids.is_empty() {
            let mut loaded_nodes: Vec<CompositorNode> = Vec::new();
            for id in &node_ids {
                let id = *id;
                let kind = match properties
                    .get(&format!("compositor.node.{id}.kind"))
                    .and_then(|v| v.parse::<usize>().ok())
                {
                    Some(k) => k,
                    None => continue,
                };
                let x = properties
                    .get(&format!("compositor.node.{id}.x"))
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.0);
                let y = properties
                    .get(&format!("compositor.node.{id}.y"))
                    .and_then(|v| v.parse::<f32>().ok())
                    .unwrap_or(0.0);
                let get_usize = |key: &str| {
                    properties
                        .get(key)
                        .and_then(|v| v.parse::<usize>().ok())
                        .unwrap_or(0)
                };
                let get_f32 = |key: &str| {
                    properties
                        .get(key)
                        .and_then(|v| v.parse::<f32>().ok())
                        .unwrap_or(0.0)
                };
                let settings = match kind {
                    0 => NodeSettings::ObjectTexture {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        channel: get_usize(&format!("compositor.node.{id}.channel")),
                    },
                    1 => {
                        let path = if let Some(p) =
                            properties.get(&format!("compositor.node.{id}.image_archive"))
                        {
                            p.clone()
                        } else {
                            properties
                                .get(&format!("compositor.node.{id}.image_path"))
                                .cloned()
                                .unwrap_or_default()
                        };
                        NodeSettings::ImageAsset { path }
                    }
                    2 => {
                        let value = get_f32(&format!("compositor.node.{id}.constant"));
                        let color = properties
                            .get(&format!("compositor.node.{id}.color"))
                            .and_then(|v| parse_compositor_color(v))
                            .unwrap_or([0.5, 0.5, 0.5]);
                        NodeSettings::ConstantValue { value, color }
                    }
                    3 => {
                        let points = properties
                            .get(&format!("compositor.node.{id}.bezier"))
                            .and_then(|v| parse_compositor_bezier(v))
                            .unwrap_or_else(|| {
                                vec![[0.0, 0.0], [0.33, 0.33], [0.67, 0.67], [1.0, 1.0]]
                            });
                        let mode = get_usize(&format!("compositor.node.{id}.remap_mode"));
                        NodeSettings::Remap {
                            points,
                            mode,
                            selected: None,
                        }
                    }
                    4 => NodeSettings::TextureMath {
                        operation: get_usize(&format!("compositor.node.{id}.math_operation")),
                        constant: get_f32(&format!("compositor.node.{id}.math_constant")),
                    },
                    5 => NodeSettings::SharpThreshold {
                        threshold: get_f32(&format!("compositor.node.{id}.threshold")),
                    },
                    6 => NodeSettings::SmoothThreshold {
                        threshold: get_f32(&format!("compositor.node.{id}.threshold")),
                        width: get_f32(&format!("compositor.node.{id}.threshold_width")),
                    },
                    7 => NodeSettings::ImageFilter {
                        filter: get_usize(&format!("compositor.node.{id}.filter")),
                        radius: get_f32(&format!("compositor.node.{id}.filter_radius")),
                    },
                    8 => NodeSettings::Output {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        channel: get_usize(&format!("compositor.node.{id}.channel")),
                    },
                    9 => NodeSettings::TextureCombine {
                        mode: get_usize(&format!("compositor.node.{id}.combine_mode")),
                        operation: get_usize(&format!("compositor.node.{id}.combine_operation")),
                        alpha: get_f32(&format!("compositor.node.{id}.combine_alpha")),
                    },
                    10 => NodeSettings::ColorSpaceConvert {
                        from: get_usize(&format!("compositor.node.{id}.cs_from")),
                        to: get_usize(&format!("compositor.node.{id}.cs_to")),
                    },
                    11 => NodeSettings::ColorDecoder,
                    12 => NodeSettings::Grayscale {
                        mode: get_usize(&format!("compositor.node.{id}.grayscale_mode")),
                    },
                    13 => NodeSettings::ColorEncoder,
                    14 => NodeSettings::ObjectHandle {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        label: properties
                            .get(&format!("compositor.node.{id}.handle_label"))
                            .cloned()
                            .unwrap_or_else(|| "Compositor value".into()),
                        control: get_usize(&format!("compositor.node.{id}.handle_control")),
                        value: get_f32(&format!("compositor.node.{id}.handle_value")),
                        minimum: properties
                            .get(&format!("compositor.node.{id}.handle_minimum"))
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(0.0),
                        maximum: properties
                            .get(&format!("compositor.node.{id}.handle_maximum"))
                            .and_then(|v| v.parse().ok())
                            .unwrap_or(1.0),
                        source_handle: properties
                            .get(&format!("compositor.node.{id}.handle_source"))
                            .and_then(|value| value.parse().ok()),
                    },
                    15 => NodeSettings::Time {
                        scale: properties
                            .get(&format!("compositor.node.{id}.time_scale"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(1.0),
                        modulus: properties
                            .get(&format!("compositor.node.{id}.time_modulus"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(0.0),
                        live_update: properties
                            .get(&format!("compositor.node.{id}.time_live_update"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(true),
                    },
                    16 => NodeSettings::Debug,
                    17 => NodeSettings::ObjectTransform {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                    },
                    18 => NodeSettings::ObjectMesh {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                    },
                    19 => NodeSettings::PaintedMask {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                    },
                    20 => NodeSettings::MassDensity {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        base_density: properties
                            .get(&format!("compositor.node.{id}.base_density"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(0.15),
                        scale: properties
                            .get(&format!("compositor.node.{id}.scale"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(1.0),
                        minimum_mass: properties
                            .get(&format!("compositor.node.{id}.minimum_mass"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(1.0e-6),
                        normalize: properties
                            .get(&format!("compositor.node.{id}.normalize"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(false),
                        total_mass: properties
                            .get(&format!("compositor.node.{id}.total_mass"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(1.0),
                    },
                    21 => NodeSettings::SpringMesh {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        stiffness: get_f32(&format!("compositor.node.{id}.stiffness")),
                        bend_stiffness: get_f32(&format!("compositor.node.{id}.bend_stiffness")),
                        damping: get_f32(&format!("compositor.node.{id}.damping")),
                        iterations: get_usize(&format!("compositor.node.{id}.iterations")).max(1),
                    },
                    22 => NodeSettings::ForceField {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        formulas: ["x", "y", "z"].map(|axis| {
                            properties
                                .get(&format!("compositor.node.{id}.force_{axis}"))
                                .cloned()
                                .unwrap_or_else(|| "0".into())
                        }),
                        scale: properties
                            .get(&format!("compositor.node.{id}.force_scale"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(1.0),
                    },
                    23 => NodeSettings::VelocityField {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        formulas: ["x", "y", "z"].map(|axis| {
                            properties
                                .get(&format!("compositor.node.{id}.velocity_{axis}"))
                                .cloned()
                                .unwrap_or_else(|| "0".into())
                        }),
                        scale: properties
                            .get(&format!("compositor.node.{id}.velocity_scale"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(1.0),
                        blend: properties
                            .get(&format!("compositor.node.{id}.velocity_blend"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(1.0),
                        // Simulator velocity injection is always additive;
                        // normalize old Set/Approach projects to Add.
                        mode: 1,
                    },
                    24 => NodeSettings::Simulator {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        gravity: properties
                            .get(&format!("compositor.node.{id}.simulator_gravity"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(true),
                        time_scale: properties
                            .get(&format!("compositor.node.{id}.simulator_time_scale"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(1.0),
                    },
                    25 => NodeSettings::Position {
                        values: ["x", "y", "z"]
                            .map(|axis| get_f32(&format!("compositor.node.{id}.position_{axis}"))),
                    },
                    26 => NodeSettings::Rotation {
                        degrees: ["x", "y", "z"]
                            .map(|axis| get_f32(&format!("compositor.node.{id}.rotation_{axis}"))),
                    },
                    27 => NodeSettings::ObjectSimulator {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        mass: properties
                            .get(&format!("compositor.node.{id}.object_sim_mass"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(1.0),
                        gravity: properties
                            .get(&format!("compositor.node.{id}.object_sim_gravity"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(true),
                        linear_velocity: ["x", "y", "z"].map(|axis| {
                            get_f32(&format!("compositor.node.{id}.object_sim_velocity_{axis}"))
                        }),
                        angular_velocity: ["x", "y", "z"].map(|axis| {
                            get_f32(&format!("compositor.node.{id}.object_sim_angular_{axis}"))
                        }),
                        linear_damping: properties
                            .get(&format!("compositor.node.{id}.object_sim_linear_damping"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(0.05),
                        angular_damping: properties
                            .get(&format!("compositor.node.{id}.object_sim_angular_damping"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(0.05),
                    },
                    28 => NodeSettings::ForceOutput {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        force: ["x", "y", "z"].map(|axis| {
                            get_f32(&format!("compositor.node.{id}.force_output_{axis}"))
                        }),
                    },
                    29 => NodeSettings::Engine {
                        object_index: get_usize(&format!("compositor.node.{id}.object_index")),
                        throttle: properties
                            .get(&format!("compositor.node.{id}.engine_throttle"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(0.0),
                        torque: properties
                            .get(&format!("compositor.node.{id}.engine_torque"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(100.0),
                        reverse: properties
                            .get(&format!("compositor.node.{id}.engine_reverse"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(false),
                        auto_braking: properties
                            .get(&format!("compositor.node.{id}.engine_auto_braking"))
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(false),
                        max_rpm: properties
                            .get(&format!("compositor.node.{id}.engine_max_rpm"))
                            .or_else(|| {
                                properties
                                    .get(&format!("compositor.node.{id}.engine_throttle_rpm_ratio"))
                            })
                            .and_then(|value| value.parse().ok())
                            .unwrap_or(6_000.0),
                    },
                    30 => NodeSettings::Algebra {
                        expression: properties
                            .get(&format!("compositor.node.{id}.algebra_expression"))
                            .cloned()
                            .unwrap_or_else(|| "x + y * z".into()),
                    },
                    _ => continue,
                };
                let legacy_graph_index = properties
                    .get(&format!("compositor.node.{id}.graph_object"))
                    .and_then(|value| value.parse().ok())
                    .unwrap_or_else(|| settings.object_index().unwrap_or(0));
                let object_name = properties
                    .get(&format!("compositor.node.{id}.graph_object_name"))
                    .cloned()
                    .or_else(|| {
                        self.scene
                            .tree
                            .iter()
                            .nth(legacy_graph_index)
                            .map(|(_, object)| object.name.clone())
                    })
                    .unwrap_or_default();
                let settings_object_name = properties
                    .get(&format!("compositor.node.{id}.settings_object_name"))
                    .cloned()
                    .or_else(|| {
                        settings.object_index().and_then(|index| {
                            self.scene
                                .tree
                                .iter()
                                .nth(index)
                                .map(|(_, object)| object.name.clone())
                        })
                    });
                loaded_nodes.push(CompositorNode {
                    id,
                    object_index: legacy_graph_index,
                    object_name,
                    settings_object_name,
                    settings,
                    position: Vec2::new(x, y),
                });
            }
            self.compositor_nodes = loaded_nodes;
            // Set next_id
            let max_id = self
                .compositor_nodes
                .iter()
                .map(|n| n.id)
                .max()
                .unwrap_or(0);
            self.compositor_next_id = properties
                .get("compositor.next_id")
                .and_then(|v| v.parse::<usize>().ok())
                .unwrap_or(max_id + 1);
        }
        // Load selected
        if let Some(sel) = unsigned("compositor.selected") {
            if self.compositor_nodes.iter().any(|n| n.id == sel) {
                self.compositor_selected_node = sel;
            } else {
                self.compositor_selected_node = self
                    .compositor_nodes
                    .iter()
                    .find(|n| matches!(n.settings, NodeSettings::Output { .. }))
                    .map(|n| n.id)
                    .unwrap_or(0);
            }
        }
        // Load links - support both 3-number (old) and 4-number (new) formats
        let stored_links: Vec<_> = properties
            .iter()
            .filter_map(|(key, value)| {
                key.starts_with("compositor.link.")
                    .then(|| parse_compositor_link(value))
                    .flatten()
            })
            .filter(|(from_id, _from_out, to_id, _to_input)| {
                from_id != to_id
                    && self.compositor_nodes.iter().any(|n| n.id == *from_id)
                    && self.compositor_nodes.iter().any(|n| n.id == *to_id)
            })
            .collect();
        if properties
            .keys()
            .any(|key| key.starts_with("compositor.link."))
        {
            self.compositor_links = stored_links;
        }
        if !properties.keys().any(|key| key.ends_with(".graph_object")) {
            let legacy_outputs = self
                .compositor_nodes
                .iter()
                .filter_map(|node| match node.settings {
                    NodeSettings::Output { object_index, .. } => Some((node.id, object_index)),
                    _ => None,
                })
                .collect::<Vec<_>>();
            for (output_id, object_index) in legacy_outputs {
                let mut pending = vec![output_id];
                let mut visited = BTreeSet::new();
                while let Some(node_id) = pending.pop() {
                    if !visited.insert(node_id) {
                        continue;
                    }
                    if let Some(node) = self.compositor_nodes.iter_mut().find(|n| n.id == node_id) {
                        node.object_index = object_index;
                        node.object_name = self
                            .scene
                            .tree
                            .iter()
                            .nth(object_index)
                            .map(|(_, object)| object.name.clone())
                            .unwrap_or_default();
                    }
                    pending.extend(
                        self.compositor_links
                            .iter()
                            .filter_map(|&(from, _, to, _)| (to == node_id).then_some(from)),
                    );
                }
            }
        }
    }

    fn compositor_node_controls_ui(&mut self, node_id: usize, ui: &mut egui::Ui) {
        let mut changed = false;
        let mut open_browse = false;
        let kind = self
            .compositor_nodes
            .iter()
            .find(|n| n.id == node_id)
            .map(|n| n.settings.kind())
            .unwrap_or(usize::MAX);
        match kind {
            0 => {
                let objects: Vec<String> = self
                    .scene
                    .tree
                    .iter()
                    .map(|(_, n)| n.name.clone())
                    .collect();
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::ObjectTexture {
                    ref mut object_index,
                    ref mut channel,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Object");
                egui::ComboBox::from_id_salt(("compositor_object", node_id))
                    .selected_text(
                        objects
                            .get(*object_index)
                            .map(String::as_str)
                            .unwrap_or("No scene objects"),
                    )
                    .show_ui(ui, |ui| {
                        for (index, name) in objects.iter().enumerate() {
                            changed |= ui
                                .selectable_value(object_index, index, name.as_str())
                                .changed();
                        }
                    });
                ui.label("Texture channel");
                let channels = [
                    "Base Color",
                    "Normal",
                    "Roughness",
                    "Metallic",
                    "Emissive",
                    "Occlusion",
                ];
                egui::ComboBox::from_id_salt(("compositor_texture_channel", node_id))
                    .selected_text(channels[*channel])
                    .show_ui(ui, |ui| {
                        for (index, ch) in channels.iter().enumerate() {
                            changed |= ui.selectable_value(channel, index, *ch).changed();
                        }
                    });
            }
            1 => {
                let (current_path, dialog_open) = {
                    let pos = self
                        .compositor_nodes
                        .iter()
                        .position(|n| n.id == node_id)
                        .unwrap();
                    let NodeSettings::ImageAsset { ref path } = self.compositor_nodes[pos].settings
                    else {
                        return;
                    };
                    (path.clone(), self.compositor_image_dialog_result.is_some())
                };
                ui.label("External texture");
                ui.horizontal(|ui| {
                    let mut path_display = current_path.clone();
                    ui.add_enabled(
                        false,
                        egui::TextEdit::singleline(&mut path_display)
                            .hint_text("Choose PNG or JPEG")
                            .desired_width(125.0),
                    );
                    if ui
                        .add_enabled(!dialog_open, egui::Button::new("Browse…"))
                        .clicked()
                    {
                        open_browse = true;
                        self.compositor_image_dialog_target = Some(node_id);
                    }
                });
                ui.label("Color space");
                ui.label("Auto (from image metadata)");
            }
            2 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::ConstantValue {
                    ref mut value,
                    ref mut color,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Scalar");
                changed |= ui.add(egui::Slider::new(value, 0.0..=1.0)).changed();
                ui.label("RGB color");
                changed |= ui.color_edit_button_rgb(color).changed();
                ui.horizontal(|ui| {
                    for channel in color.iter_mut() {
                        changed |= ui
                            .add(egui::DragValue::new(channel).range(0.0..=1.0).speed(0.01))
                            .changed();
                    }
                });
            }
            3 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::Remap {
                    ref mut points,
                    ref mut mode,
                    ref mut selected,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Interpolation");
                let modes = ["Polyline", "Bézier"];
                egui::ComboBox::from_id_salt(("compositor_remap_mode", node_id))
                    .selected_text(modes[*mode])
                    .show_ui(ui, |ui| {
                        for (index, m) in modes.iter().enumerate() {
                            changed |= ui.selectable_value(mode, index, *m).changed();
                        }
                    });
                ui.label("Curve");
                if points.len() == 4 {
                    changed |= remap_curve_editor(ui, points, *mode == 1);
                } else {
                    // For non-standard point counts show basic info
                    ui.small(format!("{} control points", points.len()));
                }
                let can_delete = selected
                    .map(|idx| idx > 0 && idx + 1 < points.len())
                    .unwrap_or(false);
                if ui
                    .add_enabled(can_delete, egui::Button::new("Remove selected point"))
                    .clicked()
                {
                    let idx = selected.take().unwrap();
                    points.remove(idx);
                    changed = true;
                }
            }
            4 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::TextureMath {
                    ref mut operation,
                    ref mut constant,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                let operations = [
                    "Add",
                    "Subtract",
                    "Multiply",
                    "Divide",
                    "Power",
                    "Minimum",
                    "Maximum",
                    "Absolute Difference",
                ];
                ui.label("Operation");
                egui::ComboBox::from_id_salt(("compositor_math_operation", node_id))
                    .selected_text(operations[*operation])
                    .show_ui(ui, |ui| {
                        for (index, op) in operations.iter().enumerate() {
                            changed |= ui.selectable_value(operation, index, *op).changed();
                        }
                    });
                ui.label("Fallback value");
                changed |= ui.add(egui::DragValue::new(constant).speed(0.01)).changed();
            }
            30 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::Algebra { ref mut expression } =
                    self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Expression");
                changed |= ui.text_edit_singleline(expression).changed();
                ui.small("Variables: x, y, z");
                match compositor_graph::compile_algebra_expression(expression) {
                    Ok(program) => {
                        ui.small(
                            RichText::new(format!("Valid · {} operations", program.len()))
                                .color(Color32::from_rgb(100, 190, 125)),
                        );
                    }
                    Err(error) => {
                        ui.small(RichText::new(error).color(Color32::from_rgb(225, 105, 95)));
                    }
                }
            }
            5 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::SharpThreshold { ref mut threshold } =
                    self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Threshold");
                changed |= ui.add(egui::Slider::new(threshold, 0.0..=1.0)).changed();
            }
            6 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::SmoothThreshold {
                    ref mut threshold,
                    ref mut width,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Threshold");
                changed |= ui.add(egui::Slider::new(threshold, 0.0..=1.0)).changed();
                ui.label("Transition width");
                changed |= ui.add(egui::Slider::new(width, 0.0..=1.0)).changed();
            }
            7 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::ImageFilter {
                    ref mut filter,
                    ref mut radius,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                let filters = [
                    "Gaussian Blur",
                    "Box Blur",
                    "Sharpen",
                    "Sobel Edge",
                    "Median",
                    "Dilate",
                    "Erode",
                ];
                ui.label("Filter");
                egui::ComboBox::from_id_salt(("compositor_filter", node_id))
                    .selected_text(filters[*filter])
                    .show_ui(ui, |ui| {
                        for (index, f) in filters.iter().enumerate() {
                            changed |= ui.selectable_value(filter, index, *f).changed();
                        }
                    });
                ui.label("Radius");
                changed |= ui
                    .add(egui::DragValue::new(radius).range(0.0..=128.0).speed(0.25))
                    .changed();
            }
            8 => {
                let objects: Vec<String> = self
                    .scene
                    .tree
                    .iter()
                    .map(|(_, n)| n.name.clone())
                    .collect();
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::Output {
                    ref mut object_index,
                    ref mut channel,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Target object");
                ui.strong(
                    objects
                        .get(*object_index)
                        .map(String::as_str)
                        .unwrap_or("Object unavailable"),
                );
                let channels = [
                    "Base Color",
                    "Normal",
                    "Roughness",
                    "Metallic",
                    "Emissive",
                    "Occlusion",
                ];
                ui.label("Overwrite texture");
                egui::ComboBox::from_id_salt(("compositor_output_channel", node_id))
                    .selected_text(channels[*channel])
                    .show_ui(ui, |ui| {
                        for (index, ch) in channels.iter().enumerate() {
                            changed |= ui.selectable_value(channel, index, *ch).changed();
                        }
                    });
            }
            9 => {
                let (mode, operation, alpha) = {
                    let pos = self
                        .compositor_nodes
                        .iter()
                        .position(|n| n.id == node_id)
                        .unwrap();
                    let NodeSettings::TextureCombine {
                        mode,
                        operation,
                        alpha,
                    } = self.compositor_nodes[pos].settings
                    else {
                        return;
                    };
                    (mode, operation, alpha)
                };
                let alpha_connected = self
                    .compositor_links
                    .iter()
                    .any(|&(_, _, to, input)| to == node_id && input == 2);
                let modes = ["Algebra", "Mix"];
                let mut new_mode = mode;
                let mut new_operation = operation;
                let mut new_alpha = alpha;
                ui.label("Combine mode");
                egui::ComboBox::from_id_salt(("compositor_combine_mode", node_id))
                    .selected_text(modes[mode])
                    .show_ui(ui, |ui| {
                        for (index, m) in modes.iter().enumerate() {
                            if ui.selectable_value(&mut new_mode, index, *m).changed() {
                                changed = true;
                            }
                        }
                    });
                if new_mode == 0 {
                    let operations = [
                        "Add",
                        "Subtract",
                        "Multiply",
                        "Divide",
                        "Power",
                        "Minimum",
                        "Maximum",
                        "Absolute Difference",
                    ];
                    ui.label("Algebra operation");
                    egui::ComboBox::from_id_salt(("compositor_combine_operation", node_id))
                        .selected_text(operations[operation])
                        .show_ui(ui, |ui| {
                            for (index, op) in operations.iter().enumerate() {
                                if ui
                                    .selectable_value(&mut new_operation, index, *op)
                                    .changed()
                                {
                                    changed = true;
                                }
                            }
                        });
                } else {
                    ui.label("Alpha");
                    if ui
                        .add_enabled(
                            !alpha_connected,
                            egui::DragValue::new(&mut new_alpha)
                                .range(0.0..=1.0)
                                .speed(0.01),
                        )
                        .changed()
                    {
                        changed = true;
                    }
                }
                if changed {
                    let pos = self
                        .compositor_nodes
                        .iter()
                        .position(|n| n.id == node_id)
                        .unwrap();
                    let NodeSettings::TextureCombine {
                        mode: ref mut m,
                        operation: ref mut op,
                        alpha: ref mut a,
                    } = self.compositor_nodes[pos].settings
                    else {
                        return;
                    };
                    *m = new_mode;
                    *op = new_operation;
                    *a = new_alpha;
                }
                if new_mode != mode && new_mode == 0 {
                    self.compositor_links
                        .retain(|&(_, _, to, input)| to != node_id || input != 2);
                }
            }
            10 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::ColorSpaceConvert {
                    ref mut from,
                    ref mut to,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                let spaces = ["sRGB", "Linear RGB"];
                ui.label("From");
                egui::ComboBox::from_id_salt(("compositor_color_space_from", node_id))
                    .selected_text(spaces[*from])
                    .show_ui(ui, |ui| {
                        for (index, space) in spaces.iter().enumerate() {
                            changed |= ui.selectable_value(from, index, *space).changed();
                        }
                    });
                ui.label("To");
                egui::ComboBox::from_id_salt(("compositor_color_space_to", node_id))
                    .selected_text(spaces[*to])
                    .show_ui(ui, |ui| {
                        for (index, space) in spaces.iter().enumerate() {
                            changed |= ui.selectable_value(to, index, *space).changed();
                        }
                    });
            }
            11 => {}
            12 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::Grayscale { ref mut mode } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                let modes = ["Luminance (Rec. 709)", "Average", "Lightness"];
                ui.label("Conversion");
                egui::ComboBox::from_id_salt(("compositor_grayscale_mode", node_id))
                    .selected_text(modes[*mode])
                    .show_ui(ui, |ui| {
                        for (index, m) in modes.iter().enumerate() {
                            changed |= ui.selectable_value(mode, index, *m).changed();
                        }
                    });
            }
            13 => {}
            14 => {
                let objects: Vec<String> = self
                    .scene
                    .tree
                    .iter()
                    .map(|(_, n)| n.name.clone())
                    .collect();
                let handles = self
                    .compositor_nodes
                    .iter()
                    .filter_map(|node| match &node.settings {
                        NodeSettings::ObjectHandle { label, .. } if node.id != node_id => (!self
                            .object_handle_depends_on(node.id, node_id))
                        .then(|| (node.id, label.clone())),
                        _ => None,
                    })
                    .collect::<Vec<_>>();
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|n| n.id == node_id)
                    .unwrap();
                let NodeSettings::ObjectHandle {
                    ref mut object_index,
                    ref mut label,
                    ref mut control,
                    ref mut value,
                    ref mut minimum,
                    ref mut maximum,
                    ref mut source_handle,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Scene object");
                egui::ComboBox::from_id_salt(("compositor_handle_object", node_id))
                    .selected_text(
                        objects
                            .get(*object_index)
                            .map(String::as_str)
                            .unwrap_or("No scene objects"),
                    )
                    .show_ui(ui, |ui| {
                        for (index, name) in objects.iter().enumerate() {
                            changed |= ui.selectable_value(object_index, index, name).changed();
                        }
                    });
                ui.label("Label");
                changed |= ui.text_edit_singleline(label).changed();
                ui.label("Scene control");
                egui::ComboBox::from_id_salt(("compositor_handle_control", node_id))
                    .selected_text(if *control == 0 {
                        "Slider"
                    } else {
                        "Number field"
                    })
                    .show_ui(ui, |ui| {
                        changed |= ui.selectable_value(control, 0, "Slider").changed();
                        changed |= ui.selectable_value(control, 1, "Number field").changed();
                    });
                ui.horizontal(|ui| {
                    ui.label("Range");
                    changed |= ui.add(egui::DragValue::new(minimum).speed(0.01)).changed();
                    changed |= ui.add(egui::DragValue::new(maximum).speed(0.01)).changed();
                });
                ui.label("Copy value from");
                egui::ComboBox::from_id_salt(("compositor_handle_source", node_id))
                    .selected_text(
                        source_handle
                            .and_then(|source| {
                                handles
                                    .iter()
                                    .find(|(id, _)| *id == source)
                                    .map(|(_, label)| label.as_str())
                            })
                            .unwrap_or("Own value"),
                    )
                    .show_ui(ui, |ui| {
                        changed |= ui
                            .selectable_value(source_handle, None, "Own value")
                            .changed();
                        for (source, source_label) in &handles {
                            changed |= ui
                                .selectable_value(
                                    source_handle,
                                    Some(*source),
                                    format!("{source_label}  ·  #{source}"),
                                )
                                .changed();
                        }
                    });
                if *maximum < *minimum {
                    std::mem::swap(minimum, maximum);
                    changed = true;
                }
                *value = value.clamp(*minimum, *maximum);
            }
            15 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::Time {
                    ref mut scale,
                    ref mut modulus,
                    ref mut live_update,
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                changed |= ui.checkbox(live_update, "Live update").changed();
                ui.label("Scale factor");
                changed |= ui.add(egui::DragValue::new(scale).speed(0.01)).changed();
                ui.label("Modulus");
                changed |= ui
                    .add(egui::DragValue::new(modulus).speed(0.01))
                    .on_hover_text("Values at or below zero disable wrapping.")
                    .changed();
                let value = scaled_modulated_time(
                    self.compositor_clock_started.elapsed().as_secs_f32(),
                    *scale,
                    *modulus,
                );
                ui.monospace(format!("Time: {value:.3} s"));
            }
            16 => {
                ui.label("Live input preview");
                let preview =
                    self.compositor_input_source(node_id, 0)
                        .and_then(|(source_id, output)| {
                            self.probe_compositor_node(source_id, output, &mut BTreeSet::new())
                        });
                match preview {
                    Ok(CompositorProbeValue::Image(texture)) => {
                        let display = texture.to_texture_asset_clamped();
                        let image = ColorImage::from_rgba_unmultiplied(
                            [display.width as usize, display.height as usize],
                            &display.pixels,
                        );
                        if let Some(handle) = self.compositor_debug_textures.get_mut(&node_id) {
                            handle.set(image, TextureOptions::LINEAR);
                        } else {
                            let handle = ui.ctx().load_texture(
                                format!("compositor-debug-{node_id}"),
                                image,
                                TextureOptions::LINEAR,
                            );
                            self.compositor_debug_textures.insert(node_id, handle);
                        }
                        if let Some(handle) = self.compositor_debug_textures.get(&node_id) {
                            let available = ui.available_width().max(1.0);
                            let aspect = texture.width.max(1) as f32 / texture.height.max(1) as f32;
                            let size = Vec2::new(available, (available / aspect).min(120.0));
                            ui.add(egui::Image::new((handle.id(), size)));
                        }
                        ui.monospace(format!("{} × {}", texture.width, texture.height));
                        if texture.pixels.len() >= 4 {
                            ui.monospace(format!(
                                "RGBA {:.3}, {:.3}, {:.3}, {:.3}",
                                texture.pixels[0],
                                texture.pixels[1],
                                texture.pixels[2],
                                texture.pixels[3],
                            ));
                        }
                    }
                    Ok(CompositorProbeValue::Number(value)) => {
                        self.compositor_debug_textures.remove(&node_id);
                        ui.strong("Number");
                        ui.monospace(format!("{value:.6}"));
                    }
                    Ok(CompositorProbeValue::Triple(values)) => {
                        self.compositor_debug_textures.remove(&node_id);
                        ui.strong("XYZ triple");
                        ui.monospace(format!("X  {:.6}", values[0]));
                        ui.monospace(format!("Y  {:.6}", values[1]));
                        ui.monospace(format!("Z  {:.6}", values[2]));
                    }
                    Err(message) => {
                        self.compositor_debug_textures.remove(&node_id);
                        ui.small(message);
                    }
                }
            }
            17 => {
                let object_index = self
                    .compositor_nodes
                    .iter()
                    .find_map(|node| match node.settings {
                        NodeSettings::ObjectTransform { object_index } if node.id == node_id => {
                            Some(object_index)
                        }
                        _ => None,
                    })
                    .unwrap_or(0);
                if let Some(id) = self.object_node_id(object_index)
                    && let Ok(node) = self.scene.tree.node(id)
                {
                    let transform = node.local_transform();
                    let euler = transform.rotation.to_euler_xyz();
                    ui.small(format!(
                        "Position  {:.3}, {:.3}, {:.3}",
                        transform.translation.x, transform.translation.y, transform.translation.z
                    ));
                    ui.small(format!(
                        "Rotation  {:.1}°, {:.1}°, {:.1}°",
                        euler.x.to_degrees(),
                        euler.y.to_degrees(),
                        euler.z.to_degrees()
                    ));
                    ui.small(format!(
                        "Scale     {:.3}, {:.3}, {:.3}",
                        transform.scale.x, transform.scale.y, transform.scale.z
                    ));
                }
            }
            18 => {
                let object_index = self
                    .compositor_nodes
                    .iter()
                    .find_map(|node| match node.settings {
                        NodeSettings::ObjectMesh { object_index } if node.id == node_id => {
                            Some(object_index)
                        }
                        _ => None,
                    })
                    .unwrap_or(0);
                ui.label("Final object geometry");
                ui.small(
                    self.object_asset_path(object_index)
                        .unwrap_or("No mesh asset assigned"),
                );
            }
            19 => {
                let object_index = self
                    .compositor_nodes
                    .iter()
                    .find_map(|node| match node.settings {
                        NodeSettings::PaintedMask { object_index } if node.id == node_id => {
                            Some(object_index)
                        }
                        _ => None,
                    })
                    .unwrap_or(0);
                if let Some(object) = self.object_node_id(object_index)
                    && let Some(mask) = self.painted_masks.get(&object)
                {
                    ui.label("Scalar painted texture");
                    ui.monospace(format!("{} × {} · R8", mask.width, mask.height));
                } else {
                    ui.weak("No painted texture yet.");
                }
                if ui.button("Open Texture Painter").clicked()
                    && let Some(object) = self.object_node_id(object_index)
                {
                    self.scene.selected = Some(object);
                    self.ensure_painted_texture(object);
                    self.active_tool = Tool::TexturePaint;
                    self.workspace_tab = WorkspaceTab::Scene;
                    self.scene_revision = self.scene_revision.wrapping_add(1);
                }
            }
            20 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::MassDensity {
                    ref mut base_density,
                    ref mut scale,
                    ref mut minimum_mass,
                    ref mut normalize,
                    ref mut total_mass,
                    ..
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Surface density (kg/m²)");
                changed |= ui
                    .add(
                        egui::DragValue::new(base_density)
                            .speed(0.01)
                            .range(0.0..=10_000.0),
                    )
                    .changed();
                ui.label("Texture scale factor");
                changed |= ui
                    .add(
                        egui::DragValue::new(scale)
                            .speed(0.01)
                            .range(0.0..=10_000.0),
                    )
                    .changed();
                ui.label("Minimum particle mass (kg)");
                changed |= ui
                    .add(
                        egui::DragValue::new(minimum_mass)
                            .speed(0.000_001)
                            .range(0.000_001..=1.0),
                    )
                    .changed();
                changed |= ui.checkbox(normalize, "Normalize total mass").changed();
                if *normalize {
                    ui.label("Total mass (kg)");
                    changed |= ui
                        .add(
                            egui::DragValue::new(total_mass)
                                .speed(0.01)
                                .range(0.000_001..=10_000.0),
                        )
                        .changed();
                }
            }
            21 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::SpringMesh {
                    ref mut stiffness,
                    ref mut bend_stiffness,
                    ref mut damping,
                    ref mut iterations,
                    ..
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                changed |= ui
                    .add(egui::Slider::new(stiffness, 0.0..=1.0).text("Stretch stiffness"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(bend_stiffness, 0.0..=1.0).text("Bend stiffness"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(damping, 0.0..=0.95).text("Energy dissipation"))
                    .on_hover_text(
                        "Fraction of particle velocity removed each fixed tick. \
                         Stiffness stores and returns elastic energy; this control dissipates it.",
                    )
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(iterations, 1..=16).text("Iterations"))
                    .changed();
            }
            22 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::ForceField {
                    ref mut formulas,
                    ref mut scale,
                    ..
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Cartesian force formulas (N)");
                for (axis, formula) in ["Fx", "Fy", "Fz"].into_iter().zip(formulas.iter_mut()) {
                    ui.horizontal(|ui| {
                        ui.monospace(axis);
                        changed |= ui.text_edit_singleline(formula).changed();
                    });
                }
                changed |= ui
                    .add(egui::DragValue::new(scale).speed(0.01).prefix("Scale "))
                    .changed();
                let valid = formulas
                    .iter()
                    .all(|formula| evaluate_force_formula(formula, 0.3, 0.4, 0.5, 0.6).is_ok());
                if !valid {
                    ui.colored_label(Color32::from_rgb(235, 91, 91), "Invalid force formula");
                }
            }
            23 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::VelocityField {
                    ref mut formulas,
                    ref mut scale,
                    ref mut blend,
                    ref mut mode,
                    ..
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Cartesian velocity formulas (m/s)");
                for (axis, formula) in ["Vx", "Vy", "Vz"].into_iter().zip(formulas.iter_mut()) {
                    ui.horizontal(|ui| {
                        ui.monospace(axis);
                        changed |= ui.text_edit_singleline(formula).changed();
                    });
                }
                changed |= ui
                    .add(egui::DragValue::new(scale).speed(0.01).prefix("Scale "))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(blend, 0.0..=1.0).text("Blend"))
                    .changed();
                *mode = 1;
                if !formulas
                    .iter()
                    .all(|formula| evaluate_force_formula(formula, 0.3, 0.4, 0.5, 0.6).is_ok())
                {
                    ui.colored_label(Color32::from_rgb(235, 91, 91), "Invalid velocity formula");
                }
            }
            24 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::Simulator {
                    ref mut gravity,
                    ref mut time_scale,
                    ..
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                changed |= ui.checkbox(gravity, "World gravity").changed();
                changed |= ui
                    .add(
                        egui::DragValue::new(time_scale)
                            .range(0.0..=4.0)
                            .speed(0.01)
                            .prefix("Time scale "),
                    )
                    .changed();
            }
            25 | 26 => {
                let connected =
                    [0, 1, 2].map(|axis| self.compositor_input_source(node_id, axis).is_ok());
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let values = match &mut self.compositor_nodes[pos].settings {
                    NodeSettings::Position { values } => values,
                    NodeSettings::Rotation { degrees } => degrees,
                    _ => return,
                };
                ui.label(if kind == 25 {
                    "Position (world units)"
                } else {
                    "Euler rotation (degrees)"
                });
                for (axis, index) in ["X", "Y", "Z"].into_iter().zip(0..3) {
                    ui.horizontal(|ui| {
                        ui.monospace(axis);
                        let response = ui.add_enabled(
                            !connected[index],
                            egui::DragValue::new(&mut values[index]).speed(if kind == 25 {
                                0.01
                            } else {
                                0.25
                            }),
                        );
                        changed |= response.changed();
                    });
                }
            }
            27 => {
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::ObjectSimulator {
                    ref mut mass,
                    ref mut gravity,
                    ref mut linear_velocity,
                    ref mut angular_velocity,
                    ref mut linear_damping,
                    ref mut angular_damping,
                    ..
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                changed |= ui
                    .add(
                        egui::DragValue::new(mass)
                            .range(0.001..=1_000_000.0)
                            .prefix("Mass "),
                    )
                    .changed();
                changed |= ui.checkbox(gravity, "World gravity").changed();
                ui.label("Initial linear velocity (m/s)");
                for (axis, value) in ["X", "Y", "Z"].into_iter().zip(linear_velocity.iter_mut()) {
                    ui.horizontal(|ui| {
                        ui.monospace(axis);
                        changed |= ui.add(egui::DragValue::new(value).speed(0.05)).changed();
                    });
                }
                ui.label("Initial angular velocity (°/s)");
                for (axis, value) in ["X", "Y", "Z"].into_iter().zip(angular_velocity.iter_mut()) {
                    ui.horizontal(|ui| {
                        ui.monospace(axis);
                        changed |= ui.add(egui::DragValue::new(value).speed(0.25)).changed();
                    });
                }
                changed |= ui
                    .add(egui::Slider::new(linear_damping, 0.0..=1.0).text("Linear damping"))
                    .changed();
                changed |= ui
                    .add(egui::Slider::new(angular_damping, 0.0..=1.0).text("Angular damping"))
                    .changed();
            }
            28 => {
                let connected =
                    [0, 1, 2].map(|axis| self.compositor_input_source(node_id, axis).is_ok());
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::ForceOutput { ref mut force, .. } =
                    self.compositor_nodes[pos].settings
                else {
                    return;
                };
                ui.label("Force (N)");
                for (axis, index) in ["X", "Y", "Z"].into_iter().zip(0..3) {
                    ui.horizontal(|ui| {
                        ui.monospace(axis);
                        changed |= ui
                            .add_enabled(
                                !connected[index],
                                egui::DragValue::new(&mut force[index]).speed(0.1),
                            )
                            .changed();
                    });
                }
            }
            29 => {
                let connected =
                    [0, 1].map(|input| self.compositor_input_source(node_id, input).is_ok());
                let pos = self
                    .compositor_nodes
                    .iter()
                    .position(|node| node.id == node_id)
                    .unwrap();
                let NodeSettings::Engine {
                    ref mut throttle,
                    ref mut torque,
                    ref mut reverse,
                    ref mut auto_braking,
                    ref mut max_rpm,
                    ..
                } = self.compositor_nodes[pos].settings
                else {
                    return;
                };
                changed |= ui
                    .add_enabled(
                        !connected[0],
                        egui::Slider::new(throttle, -1.0..=1.0).text("Throttle"),
                    )
                    .changed();
                changed |= ui
                    .add_enabled(
                        !connected[1],
                        egui::DragValue::new(torque)
                            .range(0.0..=10_000_000.0)
                            .speed(1.0)
                            .suffix(" N·m"),
                    )
                    .changed();
                changed |= ui.checkbox(reverse, "Reverse direction").changed();
                changed |= ui.checkbox(auto_braking, "Auto-Braking").changed();
                changed |= ui
                    .add_enabled(
                        *auto_braking,
                        egui::DragValue::new(max_rpm)
                            .range(0.0..=1_000_000.0)
                            .speed(50.0)
                            .suffix(" RPM")
                            .prefix("Max RPM "),
                    )
                    .on_hover_text(
                        "Available when Auto-Braking is enabled. Throttle selects a target up to this RPM; the engine accelerates toward it and applies reverse torque above it.",
                    )
                    .changed();
                ui.small("Drives an Engine cylinder around its local Z axle.");
            }
            _ => {}
        }
        if open_browse {
            self.start_compositor_image_import(ui.ctx());
        }
        if changed {
            if let Some(node) = self
                .compositor_nodes
                .iter_mut()
                .find(|node| node.id == node_id)
                && let Some(index) = node.settings.object_index()
                && let Some(name) = self
                    .scene
                    .tree
                    .iter()
                    .nth(index)
                    .map(|(_, object)| object.name.clone())
            {
                node.settings_object_name = Some(name);
            }
            if kind == 27 {
                self.object_simulation_states.remove(&node_id);
            }
            if kind == 14 {
                self.invalidate_all_object_handles();
            } else {
                self.invalidate_compositor_from(node_id);
            }
        }
        self.project_dirty |= changed;
    }

    fn start_play(&mut self) {
        self.play_state = PlayState::Building;
        self.build_started = Some(Instant::now());
        self.logs.push(LogEntry {
            level: "BUILD",
            color: Color32::from_rgb(244, 190, 88),
            message: "Capturing scene snapshot and compiling zerofps-player…".into(),
        });
    }

    fn stop_play(&mut self) {
        self.play_state = PlayState::Editing;
        self.build_started = None;
        self.logs.push(LogEntry {
            level: "INFO",
            color: Color32::from_rgb(103, 191, 255),
            message: "Preview process stopped; editor state was preserved.".into(),
        });
    }

    fn poll_build(&mut self, ctx: &egui::Context) {
        if self.play_state == PlayState::Building {
            ctx.request_repaint_after(Duration::from_millis(50));
            if self
                .build_started
                .is_some_and(|started| started.elapsed() > Duration::from_millis(900))
            {
                self.play_state = PlayState::Running;
                self.logs.push(LogEntry {
                    level: "LIVE",
                    color: Color32::from_rgb(97, 219, 141),
                    message: "Game connected on telemetry://127.0.0.1 (prototype).".into(),
                });
            }
        }
    }

    fn shortcuts(&mut self, ctx: &egui::Context) {
        if ctx.wants_keyboard_input() {
            return;
        }
        if self.workspace_tab == WorkspaceTab::Scene {
            if ctx.input(|i| i.key_pressed(Key::Q)) {
                self.active_tool = Tool::Select;
            }
            if ctx.input(|i| i.key_pressed(Key::W)) {
                self.active_tool = Tool::Move;
            }
            if ctx.input(|i| i.key_pressed(Key::E)) {
                self.active_tool = Tool::Rotate;
            }
            if ctx.input(|i| i.key_pressed(Key::R)) {
                self.active_tool = Tool::Scale;
            }
            if ctx.input(|i| i.key_pressed(Key::Delete)) {
                let previous = self.scene.tree.clone();
                self.scene.remove_selected();
                self.record_undo(previous);
            }
        }
        if ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(
                egui::Modifiers::CTRL | egui::Modifiers::SHIFT,
                Key::S,
            ))
        }) {
            self.start_save_as(ctx);
        } else if ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::CTRL, Key::S))
        }) {
            self.save_or_save_as(ctx);
        }
        if ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::CTRL, Key::Z))
        }) {
            self.undo();
        }
        if ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::CTRL, Key::Y))
        }) {
            self.redo();
        }
        if ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::CTRL, Key::O))
        }) {
            self.start_load_project(ctx);
        }
    }

    fn top_bar(&mut self, ctx: &egui::Context) {
        let previous_workspace = self.workspace_tab;
        let previous_tool = self.active_tool;
        egui::TopBottomPanel::top("top_bar")
            .exact_height(70.0)
            .frame(panel_frame(Color32::from_rgb(25, 27, 34)))
            .show(ctx, |ui| {
                egui::menu::bar(ui, |ui| {
                    ui.label(
                        RichText::new("ZERO")
                            .strong()
                            .color(Color32::from_rgb(108, 190, 255)),
                    );
                    ui.label(RichText::new("FPS").strong());
                    ui.separator();
                    ui.menu_button("File", |ui| {
                        if ui.button("Save project  Ctrl+S").clicked() {
                            self.save_or_save_as(ui.ctx());
                            ui.close_menu();
                        }
                        if ui.button("Save As…  Ctrl+Shift+S").clicked() {
                            self.start_save_as(ui.ctx());
                            ui.close_menu();
                        }
                        if ui.button("Load Project…  Ctrl+O").clicked() {
                            self.start_load_project(ui.ctx());
                            ui.close_menu();
                        }
                        ui.separator();
                        ui.small(self.project_display_name())
                            .on_hover_text(self.project_path.display().to_string());
                    });
                    ui.menu_button("Edit", |ui| {
                        if ui
                            .add_enabled(
                                !self.undo_stack.is_empty(),
                                egui::Button::new("Undo  Ctrl+Z"),
                            )
                            .clicked()
                        {
                            self.undo();
                            ui.close_menu();
                        }
                        if ui
                            .add_enabled(
                                !self.redo_stack.is_empty(),
                                egui::Button::new("Redo  Ctrl+Y"),
                            )
                            .clicked()
                        {
                            self.redo();
                            ui.close_menu();
                        }
                    });
                    ui.menu_button("Props", |ui| {
                        ui.menu_button("Primitive", |ui| {
                            for primitive in [
                                BuiltinPrimitive::Cube,
                                BuiltinPrimitive::Sphere,
                                BuiltinPrimitive::Floor,
                            ] {
                                if ui.button(primitive.label()).clicked() {
                                    self.add_builtin_primitive(primitive);
                                    ui.close_menu();
                                }
                            }
                        });
                        ui.menu_button("Light", |ui| {
                            if ui.button("Light").clicked() {
                                self.add_omnidirectional_light();
                                ui.close_menu();
                            }
                        });
                        ui.menu_button("Camera", |ui| {
                            if ui.button("Camera").clicked() {
                                self.add_camera_object();
                                ui.close_menu();
                            }
                        });
                    });
                    for title in ["Scene", "Build", "Window"] {
                        ui.menu_button(title, |ui| {
                            ui.label(format!("{title} commands"));
                            ui.small("Prototype surface");
                        });
                    }
                    ui.menu_button("Settings", |ui| {
                        ui.strong("Device");
                        let previous_device = self.render_device;
                        egui::ComboBox::from_id_salt("settings_render_device")
                            .selected_text(self.render_device.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.render_device,
                                    RenderDevice::Vulkan,
                                    "Vulkan",
                                );
                                ui.selectable_value(
                                    &mut self.render_device,
                                    RenderDevice::Cpu,
                                    "CPU",
                                );
                            });
                        if self.render_device != previous_device {
                            self.compositor_eval_cache.clear();
                            self.compositor_gpu_cache.clear();
                            if self.render_device == RenderDevice::Cpu {
                                self.compositor_texture_overrides.retain(|(_, texture)| {
                                    matches!(texture, TextureOverride::Cpu(_))
                                });
                                self.compositor_apply_due = Some(Instant::now());
                            }
                            self.viewport_depth_key = None;
                            self.viewport_requested_key = None;
                            self.scene_revision = self.scene_revision.wrapping_add(1);
                            self.project_dirty = true;
                        }
                        ui.small(match self.render_device {
                            RenderDevice::Vulkan => {
                                "GPU viewport + compositing · automatic CPU fallback"
                            }
                            RenderDevice::Cpu => "Portable reference renderer + compositor",
                        });
                        if ui
                            .checkbox(&mut self.save_cache_in_file, "Save cache in file")
                            .on_hover_text(
                                "Store validated 1/2× and 1/4× texture mipmaps beside imported models as `<model>.zfp-cache`",
                            )
                            .changed()
                        {
                            self.project_dirty = true;
                        }
                        ui.separator();
                        ui.strong("Render");
                        ui.collapsing("Shadows", |ui| {
                        let previous_quality = self.shadow_quality;
                        egui::ComboBox::from_id_salt("settings_shadow_quality")
                            .selected_text(format!(
                                "Shadow quality: {}",
                                shadow_quality_label(self.shadow_quality)
                            ))
                            .show_ui(ui, |ui| {
                                for (quality, label) in [
                                    (0, "Potato"),
                                    (1, "Low"),
                                    (2, "Medium"),
                                    (3, "High"),
                                    (4, "Ultra"),
                                ] {
                                    ui.selectable_value(&mut self.shadow_quality, quality, label);
                                }
                            });
                        if self.shadow_quality != previous_quality {
                            self.viewport_requested_key = None;
                            self.project_dirty = true;
                        }
                        ui.small("Downsamples all shadow maps, then filters their reconstruction.");
                        let previous_blur = self.shadow_blur_radius;
                        egui::ComboBox::from_id_salt("settings_shadow_blur")
                            .selected_text(match self.shadow_blur_radius {
                                0 => "Fast blur: Off".into(),
                                radius => format!("Fast blur: {radius} px"),
                            })
                            .show_ui(ui, |ui| {
                                for (radius, label) in [
                                    (0, "Off"),
                                    (1, "1 px"),
                                    (2, "2 px"),
                                    (3, "3 px"),
                                    (4, "4 px"),
                                ] {
                                    ui.selectable_value(
                                        &mut self.shadow_blur_radius,
                                        radius,
                                        label,
                                    );
                                }
                            });
                        if self.shadow_blur_radius != previous_blur {
                            self.viewport_requested_key = None;
                            self.project_dirty = true;
                        }
                        ui.horizontal_wrapped(|ui| {
                            ui.small("Continuous shadows · reduce spatial quality before temporal quality.");
                            let previous_fps = self.target_fps;
                            egui::ComboBox::from_id_salt("settings_target_fps")
                                .selected_text(format!("Target: {} FPS", self.target_fps))
                                .show_ui(ui, |ui| {
                                    for fps in [30, 45, 60, 75, 90, 120, 144, 165, 240] {
                                        ui.selectable_value(
                                            &mut self.target_fps,
                                            fps,
                                            format!("{fps} FPS"),
                                        );
                                    }
                                });
                            if self.target_fps != previous_fps {
                                self.target_fps = sanitize_target_fps(self.target_fps);
                                self.next_viewport_frame = Instant::now();
                                self.project_dirty = true;
                            }
                        });
                        });
                        ui.separator();
                        ui.strong("Viewport");
                        ui.checkbox(&mut self.show_grid, "Show grid");
                        egui::ComboBox::from_id_salt("global_shadow_resolution")
                            .selected_text(if self.global_shadow_resolution == 0 {
                                "Global shadows: Off".into()
                            } else {
                                format!("Global shadows: {} px", self.global_shadow_resolution)
                            })
                            .show_ui(ui, |ui| {
                                for (resolution, label) in [
                                    (0, "Off"),
                                    (128, "128 px"),
                                    (256, "256 px"),
                                    (512, "512 px"),
                                    (1024, "1024 px"),
                                    (2048, "2048 px"),
                                ] {
                                    if ui
                                        .selectable_value(
                                            &mut self.global_shadow_resolution,
                                            resolution,
                                            label,
                                        )
                                        .changed()
                                    {
                                        self.viewport_requested_key = None;
                                        self.project_dirty = true;
                                    }
                                }
                            });
                        egui::ComboBox::from_id_salt("settings_grid_spacing")
                            .selected_text(format_grid_spacing(self.grid_spacing))
                            .show_ui(ui, |ui| {
                                for spacing in [
                                    0.001,
                                    0.01,
                                    0.1,
                                    1.0,
                                    10.0,
                                    100.0,
                                    1000.0,
                                    10_000.0,
                                    100_000.0,
                                    1_000_000.0,
                                ] {
                                    ui.selectable_value(
                                        &mut self.grid_spacing,
                                        spacing,
                                        format_grid_spacing(spacing),
                                    );
                                }
                            });
                        ui.small("Grid spacing in world units");
                    });
                    ui.menu_button("Help", |ui| {
                        ui.label("Help commands");
                        ui.small("Prototype surface");
                    });
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.toggle_value(&mut self.advanced, "Advanced");
                        ui.label(RichText::new(self.project_display_name()).weak().small())
                            .on_hover_text(if self.project_dirty {
                                "Unsaved changes"
                            } else {
                                "Saved"
                            });
                        let (device_label, badge_fill, badge_text) = match self.render_device {
                            RenderDevice::Vulkan => (
                                " GPU ",
                                Color32::from_rgb(42, 112, 70),
                                Color32::from_rgb(182, 255, 205),
                            ),
                            RenderDevice::Cpu => (
                                " CPU ",
                                Color32::from_rgb(126, 96, 28),
                                Color32::from_rgb(255, 225, 139),
                            ),
                        };
                        ui.label(
                            RichText::new(device_label)
                                .monospace()
                                .strong()
                                .small()
                                .color(badge_text)
                                .background_color(badge_fill),
                        )
                        .on_hover_text(match self.render_device {
                            RenderDevice::Vulkan => "Vulkan GPU rendering and compositing",
                            RenderDevice::Cpu => "CPU reference rendering and compositing",
                        });
                    });
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.workspace_tab, WorkspaceTab::Scene, "Scene");
                    ui.selectable_value(
                        &mut self.workspace_tab,
                        WorkspaceTab::Compositing,
                        "Nodes",
                    );
                    ui.separator();
                    if self.workspace_tab == WorkspaceTab::Scene {
                        tool_button(ui, &mut self.active_tool, Tool::Select, "Q", "Select");
                        tool_button(ui, &mut self.active_tool, Tool::Move, "W", "Move");
                        tool_button(ui, &mut self.active_tool, Tool::Rotate, "E", "Rotate");
                        tool_button(ui, &mut self.active_tool, Tool::Scale, "R", "Scale");
                        tool_button(
                            ui,
                            &mut self.active_tool,
                            Tool::FieldPaint,
                            "P",
                            "Field Paint",
                        );
                        ui.separator();
                        ui.toggle_value(&mut self.snap, "⌗ Snap");
                        ui.toggle_value(&mut self.show_grid, "Grid");
                        ui.separator();
                        if ui
                            .button(if self.dynamics_running {
                                "Ⅱ Simulation"
                            } else {
                                "▶ Simulation"
                            })
                            .on_hover_text("Run or pause the Scene dynamics graphs")
                            .clicked()
                        {
                            self.dynamics_running = !self.dynamics_running;
                            self.dynamics_single_step = false;
                            self.dynamics_accumulator = 0.0;
                            self.dynamics_last_tick = Instant::now();
                        }
                        if ui
                            .add_enabled(!self.dynamics_running, egui::Button::new("Step"))
                            .on_hover_text("Advance one fixed 1/60 s simulation step")
                            .clicked()
                        {
                            self.dynamics_single_step = true;
                        }
                        if ui.button("Reset Sim").clicked() {
                            self.dynamics_running = false;
                            self.dynamics_single_step = false;
                            self.reset_dynamics();
                        }
                        let graph_count = self
                            .scene
                            .tree
                            .iter()
                            .enumerate()
                            .filter(|(index, _)| self.spring_graph_nodes(*index).is_some())
                            .count();
                        ui.label(
                            RichText::new(format!("{:.2}s · {graph_count}", self.dynamics_time))
                                .weak()
                                .small(),
                        )
                        .on_hover_text("Simulation time · executable object graphs");
                    } else if self.workspace_tab == WorkspaceTab::Compositing {
                        ui.label(RichText::new("Object Graph").weak().small());
                        if ui.button("Frame All").clicked() {
                            self.compositor_pan = Vec2::ZERO;
                            self.compositor_zoom = 1.0;
                        }
                        ui.label(
                            RichText::new(format!("{:.0}%", self.compositor_zoom * 100.0))
                                .weak()
                                .small(),
                        );
                    }

                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        let stop = ui.add_enabled(
                            self.play_state != PlayState::Editing,
                            egui::Button::new("■ Stop"),
                        );
                        if stop.clicked() {
                            self.stop_play();
                        }
                        let pause_label = if self.play_state == PlayState::Paused {
                            "▶ Resume"
                        } else {
                            "Ⅱ Pause"
                        };
                        if ui
                            .add_enabled(
                                matches!(self.play_state, PlayState::Running | PlayState::Paused),
                                egui::Button::new(pause_label),
                            )
                            .clicked()
                        {
                            self.play_state = if self.play_state == PlayState::Paused {
                                PlayState::Running
                            } else {
                                PlayState::Paused
                            };
                        }
                        let label = if self.play_state == PlayState::Building {
                            "◌ Building…"
                        } else {
                            "▶ Play"
                        };
                        if ui
                            .add_enabled(
                                self.play_state == PlayState::Editing,
                                egui::Button::new(
                                    RichText::new(label).color(Color32::from_rgb(124, 220, 159)),
                                ),
                            )
                            .clicked()
                        {
                            self.start_play();
                        }
                    });
                });
            });
        if previous_workspace == WorkspaceTab::Compositing
            && self.workspace_tab == WorkspaceTab::Scene
        {
            self.apply_compositor();
        }
        if previous_tool != self.active_tool {
            self.scene_revision = self.scene_revision.wrapping_add(1);
            if previous_tool == Tool::TexturePaint {
                self.apply_compositor();
            }
        }
    }

    fn hierarchy(&mut self, ctx: &egui::Context) {
        egui::SidePanel::left("hierarchy")
            .resizable(true)
            .default_width(230.0)
            .width_range(180.0..=360.0)
            .frame(panel_frame(Color32::from_rgb(28, 30, 37)))
            .show(ctx, |ui| {
                section_title(ui, "SCENE HIERARCHY", "+");
                ui.add(
                    egui::TextEdit::singleline(&mut self.hierarchy_filter)
                        .hint_text("⌕  Filter objects")
                        .desired_width(f32::INFINITY),
                );
                ui.add_space(5.0);
                egui::ScrollArea::vertical()
                    .id_salt("scene_objects_scroll")
                    .auto_shrink([false, false])
                    .scroll_bar_visibility(
                        egui::scroll_area::ScrollBarVisibility::VisibleWhenNeeded,
                    )
                    .show(ui, |ui| {
                        let roots = self.scene.tree.roots().to_vec();
                        let mut reparent = None;
                        for root in roots {
                            if let Some(operation) = self.object_tree(ui, root, 0) {
                                reparent = Some(operation);
                            }
                        }
                        if egui::DragAndDrop::has_payload_of_type::<NodeId>(ui.ctx()) {
                            let (_, dropped) = ui.dnd_drop_zone::<NodeId, _>(
                                egui::Frame::new().inner_margin(egui::Margin::symmetric(6, 5)),
                                |ui| {
                                    ui.set_min_width(ui.available_width());
                                    ui.centered_and_justified(|ui| {
                                        ui.weak("Drop here to move to scene root");
                                    });
                                },
                            );
                            if let Some(child) = dropped {
                                reparent = Some((*child, None));
                            }
                        }
                        if let Some((child, parent)) = reparent {
                            self.reparent_hierarchy_object(child, parent);
                        }
                        if !ui.input(|input| input.pointer.primary_down()) {
                            self.hierarchy_drag_candidate = None;
                        }
                        ui.add_space(10.0);
                        if ui.button("+ Add object").clicked() {
                            let previous = self.scene.tree.clone();
                            let id = self.scene.add("New Object", ObjectKind::Empty, None);
                            self.scene.selected = Some(id);
                            self.record_undo(previous);
                        }
                    });
            });
    }

    fn object_tree(
        &mut self,
        ui: &mut egui::Ui,
        id: NodeId,
        depth: usize,
    ) -> Option<(NodeId, Option<NodeId>)> {
        let Ok(object) = self.scene.tree.node(id) else {
            return None;
        };
        let name = object.name.clone();
        let children = object.children().to_vec();
        let kind = self.scene.kind(id);
        let (_, dropped) = ui.dnd_drop_zone::<NodeId, _>(egui::Frame::new(), |ui| {
            let response = ui
                .horizontal(|ui| {
                    ui.add_space(depth as f32 * 13.0);
                    if !children.is_empty() {
                        ui.small("⌄");
                    } else {
                        ui.add_space(11.0);
                    }
                    let selected = self.scene.selected == Some(id);
                    ui.selectable_label(selected, format!("{}  {}", kind.icon(), name))
                })
                .inner;
            let drag_response = ui.interact(
                response.rect,
                Id::new(("hierarchy_object_drag", id)),
                Sense::click_and_drag(),
            );
            let row_response = drag_response | response;
            if row_response.clicked() {
                self.scene.selected = Some(id);
            }
            if row_response.hovered() && ui.input(|input| input.pointer.primary_pressed()) {
                self.hierarchy_drag_candidate = Some((id, Instant::now()));
            }
            if let Some((candidate, started)) = self.hierarchy_drag_candidate
                && candidate == id
                && ui.input(|input| input.pointer.primary_down())
            {
                const DRAG_HOLD: Duration = Duration::from_millis(220);
                if started.elapsed() >= DRAG_HOLD && row_response.dragged() {
                    egui::DragAndDrop::set_payload(ui.ctx(), id);
                    ui.ctx().set_cursor_icon(egui::CursorIcon::Grabbing);
                } else {
                    ui.ctx()
                        .request_repaint_after(DRAG_HOLD.saturating_sub(started.elapsed()));
                }
            }
        });
        let mut reparent = dropped.map(|child| (*child, Some(id)));
        for child in children {
            if let Some(operation) = self.object_tree(ui, child, depth + 1) {
                reparent = Some(operation);
            }
        }
        reparent
    }

    fn reparent_hierarchy_object(&mut self, child: NodeId, parent: Option<NodeId>) {
        if self
            .scene
            .tree
            .node(child)
            .ok()
            .and_then(|node| node.parent())
            == parent
        {
            return;
        }
        let previous = self.scene.tree.clone();
        match self
            .scene
            .tree
            .reparent(child, parent, ReparentMode::KeepGlobal)
        {
            Ok(()) => {
                self.scene.selected = Some(child);
                self.record_undo(previous);
                self.scene_revision = self.scene_revision.wrapping_add(1);
                self.project_dirty = true;
                self.logs.push(LogEntry {
                    level: "SCENE",
                    color: Color32::from_rgb(103, 191, 255),
                    message: match parent {
                        Some(parent) => format!(
                            "Moved object under {}",
                            self.scene
                                .tree
                                .node(parent)
                                .map(|node| node.name.as_str())
                                .unwrap_or("parent")
                        ),
                        None => "Moved object to scene root".into(),
                    },
                });
            }
            Err(error) => self.logs.push(LogEntry {
                level: "ERROR",
                color: Color32::from_rgb(235, 91, 91),
                message: format!("Could not reparent object: {error}"),
            }),
        }
    }

    fn inspector(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("inspector")
            .resizable(true)
            .default_width(300.0)
            .width_range(240.0..=430.0)
            .frame(panel_frame(Color32::from_rgb(28, 30, 37)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.selectable_value(
                        &mut self.inspector_tab,
                        InspectorTab::Inspector,
                        "Inspector",
                    );
                    ui.selectable_value(
                        &mut self.inspector_tab,
                        InspectorTab::Inputs,
                        "Inputs",
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.weak("⋮");
                    });
                });
                ui.separator();
                if self.inspector_tab == InspectorTab::Inputs {
                    self.object_handles_panel(ui);
                    return;
                }
                let inspector_height = ui.available_height();
                let inspector_size = Vec2::new(ui.available_width(), inspector_height);
                ui.allocate_ui(inspector_size, |ui| {
                    ui.set_min_size(inspector_size);
                    egui::ScrollArea::vertical()
                        .id_salt("inspector_scroll")
                        .max_height(inspector_height)
                        .auto_shrink([false, false])
                        .scroll_bar_visibility(
                            egui::scroll_area::ScrollBarVisibility::AlwaysVisible,
                        )
                        .show(ui, |ui| {
                            ui.set_width((inspector_size.x - 14.0).max(1.0));
                            let Some(id) = self.scene.selected else {
                                ui.centered_and_justified(|ui| ui.weak("Select an object"));
                                return;
                            };
                            let Ok(node) = self.scene.tree.node(id) else {
                                return;
                            };
                            let mut name = node.name.clone();
                            let previous_name = name.clone();
                            let mut transform = node.local_transform();
                            let previous = self.scene.tree.clone();
                            let mut changed = false;
                            let mut lighting_only_changed = false;
                            let kind = self.scene.kind(id);
                            let model_asset =
                                node.components
                                    .iter()
                                    .find_map(|component| match component {
                                        Component::Model { asset, .. } => Some(asset.clone()),
                                        _ => None,
                                    });
                            let mut collider_component = node
                                .components
                                .iter()
                                .find(|component| matches!(component, Component::Collider { .. }))
                                .cloned();
                            let mut light_component =
                                node.components.iter().find_map(|component| match component {
                                    Component::Light {
                                        intensity,
                                        color,
                                        radius,
                                        shadow_resolution,
                                    } => {
                                        Some((*intensity, *color, *radius, *shadow_resolution))
                                    }
                                    _ => None,
                                });
                            let mut camera_component =
                                node.components.iter().find_map(|component| match component {
                                    Component::Camera {
                                        field_of_view_degrees,
                                        projection,
                                        aspect_ratio,
                                        near_clip,
                                        far_clip,
                                        orthographic_size,
                                    } => Some((
                                        *field_of_view_degrees,
                                        projection.clone(),
                                        *aspect_ratio,
                                        *near_clip,
                                        *far_clip,
                                        *orthographic_size,
                                    )),
                                    _ => None,
                                });
                            let mut visible = self.scene.visible(id);
                            ui.horizontal(|ui| {
                                if ui.checkbox(&mut visible, "").changed() {
                                    let _ = self.scene.tree.set_attribute(
                                        id,
                                        AttributeKey::Visible,
                                        AttributeDeclaration::Value(Attribute::Bool(visible)),
                                    );
                                    changed = true;
                                }
                                ui.add(
                                    egui::TextEdit::singleline(&mut name)
                                        .desired_width(f32::INFINITY),
                                );
                            });
                            if self.scene.tree.node(id).is_ok_and(|node| node.name != name) {
                                self.rename_compositor_object_reference(&previous_name, &name);
                                self.scene.tree.node_mut(id).expect("selected node").name = name;
                                changed = true;
                            }
                            ui.small(format!(
                                "{}  Node {}:{}",
                                kind.icon(),
                                id.slot,
                                id.generation
                            ));
                            ui.add_space(8.0);
                            egui::CollapsingHeader::new(RichText::new("Transform").strong())
                                .default_open(true)
                                .show(ui, |ui| {
                                    let mut position = [
                                        transform.translation.x,
                                        transform.translation.y,
                                        transform.translation.z,
                                    ];
                                    let euler = transform.rotation.to_euler_xyz();
                                    let mut rotation_degrees = [
                                        euler.x.to_degrees(),
                                        euler.y.to_degrees(),
                                        euler.z.to_degrees(),
                                    ];
                                    let mut scale =
                                        [transform.scale.x, transform.scale.y, transform.scale.z];
                                    vector_editor(
                                        ui,
                                        "Position",
                                        &mut position,
                                        f64::from(self.grid_spacing),
                                    );
                                    let rotation_changed = vector_editor(
                                        ui,
                                        "Rotation (degrees · XYZ)",
                                        &mut rotation_degrees,
                                        0.5,
                                    );
                                    vector_editor(ui, "Scale", &mut scale, 0.05);
                                    transform.translation =
                                        CoreVec3::new(position[0], position[1], position[2]);
                                    if rotation_changed {
                                        transform.rotation = Quat::from_euler_xyz(CoreVec3::new(
                                            rotation_degrees[0].to_radians(),
                                            rotation_degrees[1].to_radians(),
                                            rotation_degrees[2].to_radians(),
                                        ));
                                    }
                                    transform.scale = CoreVec3::new(scale[0], scale[1], scale[2]);
                                    ui.small(
                                        RichText::new("Inherited through GeometryTree").weak(),
                                    );
                                });
                            if self
                                .scene
                                .tree
                                .node(id)
                                .is_ok_and(|node| node.local_transform() != transform)
                            {
                                let _ = self.scene.tree.set_local_transform(id, transform);
                                changed = true;
                            }
                            if let Some((
                                ref mut intensity,
                                ref mut color,
                                ref mut radius,
                                ref mut shadow_resolution,
                            )) = light_component
                            {
                                let mut light_changed = false;
                                let mut intensity_exponent = intensity_to_exponent(*intensity);
                                egui::CollapsingHeader::new(RichText::new("Light").strong())
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        ui.label("Omnidirectional point light");
                                        if ui
                                            .add(
                                                egui::DragValue::new(&mut intensity_exponent)
                                                    .speed(0.1)
                                                    .prefix("Intensity I "),
                                            )
                                            .on_hover_text(
                                                "Physical light power is 10^I. This logarithmic control has no artificial maximum.",
                                            )
                                            .changed()
                                        {
                                            *intensity =
                                                light_intensity_from_exponent(intensity_exponent);
                                            light_changed = true;
                                        }
                                        ui.small(format!("Power: {:.4e} = 10^I", *intensity));
                                        if ui
                                            .add(
                                                egui::DragValue::new(radius)
                                                    .speed(
                                                        (self.grid_spacing * 0.01)
                                                            .max(f32::EPSILON),
                                                    )
                                                    .range(0.0..=f32::MAX)
                                                    .prefix("Radius "),
                                            )
                                            .on_hover_text(
                                                "Radius of the spherical emitter in world units. \
                                                 Larger values soften the diffuse light terminator.",
                                            )
                                            .changed()
                                        {
                                            light_changed = true;
                                        }
                                        ui.horizontal(|ui| {
                                            ui.label("Color");
                                            light_changed |=
                                                ui.color_edit_button_rgb(color).changed();
                                        });
                                        egui::ComboBox::from_id_salt((
                                            "light_shadow_resolution",
                                            id.slot,
                                            id.generation,
                                        ))
                                        .selected_text(if *shadow_resolution == 0 {
                                            "Shadows: Off".into()
                                        } else {
                                            format!("Shadows: {} px", *shadow_resolution)
                                        })
                                        .show_ui(ui, |ui| {
                                            for (resolution, label) in [
                                                (0, "Off"),
                                                (128, "128 px"),
                                                (256, "256 px"),
                                                (512, "512 px"),
                                                (1024, "1024 px"),
                                                (2048, "2048 px"),
                                            ] {
                                                light_changed |= ui
                                                    .selectable_value(
                                                        shadow_resolution,
                                                        resolution,
                                                        label,
                                                    )
                                                    .changed();
                                            }
                                        });
                                        ui.small(
                                            "Shadow resolution is per cube face for point lights.",
                                        );
                                        ui.small("Position is controlled by the object transform.");
                                    });
                                if light_changed {
                                    if let Ok(node) = self.scene.tree.node_mut(id)
                                        && let Some(Component::Light {
                                            intensity: stored_intensity,
                                            color: stored_color,
                                            radius: stored_radius,
                                            shadow_resolution: stored_shadow_resolution,
                                        }) = node
                                            .components
                                            .iter_mut()
                                            .find(|component| {
                                                matches!(component, Component::Light { .. })
                                            })
                                    {
                                        *stored_intensity = intensity.max(0.0);
                                        *stored_color = *color;
                                        *stored_radius = radius.max(0.0);
                                        *stored_shadow_resolution = *shadow_resolution;
                                    }
                                    lighting_only_changed = true;
                                }
                            }
                            if let Some((
                                ref mut field_of_view,
                                ref mut projection,
                                ref mut aspect_ratio,
                                ref mut near_clip,
                                ref mut far_clip,
                                ref mut orthographic_size,
                            )) = camera_component
                            {
                                let mut camera_changed = false;
                                egui::CollapsingHeader::new(RichText::new("Camera").strong())
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        ui.label("Scene camera");
                                        egui::ComboBox::from_id_salt(("camera_projection", id))
                                            .selected_text(if projection == "orthographic" {
                                                "Orthographic"
                                            } else {
                                                "Perspective"
                                            })
                                            .show_ui(ui, |ui| {
                                                camera_changed |= ui
                                                    .selectable_value(
                                                        projection,
                                                        "perspective".into(),
                                                        "Perspective",
                                                    )
                                                    .changed();
                                                camera_changed |= ui
                                                    .selectable_value(
                                                        projection,
                                                        "orthographic".into(),
                                                        "Orthographic",
                                                    )
                                                    .changed();
                                            });
                                        if projection == "perspective" {
                                            camera_changed |= ui
                                                .add(
                                                    egui::Slider::new(field_of_view, 1.0..=179.0)
                                                        .text("Field of view")
                                                        .suffix("°"),
                                                )
                                                .changed();
                                        } else {
                                            camera_changed |= ui
                                                .add(
                                                    egui::DragValue::new(orthographic_size)
                                                        .speed(0.1)
                                                        .range(0.001..=f32::MAX)
                                                        .prefix("Size "),
                                                )
                                                .changed();
                                        }
                                        camera_changed |= ui
                                            .add(
                                                egui::DragValue::new(aspect_ratio)
                                                    .speed(0.01)
                                                    .range(0.1..=10.0)
                                                    .prefix("Aspect "),
                                            )
                                            .on_hover_text("Output width divided by output height")
                                            .changed();
                                        camera_changed |= ui
                                            .add(
                                                egui::DragValue::new(near_clip)
                                                    .speed(0.01)
                                                    .range(0.0001..=f32::MAX)
                                                    .prefix("Near clip "),
                                            )
                                            .changed();
                                        camera_changed |= ui
                                            .add(
                                                egui::DragValue::new(far_clip)
                                                    .speed(1.0)
                                                    .range(0.001..=f32::MAX)
                                                    .prefix("Far clip "),
                                            )
                                            .changed();
                                        ui.small("Local +Y is forward · local +Z is up");
                                    });
                                if camera_changed {
                                    *field_of_view = field_of_view.clamp(1.0, 179.0);
                                    *aspect_ratio = aspect_ratio.clamp(0.1, 10.0);
                                    *near_clip = near_clip.max(0.0001);
                                    *far_clip = far_clip.max(*near_clip + 0.0001);
                                    *orthographic_size = orthographic_size.max(0.001);
                                    if let Ok(node) = self.scene.tree.node_mut(id)
                                        && let Some(Component::Camera {
                                            field_of_view_degrees: stored_fov,
                                            projection: stored_projection,
                                            aspect_ratio: stored_aspect,
                                            near_clip: stored_near,
                                            far_clip: stored_far,
                                            orthographic_size: stored_size,
                                        }) = node
                                            .components
                                            .iter_mut()
                                            .find(|component| {
                                                matches!(component, Component::Camera { .. })
                                            })
                                    {
                                        *stored_fov = *field_of_view;
                                        *stored_projection = projection.clone();
                                        *stored_aspect = *aspect_ratio;
                                        *stored_near = *near_clip;
                                        *stored_far = *far_clip;
                                        *stored_size = *orthographic_size;
                                    }
                                    changed = true;
                                }
                            }
                            egui::CollapsingHeader::new(RichText::new("Mesh Renderer").strong())
                                .default_open(true)
                                .show(ui, |ui| {
                                    property(
                                        ui,
                                        "Mesh",
                                        model_asset.as_deref().unwrap_or("No mesh assigned"),
                                    );
                                    if model_asset.is_some() {
                                        let mut casts_shadows = self
                                            .scene
                                            .tree
                                            .resolve_attribute(id, &AttributeKey::CastShadows)
                                            .ok()
                                            .and_then(|attribute| match attribute.value {
                                                Attribute::Bool(value) => Some(value),
                                                _ => None,
                                            })
                                            .unwrap_or(true);
                                        if ui
                                            .checkbox(&mut casts_shadows, "Cast shadows")
                                            .on_hover_text(
                                                "Include this visible mesh in direct-light \
                                                 occlusion. This inherited attribute is \
                                                 script-writable.",
                                            )
                                            .changed()
                                        {
                                            let _ = self.scene.tree.set_attribute(
                                                id,
                                                AttributeKey::CastShadows,
                                                AttributeDeclaration::Value(Attribute::Bool(
                                                    casts_shadows,
                                                )),
                                            );
                                            changed = true;
                                        }
                                        let mut mesh_autofix = self
                                            .scene
                                            .tree
                                            .resolve_attribute(id, &mesh_autofix_key())
                                            .ok()
                                            .and_then(|attribute| match attribute.value {
                                                Attribute::Bool(value) => Some(value),
                                                _ => None,
                                            })
                                            .unwrap_or(true);
                                        if ui
                                            .checkbox(&mut mesh_autofix, "Mesh Autofix")
                                            .on_hover_text(
                                                "Fill only small, simple boundary loops and \
                                                 regenerate vertex normals. The imported source \
                                                 mesh remains unchanged.",
                                            )
                                            .changed()
                                        {
                                            let _ = self.scene.tree.set_attribute(
                                                id,
                                                mesh_autofix_key(),
                                                AttributeDeclaration::Value(Attribute::Bool(
                                                    mesh_autofix,
                                                )),
                                            );
                                            changed = true;
                                        }
                                        if let Some(report) =
                                            model_asset.as_deref().and_then(|path| {
                                                self.imported_assets
                                                    .iter()
                                                    .find(|asset| asset.path == path)
                                                    .map(|asset| asset.autofix_report)
                                            })
                                        {
                                            ui.small(format!(
                                                "{} of {} small holes filled · {} triangles added",
                                                report.filled_loops,
                                                report.boundary_loops,
                                                report.added_triangles
                                            ));
                                        }
                                    }
                                    inherited_property(ui, "Visibility", "Visible", "Environment");
                                    inherited_property(ui, "Layer", "Default", "Project");
                                });
                            let mut collider_changed = false;
                            let mut remove_collider = false;
                            egui::CollapsingHeader::new(RichText::new("Collider").strong())
                                .default_open(true)
                                .show(ui, |ui| {
                                    if kind == ObjectKind::Camera {
                                        ui.weak(
                                            "Cameras are weightless editor/game viewpoints and do not participate in collisions.",
                                        );
                                        if collider_component.is_some() {
                                            collider_component = None;
                                            remove_collider = true;
                                            collider_changed = true;
                                        }
                                        return;
                                    }
                                    if collider_component.is_none() {
                                        ui.small(
                                            "Colliders are independent from visible mesh geometry.",
                                        );
                                        if ui.button("+ Add Collider").clicked() {
                                            collider_component = Some(
                                                self.best_fit_collider(id, ColliderShape::Box),
                                            );
                                            collider_changed = true;
                                        }
                                        return;
                                    }
                                    let current_shape = match collider_component.as_ref() {
                                        Some(Component::Collider { shape, .. }) => {
                                            ColliderShape::from_storage(shape)
                                        }
                                        _ => ColliderShape::Box,
                                    };
                                    let mut selected_shape = current_shape;
                                    ui.label("Shape");
                                    egui::ComboBox::from_id_salt(("collider_shape", id))
                                        .selected_text(selected_shape.label())
                                        .show_ui(ui, |ui| {
                                            for shape in ColliderShape::ALL {
                                                ui.selectable_value(
                                                    &mut selected_shape,
                                                    shape,
                                                    shape.label(),
                                                );
                                            }
                                        });
                                    if selected_shape != current_shape {
                                        collider_component =
                                            Some(self.best_fit_collider(id, selected_shape));
                                        collider_changed = true;
                                    }
                                    let Some(Component::Collider {
                                        center,
                                        half_extents,
                                        radius,
                                        height,
                                        coupling_stiffness,
                                        coupling_damping,
                                        force_cutoff,
                                        restitution,
                                        density,
                                        mass,
                                        automatic_mass,
                                        friction,
                                        friction_margin_percent,
                                        joint,
                                        ..
                                    }) = collider_component.as_mut()
                                    else {
                                        return;
                                    };
                                    let mut center_values = [center.x, center.y, center.z];
                                    if vector_editor(ui, "Center", &mut center_values, 0.01) {
                                        *center = CoreVec3::new(
                                            center_values[0],
                                            center_values[1],
                                            center_values[2],
                                        );
                                        collider_changed = true;
                                    }
                                    match selected_shape {
                                        ColliderShape::Sphere => {
                                            if ui
                                                .add(
                                                    egui::DragValue::new(radius)
                                                        .range(0.001..=1_000_000.0)
                                                        .speed(0.01)
                                                        .prefix("Radius "),
                                                )
                                                .changed()
                                            {
                                                *half_extents =
                                                    CoreVec3::new(*radius, *radius, *radius);
                                                *height = *radius * 2.0;
                                                collider_changed = true;
                                            }
                                        }
                                        ColliderShape::Cylinder => {
                                            let radius_changed = ui
                                                .add(
                                                    egui::DragValue::new(radius)
                                                        .range(0.001..=1_000_000.0)
                                                        .speed(0.01)
                                                        .prefix("Radius "),
                                                )
                                                .changed();
                                            let height_changed = ui
                                                .add(
                                                    egui::DragValue::new(height)
                                                        .range(0.001..=1_000_000.0)
                                                        .speed(0.01)
                                                        .prefix("Height "),
                                                )
                                                .changed();
                                            if radius_changed || height_changed {
                                                *half_extents =
                                                    CoreVec3::new(*radius, *radius, *height * 0.5);
                                                collider_changed = true;
                                            }
                                            ui.small("Cylinder axis: local Z");
                                            ui.separator();
                                            ui.label("Rotational joint");
                                            let current_joint =
                                                CylinderJoint::from_storage(joint);
                                            let mut selected_joint = current_joint;
                                            egui::ComboBox::from_id_salt((
                                                "cylinder_joint",
                                                id,
                                            ))
                                            .selected_text(selected_joint.label())
                                            .show_ui(
                                                ui,
                                                |ui| {
                                                    for candidate in CylinderJoint::ALL {
                                                        ui.selectable_value(
                                                            &mut selected_joint,
                                                            candidate,
                                                            candidate.label(),
                                                        );
                                                    }
                                                },
                                            );
                                            if selected_joint != current_joint {
                                                *joint =
                                                    selected_joint.storage_name().into();
                                                collider_changed = true;
                                            }
                                            ui.small(match selected_joint {
                                                CylinderJoint::None => {
                                                    "Coupling constrains every rotation axis."
                                                }
                                                CylinderJoint::Wheel => {
                                                    "Local Z axle rotates freely."
                                                }
                                                CylinderJoint::Engine => {
                                                    "Local Z axle is driven by an Engine node."
                                                }
                                            });
                                        }
                                        ColliderShape::Box => {
                                            let mut size = [
                                                half_extents.x * 2.0,
                                                half_extents.y * 2.0,
                                                half_extents.z * 2.0,
                                            ];
                                            if vector_editor(ui, "Size", &mut size, 0.01) {
                                                *half_extents = CoreVec3::new(
                                                    size[0].abs().max(0.001) * 0.5,
                                                    size[1].abs().max(0.001) * 0.5,
                                                    size[2].abs().max(0.001) * 0.5,
                                                );
                                                collider_changed = true;
                                            }
                                        }
                                        ColliderShape::Flat => {
                                            let mut size = [
                                                half_extents.x * 2.0,
                                                half_extents.y * 2.0,
                                                half_extents.z * 2.0,
                                            ];
                                            if vector_editor(
                                                ui,
                                                "Size X/Y/Thickness",
                                                &mut size,
                                                0.01,
                                            ) {
                                                *half_extents = CoreVec3::new(
                                                    size[0].abs().max(0.001) * 0.5,
                                                    size[1].abs().max(0.001) * 0.5,
                                                    size[2].abs().max(0.001) * 0.5,
                                                );
                                                *height = size[2].abs().max(0.001);
                                                collider_changed = true;
                                            }
                                            ui.small("Flat surface lies in local XY");
                                        }
                                    }
                                    ui.separator();
                                    ui.label("Component mass");
                                    if ui.checkbox(automatic_mass, "Automatic from volume").changed() {
                                        collider_changed = true;
                                    }
                                    if *automatic_mass {
                                        if ui
                                            .add(
                                                egui::DragValue::new(density)
                                                    .range(0.001..=100_000.0)
                                                    .speed(1.0)
                                                    .suffix(" kg/m³"),
                                            )
                                            .on_hover_text(
                                                "Mass density multiplied by estimated collider volume.",
                                            )
                                            .changed()
                                        {
                                            collider_changed = true;
                                        }
                                        let estimated_mass = collider_volume(
                                            selected_shape,
                                            *half_extents,
                                            *radius,
                                            *height,
                                        ) * density.max(0.001);
                                        if (*mass - estimated_mass).abs()
                                            > estimated_mass.abs().max(1.0) * f32::EPSILON
                                        {
                                            *mass = estimated_mass;
                                            collider_changed = true;
                                        }
                                        ui.label(format!("Estimated mass: {:.3} kg", *mass));
                                    } else if ui
                                        .add(
                                            egui::DragValue::new(mass)
                                                .range(0.001..=100_000_000.0)
                                                .speed(0.1)
                                                .suffix(" kg"),
                                        )
                                        .changed()
                                    {
                                        collider_changed = true;
                                    }
                                    ui.separator();
                                    ui.label("Coupling model");
                                    ui.label("Spring–damper");
                                    if ui
                                        .add(
                                            egui::DragValue::new(coupling_stiffness)
                                                .range(0.0..=100_000_000.0)
                                                .speed(100.0)
                                                .prefix("Stiffness "),
                                        )
                                        .on_hover_text(
                                            "Restoring force for displacement from the parent-relative rest transform.",
                                        )
                                        .changed()
                                    {
                                        collider_changed = true;
                                    }
                                    if ui
                                        .add(
                                            egui::DragValue::new(force_cutoff)
                                                .range(0.0..=1_000_000.0)
                                                .speed(0.01)
                                                .prefix("Force cut "),
                                        )
                                        .on_hover_text(
                                            "Forces and torques below this magnitude are set exactly to zero.",
                                        )
                                        .changed()
                                    {
                                        collider_changed = true;
                                    }
                                    if ui
                                        .add(
                                            egui::DragValue::new(coupling_damping)
                                                .range(0.0..=10_000_000.0)
                                                .speed(10.0)
                                                .prefix("Damping "),
                                        )
                                        .on_hover_text(
                                            "Damping applied to relative child/parent motion.",
                                        )
                                        .changed()
                                    {
                                        collider_changed = true;
                                    }
                                    ui.separator();
                                    ui.label("Collision response");
                                    ui.label("Non-penetrating impulse");
                                    if ui
                                        .add(
                                            egui::Slider::new(restitution, 0.0..=1.0)
                                                .text("Restitution"),
                                        )
                                        .on_hover_text(
                                            "0 stops inward motion without bouncing; 1 preserves it as a full bounce.",
                                        )
                                        .changed()
                                    {
                                        collider_changed = true;
                                    }
                                    egui::CollapsingHeader::new("Friction")
                                        .default_open(true)
                                        .show(ui, |ui| {
                                            if ui
                                                .add(
                                                    egui::DragValue::new(friction_margin_percent)
                                                        .range(0.0..=100.0)
                                                        .speed(0.25)
                                                        .suffix(" %")
                                                        .prefix("Contact tolerance "),
                                                )
                                                .on_hover_text(
                                                    "Inflates this collider only while detecting friction contact; collision boundaries remain unchanged.",
                                                )
                                                .changed()
                                            {
                                                collider_changed = true;
                                            }
                                            match selected_shape {
                                            ColliderShape::Sphere => {
                                                let mut isotropic =
                                                    (friction.x + friction.y + friction.z) / 3.0;
                                                if ui
                                                    .add(
                                                        egui::DragValue::new(&mut isotropic)
                                                            .range(0.0..=f32::MAX)
                                                            .speed(1.0)
                                                            .prefix("All directions "),
                                                    )
                                                    .changed()
                                                {
                                                    *friction = CoreVec3::new(
                                                        isotropic, isotropic, isotropic,
                                                    );
                                                    collider_changed = true;
                                                }
                                            }
                                            ColliderShape::Cylinder => {
                                                for (label, value) in [
                                                    ("Rolling", &mut friction.x),
                                                    ("Lateral", &mut friction.y),
                                                    ("Axial", &mut friction.z),
                                                ] {
                                                    if ui
                                                        .add(
                                                            egui::DragValue::new(value)
                                                                .range(0.0..=f32::MAX)
                                                                .speed(1.0)
                                                                .prefix(format!("{label} ")),
                                                        )
                                                        .changed()
                                                    {
                                                        collider_changed = true;
                                                    }
                                                }
                                            }
                                            ColliderShape::Box | ColliderShape::Flat => {
                                                for (label, value) in [
                                                    ("Local X", &mut friction.x),
                                                    ("Local Y", &mut friction.y),
                                                    ("Local Z", &mut friction.z),
                                                ] {
                                                    if ui
                                                        .add(
                                                            egui::DragValue::new(value)
                                                                .range(0.0..=f32::MAX)
                                                                .speed(1.0)
                                                                .prefix(format!("{label} ")),
                                                        )
                                                        .changed()
                                                    {
                                                        collider_changed = true;
                                                    }
                                                }
                                            }
                                            }
                                        });
                                    ui.horizontal(|ui| {
                                        if ui.button("Best Fit").clicked() {
                                            collider_component =
                                                Some(self.best_fit_collider(id, selected_shape));
                                            collider_changed = true;
                                        }
                                        if ui.button("Remove").clicked() {
                                            remove_collider = true;
                                            collider_changed = true;
                                        }
                                    });
                                    ui.small(
                                        "Changing shape runs Best Fit once; manual values remain \
                                         untouched afterward.",
                                    );
                                });
                            if collider_changed {
                                let node = self.scene.tree.node_mut(id).expect("selected node");
                                node.components.retain(|component| {
                                    !matches!(component, Component::Collider { .. })
                                });
                                if !remove_collider && let Some(collider) = collider_component {
                                    node.components.push(collider);
                                }
                                changed = true;
                            }
                            egui::CollapsingHeader::new(RichText::new("Texture Painting").strong())
                                .default_open(true)
                                .show(ui, |ui| {
                                    if model_asset.is_none() {
                                        ui.weak("Assign a UV-mapped mesh before texture painting.");
                                        return;
                                    }
                                    ui.small("Open painting from a Painted Texture node in Nodes.");
                                    if let Some(mask) = self.painted_masks.get(&id) {
                                        ui.small(format!(
                                            "{} × {} R8 mask · revision {}",
                                            mask.width, mask.height, mask.revision
                                        ));
                                        ui.small("0 = black · 1 = white");
                                    }
                                    if self.active_tool == Tool::TexturePaint {
                                        ui.checkbox(
                                            &mut self.texture_paint_heatmap,
                                            "Heatmap preview",
                                        );
                                        egui::ComboBox::from_id_salt(("texture_paint_mode", id))
                                            .selected_text(self.paint_mode.label())
                                            .show_ui(ui, |ui| {
                                                for mode in PaintMode::ALL {
                                                    ui.selectable_value(
                                                        &mut self.paint_mode,
                                                        mode,
                                                        mode.label(),
                                                    );
                                                }
                                            });
                                        ui.add(
                                            egui::Slider::new(&mut self.paint_value, 0.0..=1.0)
                                                .text("Value"),
                                        );
                                        ui.add(
                                            egui::Slider::new(&mut self.paint_strength, 0.0..=1.0)
                                                .text("Strength"),
                                        );
                                        ui.add(
                                            egui::Slider::new(
                                                &mut self.paint_radius_pixels,
                                                4.0..=180.0,
                                            )
                                            .text("Radius px"),
                                        );
                                        ui.horizontal(|ui| {
                                            if ui
                                                .add_enabled(
                                                    !self.texture_paint_undo.is_empty(),
                                                    egui::Button::new("Undo stroke"),
                                                )
                                                .clicked()
                                                && let Some((target, before)) =
                                                    self.texture_paint_undo.pop()
                                            {
                                                if let Some(mask) =
                                                    self.painted_masks.get_mut(&target)
                                                {
                                                    self.texture_paint_redo
                                                        .push((target, mask.pixels.clone()));
                                                    mask.pixels = before;
                                                    mask.revision = mask.revision.wrapping_add(1);
                                                    self.scene_revision =
                                                        self.scene_revision.wrapping_add(1);
                                                }
                                            }
                                            if ui
                                                .add_enabled(
                                                    !self.texture_paint_redo.is_empty(),
                                                    egui::Button::new("Redo stroke"),
                                                )
                                                .clicked()
                                                && let Some((target, after)) =
                                                    self.texture_paint_redo.pop()
                                            {
                                                if let Some(mask) =
                                                    self.painted_masks.get_mut(&target)
                                                {
                                                    self.texture_paint_undo
                                                        .push((target, mask.pixels.clone()));
                                                    mask.pixels = after;
                                                    mask.revision = mask.revision.wrapping_add(1);
                                                    self.scene_revision =
                                                        self.scene_revision.wrapping_add(1);
                                                }
                                            }
                                        });
                                        if ui.button("Finish Painting and Apply Graph").clicked() {
                                            self.active_tool = Tool::Select;
                                            self.scene_revision =
                                                self.scene_revision.wrapping_add(1);
                                            self.apply_compositor();
                                        }
                                        ui.small("Overlapping UV islands share painted pixels.");
                                    }
                                });
                            egui::CollapsingHeader::new(RichText::new("Dynamics Fields").strong())
                                .default_open(true)
                                .show(ui, |ui| {
                                    if model_asset.is_none() {
                                        ui.weak("Assign a mesh before enabling dynamics.");
                                        return;
                                    }
                                    let mut enabled = self.dynamics_enabled.contains(&id);
                                    if ui
                                        .checkbox(&mut enabled, "Enable cloth deformation")
                                        .changed()
                                    {
                                        if enabled {
                                            if let Err(message) = self.enable_dynamics_for(id) {
                                                self.logs.push(LogEntry {
                                                    level: "DYNAMICS",
                                                    color: Color32::from_rgb(235, 91, 91),
                                                    message,
                                                });
                                            }
                                        } else {
                                            self.dynamics_enabled.remove(&id);
                                            self.dynamics_cloth.remove(&id);
                                            self.scene_revision =
                                                self.scene_revision.wrapping_add(1);
                                            self.project_dirty = true;
                                        }
                                    }
                                    if let Some(field) = self.dynamics_fields.get(&id) {
                                        ui.separator();
                                        ui.strong(&field.name);
                                        ui.small(format!(
                                            "{} vertex values · revision {}",
                                            field.values.len(),
                                            field.revision
                                        ));
                                        let minimum = field
                                            .values
                                            .iter()
                                            .copied()
                                            .fold(f32::INFINITY, f32::min);
                                        let maximum = field
                                            .values
                                            .iter()
                                            .copied()
                                            .fold(f32::NEG_INFINITY, f32::max);
                                        ui.monospace(format!("Range {minimum:.3} … {maximum:.3}"));
                                        if ui.button("Enter Field Paint mode").clicked() {
                                            self.active_tool = Tool::FieldPaint;
                                        }
                                    }
                                    if self.active_tool == Tool::FieldPaint {
                                        ui.separator();
                                        ui.strong("Brush");
                                        egui::ComboBox::from_id_salt(("field_paint_mode", id))
                                            .selected_text(self.paint_mode.label())
                                            .show_ui(ui, |ui| {
                                                for mode in PaintMode::ALL {
                                                    ui.selectable_value(
                                                        &mut self.paint_mode,
                                                        mode,
                                                        mode.label(),
                                                    );
                                                }
                                            });
                                        ui.add(
                                            egui::Slider::new(&mut self.paint_value, 0.0..=1.0)
                                                .text("Value"),
                                        );
                                        ui.add(
                                            egui::Slider::new(&mut self.paint_strength, 0.0..=1.0)
                                                .text("Strength"),
                                        );
                                        ui.add(
                                            egui::Slider::new(
                                                &mut self.paint_radius_pixels,
                                                4.0..=180.0,
                                            )
                                            .text("Radius px"),
                                        );
                                        ui.horizontal(|ui| {
                                            if ui
                                                .add_enabled(
                                                    !self.field_undo.is_empty(),
                                                    egui::Button::new("Undo stroke"),
                                                )
                                                .clicked()
                                            {
                                                if let Some((target, values)) =
                                                    self.field_undo.pop()
                                                {
                                                    if let Some(field) =
                                                        self.dynamics_fields.get_mut(&target)
                                                    {
                                                        self.field_redo
                                                            .push((target, field.values.clone()));
                                                        field.values = values;
                                                        field.revision =
                                                            field.revision.wrapping_add(1);
                                                        if let Some(cloth) =
                                                            self.dynamics_cloth.get_mut(&target)
                                                        {
                                                            cloth.sync_mobility(field);
                                                        }
                                                        self.scene_revision =
                                                            self.scene_revision.wrapping_add(1);
                                                    }
                                                }
                                            }
                                            ui.small("Red=fixed · green=free");
                                        });
                                    }
                                });
                            egui::CollapsingHeader::new(RichText::new("Material").strong())
                                .default_open(true)
                                .show(ui, |ui| {
                                    ui.horizontal(|ui| {
                                        ui.selectable_value(
                                            &mut self.material_tab,
                                            MaterialTab::Shader,
                                            "Shader",
                                        );
                                        ui.selectable_value(
                                            &mut self.material_tab,
                                            MaterialTab::Surface,
                                            "Surface",
                                        );
                                    });
                                    ui.separator();
                                    if self.material_tab == MaterialTab::Shader {
                                        let resolved = self
                                            .scene
                                            .tree
                                            .resolve_attribute(id, &material_shader_key())
                                            .ok();
                                        let mut shader = resolved
                                            .as_ref()
                                            .and_then(|attribute| match &attribute.value {
                                                Attribute::Text(value) => {
                                                    ShaderMode::from_attribute(value)
                                                }
                                                _ => None,
                                            })
                                            .unwrap_or(ShaderMode::Diffuse);
                                        egui::ComboBox::from_id_salt((
                                            "material_shader",
                                            id.slot,
                                            id.generation,
                                        ))
                                        .selected_text(shader.label())
                                        .show_ui(
                                            ui,
                                            |ui| {
                                                ui.selectable_value(
                                                    &mut shader,
                                                    ShaderMode::Toon,
                                                    "Toon",
                                                );
                                                ui.selectable_value(
                                                    &mut shader,
                                                    ShaderMode::Diffuse,
                                                    "Diffuse",
                                                );
                                            },
                                        );
                                        let current_shader = resolved.as_ref().and_then(
                                            |attribute| match &attribute.value {
                                                Attribute::Text(value) => {
                                                    ShaderMode::from_attribute(value)
                                                }
                                                _ => None,
                                            },
                                        );
                                        if current_shader != Some(shader) {
                                            let _ = self.scene.tree.set_attribute(
                                                id,
                                                material_shader_key(),
                                                AttributeDeclaration::Value(Attribute::Text(
                                                    shader.attribute_value().into(),
                                                )),
                                            );
                                            changed = true;
                                        }
                                        let mut smooth_normals = self
                                            .scene
                                            .tree
                                            .resolve_attribute(id, &material_smooth_normals_key())
                                            .ok()
                                            .and_then(|attribute| match attribute.value {
                                                Attribute::Bool(value) => Some(value),
                                                _ => None,
                                            })
                                            .unwrap_or(true);
                                        if ui
                                        .checkbox(&mut smooth_normals, "Smooth normals")
                                        .on_hover_text(
                                            "Interpolate vertex normals across each face instead \
                                             of using one flat face normal",
                                        )
                                        .changed()
                                    {
                                        let _ = self.scene.tree.set_attribute(
                                            id,
                                            material_smooth_normals_key(),
                                            AttributeDeclaration::Value(Attribute::Bool(
                                                smooth_normals,
                                            )),
                                        );
                                        changed = true;
                                    }
                                    } else {
                                        let mut material_tint = self
                                            .scene
                                            .tree
                                            .resolve_attribute(id, &material_tint_key())
                                            .ok()
                                            .and_then(|attribute| match attribute.value {
                                                Attribute::Color(value) => Some(value),
                                                _ => None,
                                            })
                                            .unwrap_or([1.0; 4]);
                                        ui.horizontal(|ui| {
                                            ui.label("Base color");
                                            if ui
                                                .color_edit_button_rgb(
                                                    (&mut material_tint[..3])
                                                        .try_into()
                                                        .expect("RGB color slice"),
                                                )
                                                .on_hover_text(
                                                    "Multiply the imported or built-in material \
                                                     color. This object attribute is inherited by \
                                                     children and can be changed by scripts.",
                                                )
                                                .changed()
                                            {
                                                let _ = self.scene.tree.set_attribute(
                                                    id,
                                                    material_tint_key(),
                                                    AttributeDeclaration::Value(
                                                        Attribute::Color(material_tint),
                                                    ),
                                                );
                                                changed = true;
                                            }
                                        });
                                        let mut use_imported_optics = self
                                            .scene
                                            .tree
                                            .resolve_attribute(
                                                id,
                                                &material_use_imported_optics_key(),
                                            )
                                            .ok()
                                            .and_then(|attribute| match attribute.value {
                                                Attribute::Bool(value) => Some(value),
                                                _ => None,
                                            })
                                            .unwrap_or(true);
                                        if ui
                                            .checkbox(
                                                &mut use_imported_optics,
                                                "Use imported GLB optical settings",
                                            )
                                            .on_hover_text(
                                                "Apply per-material KHR_materials_transmission \
                                                 and KHR_materials_ior values from glTF/GLB files",
                                            )
                                            .changed()
                                        {
                                            let _ = self.scene.tree.set_attribute(
                                                id,
                                                material_use_imported_optics_key(),
                                                AttributeDeclaration::Value(Attribute::Bool(
                                                    use_imported_optics,
                                                )),
                                            );
                                            changed = true;
                                        }
                                        ui.small(
                                            "Per-material GLB transmission + IOR · preview Fresnel",
                                        );
                                    }
                                    ui.small(
                            RichText::new(
                                "Object attribute · inherited by children · script-writable",
                            )
                            .weak(),
                        );
                                });
                            egui::CollapsingHeader::new(
                                RichText::new("Script Components").strong(),
                            )
                            .default_open(true)
                            .show(ui, |ui| {
                                ui.label("◇  spin_controller.rs");
                                ui.small(
                                    RichText::new("All attributes script-readable and writable")
                                        .weak(),
                                );
                            });
                            if self.advanced {
                                egui::CollapsingHeader::new("Advanced")
                                    .default_open(true)
                                    .show(ui, |ui| {
                                        property(
                                            ui,
                                            "Stable ID",
                                            &format!("{}:{}", id.slot, id.generation),
                                        );
                                        property(ui, "Transform owner", "Editor");
                                        property(ui, "Replication", "Inherited: Server");
                                        ui.checkbox(&mut self.snap, "Quantize transform writes");
                                    });
                            }
                            if changed {
                                if kind == ObjectKind::Camera
                                    && !self.subtree_affects_render_or_shadows(id)
                                {
                                    // A leaf camera has no renderable mesh and cannot alter a
                                    // shadow map. Its view tuple already invalidates the viewport.
                                    self.record_camera_only_undo(previous);
                                } else {
                                    self.record_undo(previous);
                                }
                            } else if lighting_only_changed {
                                self.undo_stack.push(previous);
                                if self.undo_stack.len() > 100 {
                                    self.undo_stack.remove(0);
                                }
                                self.redo_stack.clear();
                                self.project_dirty = true;
                                self.viewport_requested_key = None;
                            }
                            ui.add_space(10.0);
                            ui.add_sized(
                                [ui.available_width(), 28.0],
                                egui::Button::new("+ Add component"),
                            );
                        });
                });
            });
    }

    fn bottom_panel(&mut self, ctx: &egui::Context) {
        egui::TopBottomPanel::bottom("bottom")
            .resizable(true)
            .default_height(215.0)
            .height_range(120.0..=420.0)
            .frame(panel_frame(Color32::from_rgb(26, 28, 35)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    tab(ui, &mut self.bottom_tab, BottomTab::Assets, "Assets");
                    tab(ui, &mut self.bottom_tab, BottomTab::Scripts, "Scripts");
                    tab(ui, &mut self.bottom_tab, BottomTab::Console, "Console");
                    tab(ui, &mut self.bottom_tab, BottomTab::Telemetry, "Telemetry");
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        ui.small(RichText::new("⌕  Search").weak());
                    });
                });
                ui.separator();
                match self.bottom_tab {
                    BottomTab::Assets => self.assets_panel(ui),
                    BottomTab::Scripts => scripts_panel(ui),
                    BottomTab::Console => console_panel(ui, &self.logs),
                    BottomTab::Telemetry => telemetry_panel(ui, self.play_state, &self.performance),
                }
            });
    }

    fn object_handles_panel(&mut self, ui: &mut egui::Ui) {
        let handles = self
            .compositor_nodes
            .iter()
            .filter_map(|node| match &node.settings {
                NodeSettings::ObjectHandle {
                    object_index,
                    label,
                    control,
                    value,
                    minimum,
                    maximum,
                    source_handle,
                } => Some((
                    node.id,
                    *object_index,
                    label.clone(),
                    *control,
                    *value,
                    *minimum,
                    *maximum,
                    *source_handle,
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        if handles.is_empty() {
            ui.centered_and_justified(|ui| {
                ui.label("No Object Handles yet. Add one from Nodes → Add → Input.");
            });
            return;
        }
        let labels = handles
            .iter()
            .map(|(id, _, label, ..)| (*id, label.clone()))
            .collect::<BTreeMap<_, _>>();
        let mut writes = Vec::new();
        egui::ScrollArea::vertical().show(ui, |ui| {
            for (id, object_index, label, control, value, minimum, maximum, source) in &handles {
                let object = self
                    .scene
                    .tree
                    .iter()
                    .nth(*object_index)
                    .map(|(_, node)| node.name.as_str())
                    .unwrap_or("Missing object");
                ui.horizontal(|ui| {
                    ui.label(RichText::new(label).strong());
                    ui.small(format!("{object} · #{id}"));
                    if let Some(source) = source {
                        ui.small(format!(
                            "copies {}",
                            labels
                                .get(source)
                                .map(String::as_str)
                                .unwrap_or("missing handle")
                        ));
                    }
                });
                let effective = self.resolve_object_handle_value(*id).unwrap_or(*value);
                let mut edited = if source.is_some() { effective } else { *value };
                let response = if *control == 0 {
                    ui.add_enabled(
                        source.is_none(),
                        egui::Slider::new(&mut edited, *minimum..=*maximum).show_value(true),
                    )
                } else {
                    ui.add_enabled(
                        source.is_none(),
                        egui::DragValue::new(&mut edited)
                            .range(*minimum..=*maximum)
                            .speed(((*maximum - *minimum).abs() / 100.0).max(0.001)),
                    )
                };
                if response.changed() {
                    writes.push((*id, edited));
                }
                ui.separator();
            }
        });
        if !writes.is_empty() {
            for (id, value) in writes {
                if let Some(node) = self.compositor_nodes.iter_mut().find(|node| node.id == id)
                    && let NodeSettings::ObjectHandle {
                        value: handle_value,
                        ..
                    } = &mut node.settings
                {
                    *handle_value = value;
                }
            }
            self.invalidate_all_object_handles();
            self.compositor_control_started = Some(Instant::now());
            self.compositor_apply_due = Some(Instant::now());
            self.project_dirty = true;
        }
    }

    fn assets_panel(&mut self, ui: &mut egui::Ui) {
        if let Some(receiver) = &self.dialog_result {
            match receiver.try_recv() {
                Ok(path) => {
                    self.dialog_result = None;
                    if let Some(path) = path {
                        self.asset_import_path = path.to_string_lossy().into_owned();
                        self.import_asset(true);
                    }
                }
                Err(mpsc::TryRecvError::Disconnected) => self.dialog_result = None,
                Err(mpsc::TryRecvError::Empty) => {}
            }
        }
        ui.horizontal(|ui| {
            ui.label("Import GLB / glTF / OBJ / PLY / STL");
            let response = ui.add(
                egui::TextEdit::singleline(&mut self.asset_import_path)
                    .hint_text("/path/to/model.obj")
                    .desired_width(320.0),
            );
            let enter = response.lost_focus() && ui.input(|input| input.key_pressed(Key::Enter));
            if ui.button("Import").clicked() || enter {
                self.import_asset(true);
            }
            if ui
                .add_enabled(self.dialog_result.is_none(), egui::Button::new("Load…"))
                .clicked()
            {
                let directory = std::env::current_dir().unwrap_or_else(|_| PathBuf::from("."));
                let repaint = ui.ctx().clone();
                let (sender, receiver) = mpsc::channel();
                self.dialog_result = Some(receiver);
                std::thread::Builder::new()
                    .name("zerofps-file-dialog".into())
                    .spawn(move || {
                        let path = rfd::FileDialog::new()
                            .set_title("Load a 3D model")
                            .set_directory(directory)
                            .add_filter(
                                "Supported 3D models",
                                &["glb", "gltf", "obj", "ply", "stl"],
                            )
                            .add_filter("glTF 2.0", &["glb", "gltf"])
                            .add_filter("Wavefront OBJ", &["obj"])
                            .add_filter("Polygon File Format", &["ply"])
                            .add_filter("Stereolithography", &["stl"])
                            .pick_file();
                        let _ = sender.send(path);
                        repaint.request_repaint();
                    })
                    .expect("file-dialog worker thread should start");
            }
        });
        ui.small(
            RichText::new(
                "Selecting a supported file imports it directly as a selected scene object.",
            )
            .weak(),
        );
    }

    fn activate_compositor_node(&mut self, kind: usize) {
        let Some(mut settings) = NodeSettings::default_for_kind(kind) else {
            return;
        };
        let object_index = self.selected_object_index().unwrap_or(0);
        match &mut settings {
            NodeSettings::ObjectTexture {
                object_index: target,
                ..
            }
            | NodeSettings::PaintedMask {
                object_index: target,
            }
            | NodeSettings::MassDensity {
                object_index: target,
                ..
            }
            | NodeSettings::SpringMesh {
                object_index: target,
                ..
            }
            | NodeSettings::ForceField {
                object_index: target,
                ..
            }
            | NodeSettings::VelocityField {
                object_index: target,
                ..
            }
            | NodeSettings::Simulator {
                object_index: target,
                ..
            }
            | NodeSettings::ObjectSimulator {
                object_index: target,
                ..
            }
            | NodeSettings::Engine {
                object_index: target,
                ..
            }
            | NodeSettings::ObjectHandle {
                object_index: target,
                ..
            }
            | NodeSettings::Output {
                object_index: target,
                ..
            } => *target = object_index,
            _ => {}
        }
        let id = self.compositor_next_id;
        self.compositor_next_id += 1;
        let object_name = self
            .scene
            .tree
            .iter()
            .nth(object_index)
            .map(|(_, object)| object.name.clone())
            .unwrap_or_default();
        let settings_object_name = settings.object_index().map(|_| object_name.clone());
        self.compositor_nodes.push(CompositorNode {
            id,
            object_index,
            object_name,
            settings_object_name,
            settings,
            position: Vec2::ZERO,
        });
        self.compositor_selected_node = id;
        self.compositor_pending_spawn = Some(id);
        self.compositor_pending_output = None;
        self.compositor_eval_cache.clear();
        self.compositor_gpu_cache.clear();
        self.project_dirty = true;
    }

    fn add_omnidirectional_light(&mut self) {
        let previous = self.scene.tree.clone();
        let id = self.scene.add("Light", ObjectKind::Light, None);
        let mut transform = Transform::IDENTITY;
        transform.translation =
            self.camera_target + CoreVec3::new(0.0, 0.0, self.grid_spacing.max(1.0) * 2.0);
        let _ = self.scene.tree.set_local_transform(id, transform);
        self.scene.selected = Some(id);
        self.record_undo(previous);
        self.scene_revision = self.scene_revision.wrapping_add(1);
        self.project_dirty = true;
        self.logs.push(LogEntry {
            level: "SCENE",
            color: Color32::from_rgb(255, 210, 48),
            message: "Added omnidirectional `Light` to the scene".into(),
        });
    }

    fn add_camera_object(&mut self) {
        let previous = self.scene.tree.clone();
        let id = self.scene.add("Camera", ObjectKind::Camera, None);
        let mut transform = Transform::IDENTITY;
        transform.translation = self.camera_target
            + CoreVec3::new(0.0, -self.grid_spacing * 4.0, self.grid_spacing * 2.0);
        let _ = self.scene.tree.set_local_transform(id, transform);
        self.scene.selected = Some(id);
        self.record_undo(previous);
        self.scene_revision = self.scene_revision.wrapping_add(1);
        self.project_dirty = true;
        self.logs.push(LogEntry {
            level: "SCENE",
            color: Color32::from_rgb(255, 210, 48),
            message: "Added `Camera` to the scene".into(),
        });
    }

    fn viewport_lighting(&self, _triangles: &[PreviewTriangle]) -> ViewportLighting {
        let points = self
            .scene
            .tree
            .iter()
            .filter(|(id, _)| self.scene.visible(*id))
            .filter_map(|(_, node)| {
                node.components
                    .iter()
                    .find_map(|component| match component {
                        Component::Light {
                            intensity,
                            color,
                            radius,
                            shadow_resolution,
                        } => Some(ViewportLight {
                            position: node.global_transform().translation,
                            color: *color,
                            intensity: intensity.max(0.0),
                            radius: radius.max(0.0),
                            shadow_resolution: effective_shadow_resolution(
                                *shadow_resolution,
                                self.shadow_quality,
                            ),
                        }),
                        _ => None,
                    })
            })
            .take(MAX_VIEWPORT_LIGHTS)
            .collect();
        ViewportLighting {
            global_enabled: self.global_light_enabled,
            global_shadow_resolution: effective_shadow_resolution(
                self.global_shadow_resolution,
                self.shadow_quality,
            ),
            shadow_filter_radius: shadow_filter_radius(self.shadow_quality),
            shadow_blur_radius: self.shadow_blur_radius,
            points,
            directional_shadow: None,
            point_shadows: None,
        }
    }

    fn sync_compositor_outputs(&mut self) {
        self.resolve_compositor_object_names();
        let object_count = self.scene.tree.iter().count();
        let engine_objects = self
            .scene
            .tree
            .iter()
            .enumerate()
            .filter_map(|(object_index, (_, object))| {
                object
                    .components
                    .iter()
                    .any(|component| {
                        matches!(
                            component,
                            Component::Collider { shape, joint, .. }
                                if ColliderShape::from_storage(shape) == ColliderShape::Cylinder
                                    && CylinderJoint::from_storage(joint)
                                        == CylinderJoint::Engine
                        )
                    })
                    .then_some(object_index)
            })
            .collect::<BTreeSet<_>>();
        let mut seen_transform = BTreeSet::new();
        let mut seen_mesh = BTreeSet::new();
        let mut seen_force = BTreeSet::new();
        let mut seen_engine = BTreeSet::new();
        let removed_ids: BTreeSet<usize> = self
            .compositor_nodes
            .iter()
            .filter_map(|node| match node.settings {
                NodeSettings::ObjectTransform { object_index }
                    if object_index >= object_count || !seen_transform.insert(object_index) =>
                {
                    Some(node.id)
                }
                NodeSettings::ObjectMesh { object_index }
                    if object_index >= object_count || !seen_mesh.insert(object_index) =>
                {
                    Some(node.id)
                }
                NodeSettings::ForceOutput { object_index, .. }
                    if object_index >= object_count || !seen_force.insert(object_index) =>
                {
                    Some(node.id)
                }
                NodeSettings::Engine { object_index, .. }
                    if !engine_objects.contains(&object_index)
                        || !seen_engine.insert(object_index) =>
                {
                    Some(node.id)
                }
                NodeSettings::Output { object_index, .. } if object_index >= object_count => {
                    Some(node.id)
                }
                _ => None,
            })
            .collect();
        if !removed_ids.is_empty() {
            self.compositor_nodes
                .retain(|node| !removed_ids.contains(&node.id));
            self.compositor_links.retain(|(from, _, to, _)| {
                !removed_ids.contains(from) && !removed_ids.contains(to)
            });
        }

        for object_index in 0..object_count {
            let object_name = self
                .scene
                .tree
                .iter()
                .nth(object_index)
                .map(|(_, object)| object.name.clone())
                .unwrap_or_default();
            if !seen_transform.contains(&object_index) {
                let id = self.compositor_next_id;
                self.compositor_next_id = self.compositor_next_id.wrapping_add(1);
                self.compositor_nodes.push(CompositorNode {
                    id,
                    object_index,
                    object_name: object_name.clone(),
                    settings_object_name: Some(object_name.clone()),
                    settings: NodeSettings::ObjectTransform { object_index },
                    position: Vec2::new(80.0, 80.0),
                });
            }
            if !seen_mesh.contains(&object_index) {
                let id = self.compositor_next_id;
                self.compositor_next_id = self.compositor_next_id.wrapping_add(1);
                self.compositor_nodes.push(CompositorNode {
                    id,
                    object_index,
                    object_name: object_name.clone(),
                    settings_object_name: Some(object_name.clone()),
                    settings: NodeSettings::ObjectMesh { object_index },
                    position: Vec2::new(420.0, 80.0),
                });
            }
            if !seen_force.contains(&object_index) {
                let id = self.compositor_next_id;
                self.compositor_next_id = self.compositor_next_id.wrapping_add(1);
                self.compositor_nodes.push(CompositorNode {
                    id,
                    object_index,
                    object_name: object_name.clone(),
                    settings_object_name: Some(object_name.clone()),
                    settings: NodeSettings::ForceOutput {
                        object_index,
                        force: [0.0; 3],
                    },
                    position: Vec2::new(80.0, 340.0),
                });
            }
            if engine_objects.contains(&object_index) && !seen_engine.contains(&object_index) {
                let id = self.compositor_next_id;
                self.compositor_next_id = self.compositor_next_id.wrapping_add(1);
                self.compositor_nodes.push(CompositorNode {
                    id,
                    object_index,
                    object_name: object_name.clone(),
                    settings_object_name: Some(object_name.clone()),
                    settings: NodeSettings::Engine {
                        object_index,
                        throttle: 0.0,
                        torque: 100.0,
                        reverse: false,
                        auto_braking: false,
                        max_rpm: 6_000.0,
                    },
                    position: Vec2::new(340.0, 340.0),
                });
            }
        }
        if !self
            .compositor_nodes
            .iter()
            .any(|node| node.id == self.compositor_selected_node)
        {
            self.compositor_selected_node = self
                .compositor_nodes
                .first()
                .map(|node| node.id)
                .unwrap_or(0);
        }
    }

    fn resolve_compositor_object_names(&mut self) {
        let names = self
            .scene
            .tree
            .iter()
            .map(|(_, object)| object.name.clone())
            .collect::<Vec<_>>();
        for node in &mut self.compositor_nodes {
            if node.object_name.is_empty() {
                node.object_name = names.get(node.object_index).cloned().unwrap_or_default();
            }
            node.object_index =
                object_index_by_name(&names, &node.object_name).unwrap_or(usize::MAX);

            if let Some(index) = node.settings.object_index() {
                let name = node
                    .settings_object_name
                    .get_or_insert_with(|| names.get(index).cloned().unwrap_or_default());
                let resolved = object_index_by_name(&names, name).unwrap_or(usize::MAX);
                node.settings.set_object_index(resolved);
            }
        }
    }

    fn rename_compositor_object_reference(&mut self, previous: &str, renamed: &str) {
        for node in &mut self.compositor_nodes {
            if node.object_name == previous {
                node.object_name = renamed.to_owned();
            }
            if node.settings_object_name.as_deref() == Some(previous) {
                node.settings_object_name = Some(renamed.to_owned());
            }
        }
    }

    fn invalidate_compositor_from(&mut self, node_id: usize) {
        let mut pending = vec![node_id];
        let mut affected = BTreeSet::new();
        while let Some(current) = pending.pop() {
            if !affected.insert(current) {
                continue;
            }
            pending.extend(
                self.compositor_links
                    .iter()
                    .filter_map(|&(from, _, to, _)| (from == current).then_some(to)),
            );
        }
        self.compositor_eval_cache
            .retain(|(cached_node, _, _), _| !affected.contains(cached_node));
        self.compositor_gpu_cache
            .retain(|(cached_node, _, _), _| !affected.contains(cached_node));
    }

    fn compositor_input_source(
        &self,
        to_id: usize,
        to_input: usize,
    ) -> Result<(usize, usize), String> {
        self.compositor_links
            .iter()
            .find_map(|&(from_id, from_output, to, input)| {
                (to == to_id && input == to_input).then_some((from_id, from_output))
            })
            .ok_or_else(|| format!("input {} on node {} is not connected", to_input + 1, to_id))
    }

    fn object_asset_path(&self, object_index: usize) -> Option<&str> {
        self.scene
            .tree
            .iter()
            .nth(object_index)
            .and_then(|(_, node)| {
                node.components
                    .iter()
                    .find_map(|component| match component {
                        Component::Model { asset, .. } => Some(asset.as_str()),
                        _ => None,
                    })
            })
    }

    fn object_node_id(&self, object_index: usize) -> Option<NodeId> {
        self.scene.tree.iter().nth(object_index).map(|(id, _)| id)
    }

    fn resolve_object_handle_value(&self, node_id: usize) -> Option<f32> {
        resolve_object_handle_value(&self.compositor_nodes, node_id)
    }

    fn object_handle_depends_on(&self, start: usize, target: usize) -> bool {
        let mut current = Some(start);
        let mut visited = BTreeSet::new();
        while let Some(id) = current {
            if id == target {
                return true;
            }
            if !visited.insert(id) {
                return true;
            }
            current = self
                .compositor_nodes
                .iter()
                .find(|node| node.id == id)
                .and_then(|node| match node.settings {
                    NodeSettings::ObjectHandle { source_handle, .. } => source_handle,
                    _ => None,
                });
        }
        false
    }

    fn invalidate_all_object_handles(&mut self) {
        let handles = self
            .compositor_nodes
            .iter()
            .filter_map(|node| {
                matches!(node.settings, NodeSettings::ObjectHandle { .. }).then_some(node.id)
            })
            .collect::<Vec<_>>();
        for id in handles {
            self.invalidate_compositor_from(id);
        }
    }

    fn probe_compositor_node(
        &mut self,
        node_id: usize,
        output: usize,
        visiting: &mut BTreeSet<usize>,
    ) -> Result<CompositorProbeValue, String> {
        if !visiting.insert(node_id) {
            return Err("probe found a compositor cycle".into());
        }
        let settings = self
            .compositor_nodes
            .iter()
            .find(|node| node.id == node_id)
            .map(|node| node.settings.clone())
            .ok_or_else(|| "probe source no longer exists".to_owned())?;
        let result = match settings {
            NodeSettings::ConstantValue { value, .. } => Ok(CompositorProbeValue::Number(value)),
            NodeSettings::ObjectHandle { .. } => self
                .resolve_object_handle_value(node_id)
                .map(CompositorProbeValue::Number)
                .ok_or_else(|| "Object Handle has a copy cycle or missing source".into()),
            NodeSettings::Time { scale, modulus, .. } => {
                Ok(CompositorProbeValue::Number(scaled_modulated_time(
                    self.compositor_clock_started.elapsed().as_secs_f32(),
                    scale,
                    modulus,
                )))
            }
            NodeSettings::Position { .. } | NodeSettings::Rotation { .. } => self
                .transform_vector_value(node_id)
                .map(|(values, _)| CompositorProbeValue::Triple(values))
                .ok_or_else(|| "could not evaluate the XYZ value".into()),
            NodeSettings::Algebra { expression } => {
                let program = compositor_graph::compile_algebra_expression(&expression)?;
                let mut inputs = Vec::with_capacity(3);
                for index in 0..3 {
                    inputs.push(
                        self.compositor_input_source(node_id, index)
                            .ok()
                            .map(|(source, source_output)| {
                                self.probe_compositor_node(source, source_output, visiting)
                            })
                            .transpose()?,
                    );
                }
                if inputs
                    .iter()
                    .flatten()
                    .any(|value| matches!(value, CompositorProbeValue::Image(_)))
                {
                    let graph = self
                        .compile_compositor_preview(
                            node_id,
                            output,
                            self.compositor_next_generation,
                            self.compositor_lod_max_dimension.min(1024),
                        )
                        .map_err(|error| error.to_string())?;
                    CpuGraphExecutor::default()
                        .execute(&graph)
                        .map(CompositorProbeValue::Image)
                        .map_err(|error| error.to_string())
                } else if inputs
                    .iter()
                    .flatten()
                    .any(|value| matches!(value, CompositorProbeValue::Triple(_)))
                {
                    let mut result = [0.0; 3];
                    for component in 0..3 {
                        let variables = std::array::from_fn(|index| match &inputs[index] {
                            Some(CompositorProbeValue::Triple(values)) => values[component],
                            Some(CompositorProbeValue::Number(value)) => *value,
                            _ => 0.0,
                        });
                        result[component] =
                            compositor_graph::evaluate_algebra_program(&program, variables)?;
                    }
                    Ok(CompositorProbeValue::Triple(result))
                } else {
                    let variables = std::array::from_fn(|index| match &inputs[index] {
                        Some(CompositorProbeValue::Number(value)) => *value,
                        _ => 0.0,
                    });
                    compositor_graph::evaluate_algebra_program(&program, variables)
                        .map(CompositorProbeValue::Number)
                }
            }
            _ => {
                let graph = self
                    .compile_compositor_preview(
                        node_id,
                        output,
                        self.compositor_next_generation,
                        self.compositor_lod_max_dimension.min(1024),
                    )
                    .map_err(|error| error.to_string())?;
                CpuGraphExecutor::default()
                    .execute(&graph)
                    .map(CompositorProbeValue::Image)
                    .map_err(|error| error.to_string())
            }
        };
        visiting.remove(&node_id);
        result
    }

    fn scalar_node_value(
        &mut self,
        node_id: usize,
        output: usize,
        visiting: &mut BTreeSet<usize>,
    ) -> Option<f32> {
        if !visiting.insert(node_id) {
            return None;
        }
        let settings = self
            .compositor_nodes
            .iter()
            .find(|node| node.id == node_id)?
            .settings
            .clone();
        let value = match settings {
            NodeSettings::ConstantValue { value, .. } => Some(value),
            NodeSettings::ObjectHandle { .. } => self.resolve_object_handle_value(node_id),
            NodeSettings::Time { scale, modulus, .. } => Some(scaled_modulated_time(
                self.compositor_clock_started.elapsed().as_secs_f32(),
                scale,
                modulus,
            )),
            NodeSettings::Algebra { .. } => {
                match self.probe_compositor_node(node_id, output, &mut BTreeSet::new()) {
                    Ok(CompositorProbeValue::Number(value)) => Some(value),
                    _ => None,
                }
            }
            _ => {
                let mut texture_visiting = BTreeSet::new();
                self.evaluate_compositor_node(node_id, output, &mut texture_visiting)
                    .ok()
                    .and_then(|texture| texture.pixels.first().copied())
                    .map(|value| value as f32 / 255.0)
            }
        };
        visiting.remove(&node_id);
        value.filter(|value| value.is_finite())
    }

    fn transform_vector_value(&mut self, node_id: usize) -> Option<([f32; 3], bool)> {
        let settings = self
            .compositor_nodes
            .iter()
            .find(|node| node.id == node_id)?
            .settings
            .clone();
        let mut values = match settings {
            NodeSettings::Position { values } => values,
            NodeSettings::Rotation { degrees } => degrees,
            _ => return None,
        };
        let mut driven = false;
        for axis in 0..3 {
            if let Ok((source, output)) = self.compositor_input_source(node_id, axis) {
                let mut visiting = BTreeSet::new();
                if let Some(value) = self.scalar_node_value(source, output, &mut visiting) {
                    values[axis] = value;
                    driven = true;
                }
            }
        }
        Some((values, driven))
    }

    fn apply_object_transform_graphs(&mut self) {
        let outputs = self
            .compositor_nodes
            .iter()
            .filter_map(|node| match node.settings {
                NodeSettings::ObjectTransform { object_index } => Some((node.id, object_index)),
                _ => None,
            })
            .collect::<Vec<_>>();
        let mut changed = false;
        for (output_node, object_index) in outputs {
            let Some(id) = self.object_node_id(object_index) else {
                continue;
            };
            let Ok(node) = self.scene.tree.node(id) else {
                continue;
            };
            let mut transform = node.local_transform();
            let original = transform;
            if let Ok((source, _)) = self.compositor_input_source(output_node, 0) {
                match self
                    .compositor_nodes
                    .iter()
                    .find(|node| node.id == source)
                    .map(|node| &node.settings)
                {
                    Some(NodeSettings::Position { .. }) => {
                        if let Some((position, _)) = self.transform_vector_value(source) {
                            transform.translation =
                                CoreVec3::new(position[0], position[1], position[2]);
                        }
                    }
                    Some(NodeSettings::ObjectSimulator { .. }) => {
                        if let Some(state) = self.object_simulation_states.get(&source) {
                            transform.translation = state.position;
                        }
                    }
                    _ => {}
                }
            }
            if let Ok((source, source_output)) = self.compositor_input_source(output_node, 1) {
                let degrees = match self
                    .compositor_nodes
                    .iter()
                    .find(|node| node.id == source)
                    .map(|node| &node.settings)
                {
                    Some(NodeSettings::Rotation { .. }) => self
                        .transform_vector_value(source)
                        .map(|(value, _)| CoreVec3::new(value[0], value[1], value[2])),
                    Some(NodeSettings::ObjectSimulator { .. }) if source_output == 1 => self
                        .object_simulation_states
                        .get(&source)
                        .map(|state| state.rotation_degrees),
                    _ => None,
                };
                if let Some(degrees) = degrees {
                    transform.rotation = Quat::from_euler_xyz(CoreVec3::new(
                        degrees.x.to_radians(),
                        degrees.y.to_radians(),
                        degrees.z.to_radians(),
                    ));
                }
            }
            if let Some(joint) = self.joint_simulation_states.get(&id) {
                let base_rotation = if self.compositor_input_source(output_node, 1).is_ok() {
                    transform.rotation
                } else {
                    joint.base_rotation
                };
                transform.rotation = base_rotation
                    * Quat::from_euler_xyz(CoreVec3::new(0.0, 0.0, joint.angle_radians));
            }
            if transform != original && self.scene.tree.set_local_transform(id, transform).is_ok() {
                changed = true;
            }
        }
        if changed {
            self.scene_revision = self.scene_revision.wrapping_add(1);
        }
    }

    fn selected_object_index(&self) -> Option<usize> {
        let selected = self.scene.selected?;
        self.scene
            .tree
            .iter()
            .enumerate()
            .find_map(|(index, (id, _))| (id == selected).then_some(index))
    }

    fn projected_object_extent(&self, object_index: usize) -> Option<f32> {
        let (id, node) = self.scene.tree.iter().nth(object_index)?;
        let path = node
            .components
            .iter()
            .find_map(|component| match component {
                Component::Model { asset, .. } => Some(asset.as_str()),
                _ => None,
            })?;
        let bounds = self
            .imported_assets
            .iter()
            .find(|asset| asset.path == path)?
            .bounds;
        let size = self
            .viewport_depth_key
            .map(|key| key.size)
            .unwrap_or([1280, 720]);
        let center = Pos2::new(size[0] as f32 * 0.5, size[1] as f32 * 0.5);
        let scale = size[0].min(size[1]) as f32 * 0.18 * self.camera_zoom;
        let transform = self.scene.tree.node(id).ok()?.global_transform();
        let mut minimum = Pos2::new(f32::INFINITY, f32::INFINITY);
        let mut maximum = Pos2::new(f32::NEG_INFINITY, f32::NEG_INFINITY);
        let mut projected = 0usize;
        for x in [bounds.0[0], bounds.1[0]] {
            for y in [bounds.0[1], bounds.1[1]] {
                for z in [bounds.0[2], bounds.1[2]] {
                    let world = transform
                        .rotation
                        .rotate(transform.scale.component_mul(CoreVec3::new(x, y, z)))
                        + transform.translation;
                    if let Some(point) = project(
                        [world.x, world.y, world.z],
                        center,
                        scale,
                        self.camera_yaw,
                        self.camera_pitch,
                        0.0,
                        self.camera_target,
                        self.projection_mode,
                        self.grid_spacing,
                    ) {
                        minimum.x = minimum.x.min(point.x);
                        minimum.y = minimum.y.min(point.y);
                        maximum.x = maximum.x.max(point.x);
                        maximum.y = maximum.y.max(point.y);
                        projected += 1;
                    }
                }
            }
        }
        (projected > 0).then_some((maximum.x - minimum.x).max(maximum.y - minimum.y))
    }

    fn ensure_vulkan_compositor(&mut self) -> bool {
        if self.vulkan_compositor.is_some() {
            return true;
        }
        if self.vulkan_compositor_attempted {
            return false;
        }
        self.vulkan_compositor_attempted = true;
        match vulkan_compositor::VulkanGraphWorker::new() {
            Ok(compositor) => {
                self.logs.push(LogEntry {
                    level: "VULKAN",
                    color: Color32::from_rgb(112, 210, 156),
                    message: format!("Compositor connected to {}", compositor.device_name),
                });
                self.vulkan_compositor = Some(compositor);
                true
            }
            Err(message) => {
                self.logs.push(LogEntry {
                    level: "FALLBACK",
                    color: Color32::from_rgb(244, 190, 88),
                    message: format!("Vulkan compositor unavailable; using CPU: {message}"),
                });
                false
            }
        }
    }

    fn evaluate_compositor_node(
        &mut self,
        node_id: usize,
        output: usize,
        visiting: &mut BTreeSet<(usize, usize)>,
    ) -> Result<Arc<TextureAsset>, String> {
        // Transitional safety valve: Vulkan mode must not accidentally run a
        // CPU-only node over an unlimited 4K/8K source while GPU coverage is
        // being completed. This affects only the temporary CPU graph value;
        // source assets and GPU-resident outputs retain full resolution.
        let fallback_lod = if self.render_device == RenderDevice::Vulkan {
            self.compositor_lod_max_dimension.min(1024)
        } else {
            self.compositor_lod_max_dimension
        };
        let cache_key = (node_id, output, self.compositor_lod_max_dimension);
        if let Some(texture) = self.compositor_eval_cache.get(&cache_key) {
            return Ok(Arc::clone(texture));
        }
        if let Some(image) = self.compositor_gpu_cache.get(&cache_key) {
            let texture = Arc::new(TextureAsset {
                name: "compositor-explicit-readback".into(),
                width: image.width,
                height: image.height,
                pixels: image.readback_rgba8()?,
                cached_mips: Vec::new(),
            });
            self.compositor_eval_cache
                .insert(cache_key, Arc::clone(&texture));
            return Ok(texture);
        }
        if !visiting.insert((node_id, output)) {
            return Err("compositor graph contains a cycle".into());
        }
        let settings = self
            .compositor_nodes
            .iter()
            .find(|n| n.id == node_id)
            .map(|n| n.settings.clone())
            .ok_or_else(|| format!("node {} not found", node_id))?;
        let result = match settings {
            NodeSettings::ObjectTexture { object_index, .. } => {
                let path = self
                    .object_asset_path(object_index)
                    .ok_or("Object Texture has no model object selected")?;
                let mesh = self
                    .imported_assets
                    .iter()
                    .find(|asset| asset.path == path)
                    .map(|asset| &asset.mesh)
                    .ok_or("Object Texture model asset is unavailable")?;
                let texture_name = mesh
                    .primitives
                    .iter()
                    .filter_map(|primitive| primitive.material.as_ref())
                    .filter_map(|name| mesh.materials.get(name))
                    .find_map(|material| material.base_color_texture.as_ref())
                    .ok_or("Object Texture source has no base-color texture")?;
                let source = mesh
                    .textures
                    .get(texture_name)
                    .cloned()
                    .map(Arc::new)
                    .ok_or_else(|| format!("texture `{texture_name}` is unavailable"))?;
                Ok(resize_texture_for_lod(&source, fallback_lod))
            }
            NodeSettings::PaintedMask { object_index } => {
                let object = self
                    .object_node_id(object_index)
                    .ok_or("Painted Texture object is unavailable")?;
                let source = self
                    .painted_masks
                    .get(&object)
                    .map(|mask| Arc::new(mask.texture(false)))
                    .ok_or("Object has no painted texture")?;
                Ok(resize_texture_for_lod(&source, fallback_lod))
            }
            NodeSettings::ImageAsset { path } => {
                if path.is_empty() {
                    return Err("Image Asset has no imported image".into());
                }
                let source = if let Some(texture) = self.compositor_image_cache.get(&path) {
                    Arc::clone(texture)
                } else {
                    let image = image::open(&path)
                        .map_err(|e| format!("could not decode `{path}`: {e}"))?
                        .into_rgba8();
                    let texture = Arc::new(TextureAsset {
                        name: std::path::PathBuf::from(&path)
                            .file_name()
                            .and_then(|n| n.to_str())
                            .unwrap_or("compositor-image")
                            .into(),
                        width: image.width(),
                        height: image.height(),
                        pixels: image.into_raw(),
                        cached_mips: Vec::new(),
                    });
                    self.compositor_image_cache
                        .insert(path, Arc::clone(&texture));
                    texture
                };
                Ok(resize_texture_for_lod(&source, fallback_lod))
            }
            NodeSettings::ConstantValue { color, .. } => Ok(Arc::new(TextureAsset {
                name: "compositor-constant".into(),
                width: 1,
                height: 1,
                pixels: color
                    .iter()
                    .map(|c| (c.clamp(0.0, 1.0) * 255.0).round() as u8)
                    .chain(std::iter::once(255))
                    .collect(),
                cached_mips: Vec::new(),
            })),
            NodeSettings::Remap { .. } => {
                let (from_id, from_out) = self.compositor_input_source(node_id, 0)?;
                self.evaluate_compositor_node(from_id, from_out, visiting)
            }
            NodeSettings::TextureMath {
                operation,
                constant,
            } => {
                let (from_id, from_out) = self.compositor_input_source(node_id, 0)?;
                let texture = self.evaluate_compositor_node(from_id, from_out, visiting)?;
                Ok(Arc::new(apply_compositor_math(
                    (*texture).clone(),
                    operation,
                    constant,
                )))
            }
            NodeSettings::Algebra { expression } => {
                let program = compositor_graph::compile_algebra_expression(&expression)?;
                let mut inputs: [Option<Arc<TextureAsset>>; 3] = [None, None, None];
                for (index, target) in inputs.iter_mut().enumerate() {
                    if let Ok((from_id, from_out)) = self.compositor_input_source(node_id, index) {
                        *target = Some(self.evaluate_compositor_node(from_id, from_out, visiting)?);
                    }
                }
                Ok(Arc::new(apply_compositor_algebra(&inputs, &program)))
            }
            NodeSettings::SharpThreshold { .. }
            | NodeSettings::SmoothThreshold { .. }
            | NodeSettings::ImageFilter { .. } => {
                let (from_id, from_out) = self.compositor_input_source(node_id, 0)?;
                self.evaluate_compositor_node(from_id, from_out, visiting)
            }
            NodeSettings::Output { .. } => Err("cannot evaluate Output as image source".into()),
            NodeSettings::TextureCombine {
                mode,
                operation,
                alpha,
            } => {
                let (a_id, a_out) = self.compositor_input_source(node_id, 0)?;
                let a = self.evaluate_compositor_node(a_id, a_out, visiting)?;
                let (b_id, b_out) = self.compositor_input_source(node_id, 1)?;
                let b = self.evaluate_compositor_node(b_id, b_out, visiting)?;
                let alpha_tex = if mode == 1 {
                    self.compositor_links
                        .iter()
                        .find_map(|&(from_id, from_out, to, input)| {
                            (to == node_id && input == 2).then_some((from_id, from_out))
                        })
                        .map(|(fid, fo)| self.evaluate_compositor_node(fid, fo, visiting))
                        .transpose()?
                } else {
                    None
                };
                Ok(Arc::new(combine_compositor_textures(
                    &a,
                    &b,
                    alpha_tex.as_deref(),
                    mode,
                    operation,
                    alpha,
                )))
            }
            NodeSettings::ColorSpaceConvert { from, to } => {
                let (from_id, from_out) = self.compositor_input_source(node_id, 0)?;
                let texture = self.evaluate_compositor_node(from_id, from_out, visiting)?;
                Ok(Arc::new(convert_compositor_color_space(
                    (*texture).clone(),
                    from,
                    to,
                )))
            }
            NodeSettings::ColorDecoder => {
                let (from_id, from_out) = self.compositor_input_source(node_id, 0)?;
                let texture = self.evaluate_compositor_node(from_id, from_out, visiting)?;
                Ok(Arc::new(extract_compositor_channel(
                    (*texture).clone(),
                    output,
                )))
            }
            NodeSettings::Grayscale { mode } => {
                let (from_id, from_out) = self.compositor_input_source(node_id, 0)?;
                let texture = self.evaluate_compositor_node(from_id, from_out, visiting)?;
                Ok(Arc::new(grayscale_compositor_texture(
                    (*texture).clone(),
                    mode,
                )))
            }
            NodeSettings::ColorEncoder => {
                let make_default = |v: u8| TextureAsset {
                    name: "compositor-default".into(),
                    width: 1,
                    height: 1,
                    pixels: vec![v, v, v, 255],
                    cached_mips: Vec::new(),
                };
                let r = match self.compositor_input_source(node_id, 0) {
                    Ok((fid, fo)) => self.evaluate_compositor_node(fid, fo, visiting)?,
                    Err(_) => Arc::new(make_default(0)),
                };
                let g = match self.compositor_input_source(node_id, 1) {
                    Ok((fid, fo)) => self.evaluate_compositor_node(fid, fo, visiting)?,
                    Err(_) => Arc::new(make_default(0)),
                };
                let b = match self.compositor_input_source(node_id, 2) {
                    Ok((fid, fo)) => self.evaluate_compositor_node(fid, fo, visiting)?,
                    Err(_) => Arc::new(make_default(0)),
                };
                let a = match self.compositor_input_source(node_id, 3) {
                    Ok((fid, fo)) => self.evaluate_compositor_node(fid, fo, visiting)?,
                    Err(_) => Arc::new(make_default(255)),
                };
                Ok(Arc::new(join_compositor_channels([&r, &g, &b], Some(&a))))
            }
            NodeSettings::ObjectHandle { .. } => {
                let value = self
                    .resolve_object_handle_value(node_id)
                    .ok_or_else(|| "Object Handle copy cycle or missing source".to_owned())?;
                let channel = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                Ok(Arc::new(TextureAsset {
                    name: "compositor-object-handle".into(),
                    width: 1,
                    height: 1,
                    pixels: vec![channel, channel, channel, 255],
                    cached_mips: Vec::new(),
                }))
            }
            NodeSettings::Time { scale, modulus, .. } => {
                let value = scaled_modulated_time(
                    self.compositor_clock_started.elapsed().as_secs_f32(),
                    scale,
                    modulus,
                );
                let channel = (value.clamp(0.0, 1.0) * 255.0).round() as u8;
                Ok(Arc::new(TextureAsset {
                    name: "compositor-time".into(),
                    width: 1,
                    height: 1,
                    pixels: vec![channel, channel, channel, 255],
                    cached_mips: Vec::new(),
                }))
            }
            NodeSettings::Debug => {
                let (from_id, from_out) = self.compositor_input_source(node_id, 0)?;
                self.evaluate_compositor_node(from_id, from_out, visiting)
            }
            NodeSettings::ObjectTransform { .. }
            | NodeSettings::ObjectMesh { .. }
            | NodeSettings::Position { .. }
            | NodeSettings::Rotation { .. }
            | NodeSettings::ObjectSimulator { .. }
            | NodeSettings::ForceOutput { .. }
            | NodeSettings::Engine { .. } => {
                Err("object output nodes cannot be evaluated as textures".into())
            }
            NodeSettings::MassDensity { .. }
            | NodeSettings::SpringMesh { .. }
            | NodeSettings::ForceField { .. }
            | NodeSettings::VelocityField { .. }
            | NodeSettings::Simulator { .. } => {
                Err("particle dynamics nodes cannot be evaluated as textures".into())
            }
        };
        visiting.remove(&(node_id, output));
        if let Ok(texture) = &result {
            self.compositor_eval_cache
                .insert(cache_key, Arc::clone(texture));
        }
        result
    }

    fn apply_compositor(&mut self) {
        let outputs: Vec<(usize, usize)> = self
            .compositor_nodes
            .iter()
            .filter_map(|node| match node.settings {
                NodeSettings::Output { object_index, .. } => Some((node.id, object_index)),
                _ => None,
            })
            .collect();
        if outputs.is_empty() {
            return;
        }
        self.compositor_graph_queue.clear();
        let compile_started = Instant::now();
        for (output_id, object_index) in outputs {
            let target = self.object_node_id(object_index);
            if self.compositor_input_source(output_id, 0).is_err() {
                if let Some(target) = target {
                    let before = self.compositor_texture_overrides.len();
                    self.compositor_texture_overrides
                        .retain(|(id, _)| *id != target);
                    if before != self.compositor_texture_overrides.len() {
                        self.texture_revision = self.texture_revision.wrapping_add(1);
                    }
                }
                continue;
            }
            let Some(target) = target else {
                continue;
            };
            let projected_extent = self.projected_object_extent(object_index).unwrap_or(1024.0);
            let lod = select_compositor_lod_for_backend(
                projected_extent,
                self.compositor_lod_max_dimension,
                self.render_device,
            );
            self.compositor_next_generation = self.compositor_next_generation.wrapping_add(1);
            let generation = self.compositor_next_generation;
            match self.compile_compositor_graph(output_id, generation, lod) {
                Ok(compiled) if compiled.channel == 0 => {
                    self.compositor_graph_queue
                        .push_back(PendingCompositorGraph {
                            graph: Arc::new(compiled.graph),
                            target,
                        });
                }
                Ok(_) => {}
                Err(message) => self.logs.push(LogEntry {
                    level: "ERROR",
                    color: Color32::from_rgb(235, 91, 91),
                    message: format!(
                        "Could not compile compositor output for object {}: {message}",
                        object_index + 1
                    ),
                }),
            }
        }
        self.performance
            .graph_compile
            .record(compile_started.elapsed());
        if let Some(started) = self.compositor_control_started {
            self.performance
                .control_to_graph_apply
                .record(started.elapsed());
        }
        if self.vulkan_waiting_generation.is_none() {
            self.submit_next_compositor_graph();
        }
    }

    fn submit_next_compositor_graph(&mut self) {
        let Some(pending) = self.compositor_graph_queue.pop_front() else {
            return;
        };
        let generation = pending.graph.generation;
        self.vulkan_latest_generation = generation;
        self.compositor_pending_target = Some(pending.target);
        self.vulkan_waiting_generation = Some(generation);
        if self.render_device == RenderDevice::Vulkan && self.ensure_vulkan_compositor() {
            self.vulkan_compositor
                .as_ref()
                .expect("Vulkan graph worker was initialized")
                .submit_latest(pending.graph);
        } else {
            self.cpu_compositor.submit_latest(pending.graph);
        }
    }

    fn poll_compositor_apply(&mut self, ctx: &egui::Context) {
        let Some(due) = self.compositor_apply_due else {
            return;
        };
        if Instant::now() >= due {
            self.compositor_apply_due = None;
            self.apply_compositor();
        } else {
            ctx.request_repaint_after(due.saturating_duration_since(Instant::now()));
        }
    }

    fn tick_compositor_time(&mut self, ctx: &egui::Context) {
        let time_nodes: Vec<usize> = self
            .compositor_nodes
            .iter()
            .filter_map(|node| {
                matches!(
                    node.settings,
                    NodeSettings::Time {
                        live_update: true,
                        ..
                    }
                )
                .then_some(node.id)
            })
            .collect();
        if time_nodes.is_empty() {
            return;
        }
        let mut reachable = BTreeSet::new();
        let mut pending = time_nodes.clone();
        while let Some(node) = pending.pop() {
            if !reachable.insert(node) {
                continue;
            }
            pending.extend(
                self.compositor_links
                    .iter()
                    .filter_map(|&(from, _, to, _)| (from == node).then_some(to)),
            );
        }
        let drives_output = self.compositor_nodes.iter().any(|node| {
            reachable.contains(&node.id)
                && matches!(
                    node.settings,
                    NodeSettings::Output { .. } | NodeSettings::Debug
                )
        });
        if !drives_output {
            return;
        }

        let now = Instant::now();
        if now >= self.compositor_next_time_tick {
            let compositor_busy =
                self.vulkan_waiting_generation.is_some() || !self.compositor_graph_queue.is_empty();
            if compositor_busy {
                // Do not continually replace a partially evaluated multi-object
                // batch. Sample time again as soon as the current batch has
                // reached every assigned Output node.
                self.compositor_next_time_tick = now + Duration::from_millis(4);
            } else {
                self.compositor_next_time_tick = now + Duration::from_millis(16);
                for node in time_nodes {
                    self.invalidate_compositor_from(node);
                }
                self.compositor_apply_due = Some(now);
            }
        }
        ctx.request_repaint_after(
            self.compositor_next_time_tick
                .saturating_duration_since(Instant::now()),
        );
    }

    fn poll_vulkan_compositor(&mut self, ctx: &egui::Context) {
        enum CompletedGraph {
            Cpu(compositor_cpu::CpuGraphResult),
            Vulkan(vulkan_compositor::GraphResult),
        }
        let mut newest = None;
        while let Some(result) = self.cpu_compositor.try_result() {
            newest = Some(CompletedGraph::Cpu(result));
        }
        if let Some(worker) = &self.vulkan_compositor {
            while let Some(result) = worker.try_result() {
                newest = Some(CompletedGraph::Vulkan(result));
            }
        }
        let Some(result) = newest else {
            if self.vulkan_waiting_generation.is_some() {
                ctx.request_repaint_after(Duration::from_millis(8));
            }
            return;
        };
        let generation = match &result {
            CompletedGraph::Cpu(result) => result.generation,
            CompletedGraph::Vulkan(result) => result.generation,
        };
        if generation != self.vulkan_latest_generation {
            ctx.request_repaint_after(Duration::from_millis(8));
            return;
        }
        self.vulkan_waiting_generation = None;
        if let Some(started) = self.compositor_control_started {
            self.performance
                .control_to_composite_ready
                .record(started.elapsed());
        }
        let Some(target) = self.compositor_pending_target.take() else {
            self.submit_next_compositor_graph();
            return;
        };
        let completed = match result {
            CompletedGraph::Cpu(result) => {
                self.performance.graph_evaluation.record(result.worker_time);
                result.texture.map(TextureOverride::Cpu)
            }
            CompletedGraph::Vulkan(result) => {
                self.performance
                    .compositor_vulkan_submit
                    .record(result.worker_time);
                result.texture.map(TextureOverride::Gpu)
            }
        };
        match completed {
            Ok(texture) => {
                if let Some((_, current)) = self
                    .compositor_texture_overrides
                    .iter_mut()
                    .find(|(id, _)| *id == target)
                {
                    *current = texture;
                } else {
                    self.compositor_texture_overrides.push((target, texture));
                }
                self.texture_revision = self.texture_revision.wrapping_add(1);
                if let Some(started) = self.compositor_control_started {
                    self.compositor_present_revision = Some((self.texture_revision, started));
                }
            }
            Err(message) => {
                self.logs.push(LogEntry {
                    level: "ERROR",
                    color: Color32::from_rgb(235, 91, 91),
                    message: format!("Compositor graph execution failed: {message}"),
                });
            }
        }
        self.submit_next_compositor_graph();
        ctx.request_repaint();
    }

    fn schedule_compositor_lod_update(&mut self, ctx: &egui::Context) {
        if self.render_device == RenderDevice::Vulkan {
            return;
        }
        let object_index = self
            .compositor_nodes
            .iter()
            .find_map(|node| match node.settings {
                NodeSettings::Output { object_index, .. } => Some(object_index),
                _ => None,
            });
        let Some(projected) = object_index.and_then(|index| self.projected_object_extent(index))
        else {
            return;
        };
        if select_compositor_lod_for_backend(
            projected,
            self.compositor_lod_max_dimension,
            self.render_device,
        ) != self.compositor_lod_max_dimension
        {
            let due = Instant::now() + Duration::from_millis(100);
            self.compositor_apply_due.get_or_insert(due);
            ctx.request_repaint_after(Duration::from_millis(100));
        }
    }

    fn enable_dynamics_for(&mut self, id: NodeId) -> Result<(), String> {
        let path = self
            .scene
            .tree
            .node(id)
            .ok()
            .and_then(|node| {
                node.components
                    .iter()
                    .find_map(|component| match component {
                        Component::Model { asset, .. } => Some(asset.as_str()),
                        _ => None,
                    })
            })
            .ok_or_else(|| "Selected object has no mesh".to_owned())?;
        let mesh = self
            .imported_assets
            .iter()
            .find(|asset| asset.path == path)
            .map(|asset| &asset.mesh)
            .ok_or_else(|| "Selected mesh asset is not loaded".to_owned())?;
        let field = self
            .dynamics_fields
            .entry(id)
            .or_insert_with(|| MeshScalarField::mobility_for_mesh(mesh))
            .clone();
        self.dynamics_cloth.insert(
            id,
            ClothState::new(mesh, &field, self.dynamics_settings.clone()),
        );
        self.dynamics_enabled.insert(id);
        self.scene_revision = self.scene_revision.wrapping_add(1);
        self.project_dirty = true;
        Ok(())
    }

    fn tick_dynamics(&mut self, ctx: &egui::Context) {
        let graph_objects = self
            .scene
            .tree
            .iter()
            .enumerate()
            .filter_map(|(object_index, (id, _))| self.spring_graph_nodes(object_index).map(|_| id))
            .collect::<Vec<_>>();
        for id in graph_objects {
            if !self.dynamics_enabled.contains(&id) {
                let _ = self.enable_dynamics_for(id);
            }
        }
        let now = Instant::now();
        let elapsed = now
            .saturating_duration_since(self.dynamics_last_tick)
            .as_secs_f32()
            .min(0.1);
        self.dynamics_last_tick = now;
        let has_rigid_simulation = self.compositor_nodes.iter().any(|node| {
            matches!(
                node.settings,
                NodeSettings::ObjectSimulator { .. } | NodeSettings::Engine { .. }
            )
        }) || self.scene.tree.iter().any(|(_, object)| {
            object.components.iter().any(|component| {
                matches!(
                    component,
                    Component::Collider { shape, joint, .. }
                        if ColliderShape::from_storage(shape) == ColliderShape::Cylinder
                            && CylinderJoint::from_storage(joint) == CylinderJoint::Wheel
                )
            })
        });
        if (!self.dynamics_running && !self.dynamics_single_step)
            || (self.dynamics_enabled.is_empty() && !has_rigid_simulation)
        {
            return;
        }
        let dt = 1.0 / 60.0;
        if self.dynamics_single_step {
            self.dynamics_accumulator = dt;
        } else {
            self.dynamics_accumulator += elapsed;
        }
        let mut stepped = false;
        while self.dynamics_accumulator >= dt {
            self.dynamics_accumulator -= dt;
            self.dynamics_time += dt;
            self.step_engine_joints(dt);
            self.step_object_simulators(dt);
            for id in self.dynamics_enabled.clone() {
                let node_dynamics = self.node_particle_dynamics(id);
                if let (Some(field), Some(cloth)) = (
                    self.dynamics_fields.get(&id),
                    self.dynamics_cloth.get_mut(&id),
                ) {
                    if let Some((
                        masses,
                        settings,
                        forces,
                        velocities,
                        velocity_blend,
                        _velocity_mode,
                        gravity,
                        time_scale,
                    )) = node_dynamics
                    {
                        cloth.settings = settings;
                        cloth.sync_particle_masses(field, &masses);
                        let simulation_dt = dt * time_scale.clamp(0.0, 4.0);
                        if simulation_dt > f32::EPSILON {
                            if let Some(velocities) = velocities.as_deref() {
                                cloth.advect_velocity_field(
                                    simulation_dt,
                                    velocities,
                                    velocity_blend,
                                );
                            }
                            cloth.step_with_fields(
                                simulation_dt,
                                self.dynamics_time * time_scale,
                                &self.dynamics_wind,
                                forces.as_deref(),
                                gravity,
                            );
                        }
                    } else {
                        cloth.settings = self.dynamics_settings.clone();
                        cloth.sync_mobility(field);
                        cloth.step(dt, self.dynamics_time, &self.dynamics_wind);
                    }
                    stepped = true;
                }
            }
        }
        self.dynamics_single_step = false;
        if stepped {
            self.scene_revision = self.scene_revision.wrapping_add(1);
            ctx.request_repaint();
        }
        ctx.request_repaint_after(Duration::from_millis(8));
    }

    fn step_object_simulators(&mut self, dt: f32) {
        let mut simulators = self
            .compositor_nodes
            .iter()
            .filter_map(|node| match node.settings {
                NodeSettings::ObjectSimulator {
                    object_index,
                    mass,
                    gravity,
                    linear_velocity,
                    angular_velocity,
                    linear_damping,
                    angular_damping,
                    ..
                } => Some((
                    node.id,
                    object_index,
                    mass,
                    gravity,
                    linear_velocity,
                    angular_velocity,
                    linear_damping,
                    angular_damping,
                )),
                _ => None,
            })
            .collect::<Vec<_>>();
        // A camera is a viewpoint, not a rigid body. Keep this runtime guard so
        // legacy projects cannot give one mass, gravity, or collision response
        // through an Object Simulator node.
        simulators.retain(|(_, object_index, ..)| {
            self.object_node_id(*object_index)
                .is_some_and(|id| self.scene.participates_in_physics(id))
        });
        let simulator_roots = simulators
            .iter()
            .filter_map(|simulator| {
                self.object_node_id(simulator.1)
                    .map(|root| (root, simulator.0))
            })
            .collect::<HashMap<_, _>>();
        let state_snapshot = self.object_simulation_states.clone();
        let mut coupling_forces = HashMap::<usize, CoreVec3>::new();
        let mut coupling_torques = HashMap::<usize, CoreVec3>::new();
        for (&child_root, &child_simulator) in &simulator_roots {
            let Some(child_state) = state_snapshot.get(&child_simulator) else {
                continue;
            };
            let mut ancestor = self
                .scene
                .tree
                .node(child_root)
                .ok()
                .and_then(|node| node.parent());
            let mut parent_simulator = None;
            while let Some(id) = ancestor {
                if let Some(simulator) = simulator_roots.get(&id) {
                    parent_simulator = Some(*simulator);
                    break;
                }
                ancestor = self.scene.tree.node(id).ok().and_then(|node| node.parent());
            }
            let Some(parent_simulator) = parent_simulator else {
                continue;
            };
            let (stiffness, damping, cutoff, free_axle) = self
                .scene
                .tree
                .node(child_root)
                .ok()
                .and_then(|node| {
                    let rotation = node.global_transform().rotation;
                    node.components
                        .iter()
                        .find_map(|component| match component {
                            Component::Collider {
                                coupling_stiffness,
                                coupling_damping,
                                force_cutoff,
                                shape,
                                joint,
                                ..
                            } => {
                                let joint = CylinderJoint::from_storage(joint);
                                let axle = (ColliderShape::from_storage(shape)
                                    == ColliderShape::Cylinder
                                    && joint != CylinderJoint::None)
                                    .then(|| {
                                        rotation.rotate(CoreVec3::new(0.0, 0.0, 1.0)).normalized()
                                    });
                                Some((*coupling_stiffness, *coupling_damping, *force_cutoff, axle))
                            }
                            _ => None,
                        })
                })
                .unwrap_or((0.5, 1.0, 0.01, None));
            let displacement = child_state.position - child_state.initial_position;
            let force = force_cut_vector(
                displacement * -stiffness.max(0.0)
                    + child_state.linear_velocity * -damping.max(0.0),
                cutoff,
            );
            let angular_displacement = (child_state.rotation_degrees
                - child_state.initial_rotation_degrees)
                * (std::f32::consts::PI / 180.0);
            let angular_velocity = child_state.angular_velocity * (std::f32::consts::PI / 180.0);
            let mut torque = force_cut_vector(
                angular_displacement * -stiffness.max(0.0) + angular_velocity * -damping.max(0.0),
                cutoff,
            );
            if let Some(axle) = free_axle {
                // Wheel and engine joints retain parent coupling while leaving
                // the cylinder's local-Z axle unconstrained.
                torque = torque - axle * torque.dot(axle);
            }
            let child_force = coupling_forces
                .entry(child_simulator)
                .or_insert(CoreVec3::ZERO);
            *child_force = *child_force + force;
            let parent_force = coupling_forces
                .entry(parent_simulator)
                .or_insert(CoreVec3::ZERO);
            *parent_force = *parent_force - force;
            let child_torque = coupling_torques
                .entry(child_simulator)
                .or_insert(CoreVec3::ZERO);
            *child_torque = *child_torque + torque;
            let parent_torque = coupling_torques
                .entry(parent_simulator)
                .or_insert(CoreVec3::ZERO);
            *parent_torque = *parent_torque - torque;
        }
        for (
            node_id,
            object_index,
            fallback_mass,
            gravity,
            initial_linear,
            initial_angular,
            linear_damping,
            angular_damping,
        ) in simulators
        {
            let Some(root_id) = self.object_node_id(object_index) else {
                continue;
            };
            let authored = self
                .scene
                .tree
                .node(root_id)
                .ok()
                .map(|node| node.local_transform())
                .unwrap_or(Transform::IDENTITY);
            let initial_position = self
                .compositor_input_source(node_id, 0)
                .ok()
                .and_then(|(source, _)| self.transform_vector_value(source))
                .map(|(value, _)| CoreVec3::new(value[0], value[1], value[2]))
                .unwrap_or(authored.translation);
            let initial_rotation = self
                .compositor_input_source(node_id, 1)
                .ok()
                .and_then(|(source, _)| self.transform_vector_value(source))
                .map(|(value, _)| CoreVec3::new(value[0], value[1], value[2]))
                .unwrap_or_else(|| {
                    let euler = authored.rotation.to_euler_xyz();
                    CoreVec3::new(
                        euler.x.to_degrees(),
                        euler.y.to_degrees(),
                        euler.z.to_degrees(),
                    )
                });
            let external_force = self.subtree_force(root_id);
            let body_colliders = self
                .scene
                .tree
                .iter()
                .filter(|(id, _)| self.node_is_in_subtree(root_id, *id))
                .filter(|(id, _)| self.scene.participates_in_physics(*id))
                .filter_map(|(collider_id, node)| {
                    let transform = node.global_transform();
                    node.components.iter().find_map(|component| {
                        let Component::Collider {
                            center,
                            half_extents,
                            restitution,
                            shape,
                            friction,
                            friction_margin_percent,
                            mass,
                            radius,
                            joint,
                            ..
                        } = component
                        else {
                            return None;
                        };
                        let world_center = transform.translation
                            + transform
                                .rotation
                                .rotate(transform.scale.component_mul(*center));
                        let collider_shape = ColliderShape::from_storage(shape);
                        let extent_z = collider_vertical_extent(
                            collider_shape,
                            *half_extents,
                            transform.scale,
                            transform.rotation,
                        );
                        Some((
                            collider_id,
                            world_center,
                            extent_z,
                            restitution.clamp(0.0, 1.0),
                            collider_shape,
                            transform.rotation,
                            *friction,
                            mass.max(0.0),
                            radius.max(0.001)
                                * transform.scale.x.abs().max(transform.scale.y.abs()),
                            friction_margin_percent.max(0.0),
                            CylinderJoint::from_storage(joint),
                        ))
                    })
                })
                .collect::<Vec<_>>();
            let scene_body_position = self
                .scene
                .tree
                .node(root_id)
                .ok()
                .map(|node| node.global_transform().translation)
                .unwrap_or(authored.translation);
            let general_body_colliders = self
                .scene
                .tree
                .iter()
                .filter(|(id, _)| self.node_is_in_subtree(root_id, *id))
                .filter(|(id, _)| self.scene.participates_in_physics(*id))
                .flat_map(|(_, node)| {
                    let transform = node.global_transform();
                    node.components
                        .iter()
                        .filter_map(move |component| world_collider(transform, component))
                })
                .collect::<Vec<_>>();
            let general_external_colliders = self
                .scene
                .tree
                .iter()
                .filter(|(id, _)| !self.node_is_in_subtree(root_id, *id))
                .filter(|(id, _)| self.scene.participates_in_physics(*id))
                .flat_map(|(_, node)| {
                    let transform = node.global_transform();
                    node.components
                        .iter()
                        .filter_map(move |component| world_collider(transform, component))
                })
                .collect::<Vec<_>>();
            let support_surface =
                collider_support_surface(scene_body_position, &general_external_colliders);
            let body_mass = body_colliders
                .iter()
                .map(|(_, _, _, _, _, _, _, mass, _, _, _)| *mass)
                .sum::<f32>();
            let body_mass = if body_mass > 1.0e-6 {
                body_mass
            } else {
                fallback_mass.max(1.0e-6)
            };
            let state =
                self.object_simulation_states
                    .entry(node_id)
                    .or_insert(ObjectSimulationState {
                        initial_position,
                        initial_rotation_degrees: initial_rotation,
                        position: initial_position,
                        rotation_degrees: initial_rotation,
                        linear_velocity: CoreVec3::new(
                            initial_linear[0],
                            initial_linear[1],
                            initial_linear[2],
                        ),
                        angular_velocity: CoreVec3::new(
                            initial_angular[0],
                            initial_angular[1],
                            initial_angular[2],
                        ),
                    });
            state.linear_velocity = state.linear_velocity + external_force * (dt / body_mass);
            if let Some(force) = coupling_forces.get(&node_id) {
                state.linear_velocity = state.linear_velocity + *force * (dt / body_mass);
            }
            if let Some(torque) = coupling_torques.get(&node_id) {
                state.angular_velocity = state.angular_velocity
                    + *torque * (dt * 180.0 / std::f32::consts::PI / body_mass);
            }
            if gravity {
                state.linear_velocity.z -= 9.81 * dt;
            }
            if let Some((floor_z, surface_friction, surface_friction_tolerance)) = support_surface {
                let angular_radians = state.angular_velocity * std::f32::consts::PI / 180.0;
                let mut contact_force = CoreVec3::ZERO;
                let mut contact_torque = CoreVec3::ZERO;
                let mut maximum_penetration = 0.0_f32;
                let mut contact_restitution = 0.0_f32;
                const CONTACT_SKIN: f32 = 1.0e-4;
                let active_contact_count = body_colliders
                    .iter()
                    .filter(|(_, center, extent_z, _, _, _, _, _, _, margin, _)| {
                        let tolerance = (friction_contact_tolerance(*extent_z, *margin)
                            + surface_friction_tolerance)
                            .max(CONTACT_SKIN);
                        floor_z - (center.z - extent_z) >= -tolerance
                    })
                    .count();
                let (active_contacts, has_active_contact) =
                    contact_count_for_solver(active_contact_count);
                let support_load = body_mass * 9.81 / active_contacts;
                let mut joint_reaction_torques = Vec::<(NodeId, f32, f32)>::new();
                for (
                    collider_id,
                    center,
                    extent_z,
                    restitution,
                    shape,
                    rotation,
                    friction,
                    collider_mass,
                    radius,
                    friction_margin_percent,
                    joint,
                ) in &body_colliders
                {
                    let penetration = floor_z - (center.z - extent_z);
                    let tolerance =
                        (friction_contact_tolerance(*extent_z, *friction_margin_percent)
                            + surface_friction_tolerance)
                            .max(CONTACT_SKIN);
                    if penetration < -tolerance {
                        continue;
                    }
                    maximum_penetration = maximum_penetration.max(penetration.max(0.0));
                    contact_restitution = contact_restitution.max(*restitution);
                    let wheel_arm = CoreVec3::new(0.0, 0.0, -*extent_z);
                    let contact_point = *center + wheel_arm;
                    let body_arm = contact_point - state.position;
                    let axle = rotation.rotate(CoreVec3::new(0.0, 0.0, 1.0)).normalized();
                    let joint_surface_velocity =
                        if *shape == ColliderShape::Cylinder && *joint != CylinderJoint::None {
                            self.joint_simulation_states
                                .get(collider_id)
                                .map(|state| (axle * state.angular_velocity).cross(wheel_arm))
                                .unwrap_or(CoreVec3::ZERO)
                        } else {
                            CoreVec3::ZERO
                        };
                    let contact_velocity = state.linear_velocity
                        + angular_radians.cross(body_arm)
                        + joint_surface_velocity;
                    let tangent = CoreVec3::new(contact_velocity.x, contact_velocity.y, 0.0);
                    let body_contact_mass = body_mass / active_contacts;
                    let joint_inertia = (*shape == ColliderShape::Cylinder
                        && *joint != CylinderJoint::None)
                        .then(|| (0.5 * collider_mass.max(1.0e-6) * radius.powi(2)).max(1.0e-6));
                    let effective_contact_mass = joint_inertia
                        .map(|inertia| {
                            let rolling = axle.cross(CoreVec3::new(0.0, 0.0, 1.0));
                            let rolling_weight =
                                if tangent.length() > 1.0e-6 && rolling.length() > 1.0e-6 {
                                    tangent
                                        .normalized()
                                        .dot(rolling.normalized())
                                        .powi(2)
                                        .clamp(0.0, 1.0)
                                } else {
                                    0.0
                                };
                            coupled_contact_mass(
                                body_contact_mass,
                                inertia,
                                *radius,
                                rolling_weight,
                            )
                        })
                        .unwrap_or(body_contact_mass);
                    let wheel_friction =
                        collider_friction_coefficient(*shape, *rotation, *friction, tangent);
                    let friction_force = contact_friction_force(
                        tangent,
                        (wheel_friction * surface_friction).sqrt(),
                        support_load,
                        effective_contact_mass,
                        dt,
                    );
                    contact_force = contact_force + friction_force;
                    contact_torque = contact_torque + body_arm.cross(friction_force);
                    if let Some(inertia) = joint_inertia {
                        joint_reaction_torques.push((
                            *collider_id,
                            wheel_arm.cross(friction_force).dot(axle),
                            inertia,
                        ));
                    }
                }
                state.linear_velocity = state.linear_velocity + contact_force * (dt / body_mass);
                let angular_acceleration = contact_torque * (1.0 / body_mass);
                state.angular_velocity = state.angular_velocity
                    + angular_acceleration * (dt * 180.0 / std::f32::consts::PI);
                for (joint_id, torque, inertia) in joint_reaction_torques {
                    if let Some(joint_state) = self.joint_simulation_states.get_mut(&joint_id) {
                        joint_state.angular_velocity += torque * dt / inertia;
                    }
                }
                if maximum_penetration > 0.0 {
                    state.position.z += maximum_penetration;
                }
                if has_active_contact && state.linear_velocity.z < 0.0 {
                    state.linear_velocity.z = resolve_contact_normal_velocity(
                        state.linear_velocity.z,
                        contact_restitution,
                    );
                }
            }
            resolve_general_collider_contacts(
                &mut state.position,
                &mut state.linear_velocity,
                scene_body_position,
                &general_body_colliders,
                &general_external_colliders,
                dt,
            );
            state.position = state.position + state.linear_velocity * dt;
            state.rotation_degrees = state.rotation_degrees + state.angular_velocity * dt;
            state.linear_velocity =
                state.linear_velocity * (-linear_damping.clamp(0.0, 1.0) * dt).exp();
            state.angular_velocity =
                state.angular_velocity * (-angular_damping.clamp(0.0, 1.0) * dt).exp();
        }
    }

    fn node_is_in_subtree(&self, root: NodeId, mut candidate: NodeId) -> bool {
        loop {
            if candidate == root {
                return true;
            }
            let Some(parent) = self
                .scene
                .tree
                .node(candidate)
                .ok()
                .and_then(|node| node.parent())
            else {
                return false;
            };
            candidate = parent;
        }
    }

    fn subtree_force(&mut self, root: NodeId) -> CoreVec3 {
        let outputs = self
            .compositor_nodes
            .iter()
            .filter_map(|node| match node.settings {
                NodeSettings::ForceOutput {
                    object_index,
                    force,
                } => self
                    .object_node_id(object_index)
                    .filter(|id| self.node_is_in_subtree(root, *id))
                    .map(|_| (node.id, force)),
                _ => None,
            })
            .collect::<Vec<_>>();
        outputs
            .into_iter()
            .map(|(node_id, mut force)| {
                for axis in 0..3 {
                    if let Ok((source, output)) = self.compositor_input_source(node_id, axis) {
                        let mut visiting = BTreeSet::new();
                        if let Some(value) = self.scalar_node_value(source, output, &mut visiting) {
                            force[axis] = value;
                        }
                    }
                }
                CoreVec3::new(force[0], force[1], force[2])
            })
            .fold(CoreVec3::ZERO, |sum, force| sum + force)
    }

    fn step_engine_joints(&mut self, dt: f32) {
        let roots = self
            .scene
            .tree
            .iter()
            .filter(|(id, _)| self.scene.participates_in_physics(*id))
            .filter_map(|(id, object)| {
                object
                    .components
                    .iter()
                    .any(|component| {
                        matches!(
                            component,
                            Component::Collider { shape, joint, .. }
                                if ColliderShape::from_storage(shape) == ColliderShape::Cylinder
                                    && CylinderJoint::from_storage(joint)
                                        != CylinderJoint::None
                        )
                    })
                    .then_some(id)
            })
            .collect::<BTreeSet<_>>();
        for root in &roots {
            let current_angular_velocity = self
                .joint_simulation_states
                .get(root)
                .map_or(0.0, |state| state.angular_velocity);
            let torque = self.object_engine_torque(*root, current_angular_velocity);
            let Some((base_rotation, axle, inertia, joint)) =
                self.scene.tree.node(*root).ok().and_then(|object| {
                    let transform = object.global_transform();
                    object.components.iter().find_map(|component| {
                        let Component::Collider {
                            shape,
                            joint,
                            radius,
                            mass,
                            ..
                        } = component
                        else {
                            return None;
                        };
                        let joint = CylinderJoint::from_storage(joint);
                        if ColliderShape::from_storage(shape) != ColliderShape::Cylinder
                            || joint == CylinderJoint::None
                        {
                            return None;
                        }
                        let axle = transform
                            .rotation
                            .rotate(CoreVec3::new(0.0, 0.0, 1.0))
                            .normalized();
                        let scaled_radius = radius.max(0.001)
                            * transform.scale.x.abs().max(transform.scale.y.abs());
                        let inertia = (0.5 * mass.max(1.0e-6) * scaled_radius.powi(2)).max(1.0e-6);
                        Some((object.local_transform().rotation, axle, inertia, joint))
                    })
                })
            else {
                continue;
            };
            let state = self
                .joint_simulation_states
                .entry(*root)
                .or_insert(JointSimulationState {
                    base_rotation,
                    angle_radians: 0.0,
                    angular_velocity: 0.0,
                });
            let previous_angle = state.angle_radians;
            match joint {
                CylinderJoint::Engine => {
                    state.angular_velocity += torque.dot(axle) * dt / inertia;
                    state.angle_radians += state.angular_velocity * dt;
                }
                CylinderJoint::Wheel => {
                    // Passive wheels are accelerated by contact reaction
                    // torque in the general friction solver.
                    state.angle_radians += state.angular_velocity * dt;
                }
                CylinderJoint::None => {}
            }
            let angle_delta = state.angle_radians - previous_angle;
            state.angle_radians = state.angle_radians.rem_euclid(std::f32::consts::TAU);
            if angle_delta.abs() > f32::EPSILON
                && let Ok(object) = self.scene.tree.node(*root)
            {
                let mut transform = object.local_transform();
                transform.rotation =
                    transform.rotation * Quat::from_euler_xyz(CoreVec3::new(0.0, 0.0, angle_delta));
                let _ = self.scene.tree.set_local_transform(*root, transform);
            }
        }
        self.joint_simulation_states
            .retain(|root, _| roots.contains(root));
    }

    fn object_engine_torque(&mut self, root: NodeId, angular_velocity: f32) -> CoreVec3 {
        let engines = self
            .compositor_nodes
            .iter()
            .filter_map(|node| match node.settings {
                NodeSettings::Engine {
                    object_index,
                    throttle,
                    torque,
                    reverse,
                    auto_braking,
                    max_rpm,
                } if self.object_node_id(object_index) == Some(root) => {
                    Some((node.id, throttle, torque, reverse, auto_braking, max_rpm))
                }
                _ => None,
            })
            .collect::<Vec<_>>();
        if engines.is_empty() {
            return CoreVec3::ZERO;
        }
        let Ok(object) = self.scene.tree.node(root) else {
            return CoreVec3::ZERO;
        };
        let Some(Component::Collider { shape, joint, .. }) = object
            .components
            .iter()
            .find(|component| matches!(component, Component::Collider { .. }))
        else {
            return CoreVec3::ZERO;
        };
        if ColliderShape::from_storage(shape) != ColliderShape::Cylinder
            || CylinderJoint::from_storage(joint) != CylinderJoint::Engine
        {
            return CoreVec3::ZERO;
        }
        let axle = object
            .global_transform()
            .rotation
            .rotate(CoreVec3::new(0.0, 0.0, 1.0))
            .normalized();
        engines
            .into_iter()
            .map(
                |(node_id, mut throttle, mut torque, reverse, auto_braking, max_rpm)| {
                    for (input, value) in [(0, &mut throttle), (1, &mut torque)] {
                        if let Ok((source, output)) = self.compositor_input_source(node_id, input) {
                            let mut visiting = BTreeSet::new();
                            if let Some(resolved) =
                                self.scalar_node_value(source, output, &mut visiting)
                            {
                                *value = resolved;
                            }
                        }
                    }
                    let direction = if reverse { -1.0 } else { 1.0 };
                    let throttle = throttle.clamp(-1.0, 1.0);
                    let maximum_torque = torque.max(0.0);
                    let applied_torque = if auto_braking {
                        smooth_rpm_governor_torque(
                            throttle,
                            angular_velocity,
                            maximum_torque,
                            max_rpm,
                            direction,
                        )
                    } else {
                        throttle * maximum_torque * direction
                    };
                    axle * applied_torque
                },
            )
            .fold(CoreVec3::ZERO, |sum, torque| sum + torque)
    }

    fn spring_graph_nodes(
        &self,
        object_index: usize,
    ) -> Option<(usize, usize, Option<usize>, Option<usize>, Option<usize>)> {
        let output = self.compositor_nodes.iter().find(|node| {
            node.object_index == object_index
                && matches!(node.settings, NodeSettings::ObjectMesh { .. })
        })?;
        let has_simulator_output = self.compositor_nodes.iter().any(|node| {
            node.object_index == object_index
                && matches!(node.settings, NodeSettings::Simulator { .. })
                && self
                    .compositor_links
                    .iter()
                    .any(|&(from, _, to, input)| from == node.id && to == output.id && input == 0)
        });
        if let Some(simulator) = self.compositor_nodes.iter().find(|node| {
            node.object_index == object_index
                && matches!(node.settings, NodeSettings::Simulator { .. })
                && self
                    .compositor_links
                    .iter()
                    .any(|&(from, _, to, input)| from == node.id && to == output.id && input == 0)
        }) {
            let incoming = self
                .compositor_links
                .iter()
                .filter_map(|&(from, _, to, _)| (to == simulator.id).then_some(from))
                .collect::<Vec<_>>();
            let velocity_id = incoming.iter().copied().find(|source| {
                self.compositor_nodes.iter().any(|node| {
                    node.id == *source
                        && node.object_index == object_index
                        && matches!(node.settings, NodeSettings::VelocityField { .. })
                })
            });
            let force_id = incoming.iter().copied().find(|source| {
                self.compositor_nodes.iter().any(|node| {
                    node.id == *source
                        && node.object_index == object_index
                        && matches!(node.settings, NodeSettings::ForceField { .. })
                })
            });
            if velocity_id.is_none() && force_id.is_none() {
                return None;
            }
            let spring_id = if let Some(force_id) = force_id {
                self.compositor_links
                    .iter()
                    .find_map(|&(from, _, to, input)| {
                        (to == force_id && input == 2).then_some(from)
                    })?
            } else {
                self.compositor_nodes
                    .iter()
                    .find(|node| {
                        node.object_index == object_index
                            && matches!(node.settings, NodeSettings::SpringMesh { .. })
                    })
                    .map(|node| node.id)?
            };
            let force_ok = force_id.is_none_or(|force_id| {
                self.compositor_nodes.iter().any(|node| {
                    node.id == force_id
                        && node.object_index == object_index
                        && matches!(node.settings, NodeSettings::ForceField { .. })
                })
            });
            let spring_ok = self.compositor_nodes.iter().any(|node| {
                node.id == spring_id
                    && node.object_index == object_index
                    && matches!(node.settings, NodeSettings::SpringMesh { .. })
            });
            let force_inputs_ok = force_id.is_none_or(|force_id| {
                [0, 1].into_iter().all(|input| {
                    self.compositor_links
                        .iter()
                        .any(|&(from, _, to, target_input)| {
                            to == force_id
                                && target_input == input
                                && self.compositor_nodes.iter().any(|node| {
                                    node.id == from && node.object_index == object_index
                                })
                        })
                })
            });
            if force_ok && spring_ok && force_inputs_ok {
                return Some((
                    spring_id,
                    output.id,
                    force_id,
                    velocity_id,
                    Some(simulator.id),
                ));
            }
        }
        let spring = self.compositor_nodes.iter().find(|node| {
            node.object_index == object_index
                && matches!(node.settings, NodeSettings::SpringMesh { .. })
        })?;
        if !has_simulator_output
            && self
                .compositor_links
                .iter()
                .any(|&(from, _, to, input)| from == spring.id && to == output.id && input == 0)
        {
            return Some((spring.id, output.id, None, None, None));
        }
        let force =
            self.compositor_nodes.iter().find(|node| {
                node.object_index == object_index
                    && matches!(node.settings, NodeSettings::ForceField { .. })
                    && self.compositor_links.iter().any(|&(from, _, to, input)| {
                        from == node.id && to == output.id && input == 0
                    })
                    && self.compositor_links.iter().any(|&(from, _, to, input)| {
                        from == spring.id && to == node.id && input == 2
                    })
                    && self
                        .compositor_links
                        .iter()
                        .any(|&(_, _, to, input)| to == node.id && input == 0)
                    && self
                        .compositor_links
                        .iter()
                        .any(|&(_, _, to, input)| to == node.id && input == 1)
            })?;
        Some((spring.id, output.id, Some(force.id), None, None))
    }

    fn node_particle_dynamics(
        &self,
        id: NodeId,
    ) -> Option<(
        Vec<f32>,
        ClothSettings,
        Option<Vec<CoreVec3>>,
        Option<Vec<CoreVec3>>,
        f32,
        usize,
        bool,
        f32,
    )> {
        let object_index = self
            .scene
            .tree
            .iter()
            .enumerate()
            .find_map(|(index, (candidate, _))| (candidate == id).then_some(index))?;
        let (spring_id, _, force_id, velocity_id, simulator_id) =
            self.spring_graph_nodes(object_index)?;
        let mass_id = self
            .compositor_links
            .iter()
            .find_map(|&(from, _, to, input)| {
                (to == force_id.unwrap_or(spring_id)
                    && input == if force_id.is_some() { 1 } else { 0 })
                .then_some(from)
            })?;
        let mass_node = self
            .compositor_nodes
            .iter()
            .find(|node| node.id == mass_id)?;
        let NodeSettings::MassDensity {
            base_density,
            scale,
            minimum_mass,
            normalize,
            total_mass,
            ..
        } = mass_node.settings
        else {
            return None;
        };
        let texture_id = self
            .compositor_links
            .iter()
            .find_map(|&(from, _, to, input)| (to == mass_id && input == 0).then_some(from));
        let mass_mask = texture_id
            .and_then(|texture_id| {
                self.compositor_nodes
                    .iter()
                    .find_map(|node| match node.settings {
                        NodeSettings::PaintedMask { object_index } if node.id == texture_id => {
                            Some(object_index)
                        }
                        _ => None,
                    })
            })
            .and_then(|texture_object| self.object_node_id(texture_object))
            .and_then(|mask_object| self.painted_masks.get(&mask_object));
        let path = self.object_asset_path(object_index)?;
        let mesh = self
            .imported_assets
            .iter()
            .find(|asset| asset.path == path)
            .map(|asset| &asset.mesh)?;
        let transform = self.scene.tree.node(id).ok()?.global_transform();
        let uniform_mass;
        let mass_source = if let Some(mask) = mass_mask {
            mask
        } else {
            uniform_mass = PaintedMask::uniform(1, 1, 255);
            &uniform_mass
        };
        let masses = area_weighted_particle_masses(
            mesh,
            mass_source,
            transform.scale,
            base_density,
            scale,
            minimum_mass,
            normalize,
            total_mass,
        );
        let spring_node = self
            .compositor_nodes
            .iter()
            .find(|node| node.id == spring_id)?;
        let NodeSettings::SpringMesh {
            stiffness,
            bend_stiffness,
            damping,
            iterations,
            ..
        } = spring_node.settings
        else {
            return None;
        };
        let mut settings = self.dynamics_settings.clone();
        settings.stretch_compliance = 10.0_f32.powf(-2.0 - 5.0 * stiffness.clamp(0.0, 1.0));
        settings.bend_compliance = 10.0_f32.powf(-1.0 - 4.0 * bend_stiffness.clamp(0.0, 1.0));
        settings.damping = damping.clamp(0.0, 0.95);
        settings.iterations = iterations.clamp(1, 16);
        let forces = force_id.and_then(|force_id| {
            let force_node = self
                .compositor_nodes
                .iter()
                .find(|node| node.id == force_id)?;
            let NodeSettings::ForceField {
                ref formulas,
                scale,
                ..
            } = force_node.settings
            else {
                return None;
            };
            let strength_id = self
                .compositor_links
                .iter()
                .find_map(|&(from, _, to, input)| (to == force_id && input == 0).then_some(from))?;
            Some(
                mesh.vertices
                    .iter()
                    .map(|vertex| {
                        let strength = self
                            .sample_scalar_graph_at_uv(strength_id, vertex.uv, 0)
                            .unwrap_or(0.0);
                        CoreVec3::new(
                            evaluate_force_formula(
                                &formulas[0],
                                vertex.position[0],
                                vertex.position[1],
                                vertex.position[2],
                                self.dynamics_time,
                            )
                            .unwrap_or(0.0),
                            evaluate_force_formula(
                                &formulas[1],
                                vertex.position[0],
                                vertex.position[1],
                                vertex.position[2],
                                self.dynamics_time,
                            )
                            .unwrap_or(0.0),
                            evaluate_force_formula(
                                &formulas[2],
                                vertex.position[0],
                                vertex.position[1],
                                vertex.position[2],
                                self.dynamics_time,
                            )
                            .unwrap_or(0.0),
                        ) * (scale * strength)
                    })
                    .collect(),
            )
        });
        let forces = if simulator_id.is_some() && force_id.is_none() {
            Some(vec![CoreVec3::ZERO; mesh.vertices.len()])
        } else {
            forces
        };
        let (velocities, velocity_blend, velocity_mode) = velocity_id
            .and_then(|velocity_id| {
                let node = self
                    .compositor_nodes
                    .iter()
                    .find(|node| node.id == velocity_id)?;
                let NodeSettings::VelocityField {
                    ref formulas,
                    scale,
                    blend,
                    mode,
                    ..
                } = node.settings
                else {
                    return None;
                };
                let strength_mask = self
                    .compositor_links
                    .iter()
                    .find_map(|&(from, _, to, input)| {
                        (to == velocity_id && input == 0).then_some(from)
                    })
                    .and_then(|strength_id| {
                        self.compositor_nodes
                            .iter()
                            .find_map(|node| match node.settings {
                                NodeSettings::PaintedMask { object_index }
                                    if node.id == strength_id =>
                                {
                                    Some(object_index)
                                }
                                _ => None,
                            })
                    })
                    .and_then(|strength_object| self.object_node_id(strength_object))
                    .and_then(|object_id| self.painted_masks.get(&object_id));
                let values = mesh
                    .vertices
                    .iter()
                    .map(|vertex| {
                        let strength = strength_mask
                            .map(|mask| sample_painted_mask(mask, vertex.uv))
                            .unwrap_or(1.0);
                        CoreVec3::new(
                            evaluate_force_formula(
                                &formulas[0],
                                vertex.position[0],
                                vertex.position[1],
                                vertex.position[2],
                                self.dynamics_time,
                            )
                            .unwrap_or(0.0),
                            evaluate_force_formula(
                                &formulas[1],
                                vertex.position[0],
                                vertex.position[1],
                                vertex.position[2],
                                self.dynamics_time,
                            )
                            .unwrap_or(0.0),
                            evaluate_force_formula(
                                &formulas[2],
                                vertex.position[0],
                                vertex.position[1],
                                vertex.position[2],
                                self.dynamics_time,
                            )
                            .unwrap_or(0.0),
                        ) * (scale * strength)
                    })
                    .collect();
                Some((Some(values), blend.clamp(0.0, 1.0), mode.min(2)))
            })
            .unwrap_or((None, 1.0, 0));
        let (velocities, velocity_blend, velocity_mode) =
            if simulator_id.is_some() && velocity_id.is_none() {
                // A disconnected velocity socket is the additive identity. Using
                // Add rather than Set is important: a zero Set field would erase
                // motion produced by forces, gravity, and spring constraints.
                (Some(vec![CoreVec3::ZERO; mesh.vertices.len()]), 1.0, 1)
            } else {
                (velocities, velocity_blend, velocity_mode)
            };
        let (gravity, time_scale) = simulator_id
            .and_then(|simulator_id| {
                self.compositor_nodes
                    .iter()
                    .find_map(|node| match node.settings {
                        NodeSettings::Simulator {
                            gravity,
                            time_scale,
                            ..
                        } if node.id == simulator_id => Some((gravity, time_scale)),
                        _ => None,
                    })
            })
            .unwrap_or((true, 1.0));
        Some((
            masses,
            settings,
            forces,
            velocities,
            velocity_blend,
            velocity_mode,
            gravity,
            time_scale,
        ))
    }

    fn sample_scalar_graph_at_uv(&self, node_id: usize, uv: [f32; 2], depth: usize) -> Option<f32> {
        if depth > 64 {
            return None;
        }
        let node = self
            .compositor_nodes
            .iter()
            .find(|node| node.id == node_id)?;
        match &node.settings {
            NodeSettings::PaintedMask { object_index } => {
                let object_id = self.object_node_id(*object_index)?;
                self.painted_masks
                    .get(&object_id)
                    .map(|mask| sample_painted_mask(mask, uv))
            }
            NodeSettings::ConstantValue { value, .. } => Some(*value),
            NodeSettings::Remap { points, mode, .. } => {
                let source = self
                    .compositor_links
                    .iter()
                    .find_map(|&(from, _, to, input)| {
                        (to == node_id && input == 0).then_some(from)
                    })?;
                let value = self.sample_scalar_graph_at_uv(source, uv, depth + 1)?;
                Some(compositor_graph::remap_value(value, points, *mode == 1))
            }
            _ => None,
        }
    }

    fn reset_dynamics(&mut self) {
        self.dynamics_time = 0.0;
        self.dynamics_accumulator = 0.0;
        for (&simulator_id, state) in &self.object_simulation_states {
            let object_index = self
                .compositor_nodes
                .iter()
                .find_map(|node| match node.settings {
                    NodeSettings::ObjectSimulator { object_index, .. }
                        if node.id == simulator_id =>
                    {
                        Some(object_index)
                    }
                    _ => None,
                });
            if let Some(id) = object_index.and_then(|index| self.object_node_id(index))
                && let Ok(node) = self.scene.tree.node(id)
            {
                let mut transform = node.local_transform();
                transform.translation = state.initial_position;
                transform.rotation = Quat::from_euler_xyz(CoreVec3::new(
                    state.initial_rotation_degrees.x.to_radians(),
                    state.initial_rotation_degrees.y.to_radians(),
                    state.initial_rotation_degrees.z.to_radians(),
                ));
                let _ = self.scene.tree.set_local_transform(id, transform);
            }
        }
        self.object_simulation_states.clear();
        for (&id, state) in &self.joint_simulation_states {
            if let Ok(node) = self.scene.tree.node(id) {
                let mut transform = node.local_transform();
                transform.rotation = state.base_rotation;
                let _ = self.scene.tree.set_local_transform(id, transform);
            }
        }
        self.joint_simulation_states.clear();
        for cloth in self.dynamics_cloth.values_mut() {
            cloth.reset();
        }
        self.scene_revision = self.scene_revision.wrapping_add(1);
    }

    fn paint_selected_field(&mut self, pointer: Pos2, viewport: Rect) {
        let Some(id) = self.scene.selected else {
            return;
        };
        let Some(presented) = self.presented_view.as_ref() else {
            return;
        };
        let center = viewport.center();
        let projection_scale = viewport.width().min(viewport.height()) * 0.18 * self.camera_zoom;
        let mut affected = HashMap::<usize, f32>::new();
        for triangle in presented.triangles.iter() {
            if triangle.object_id != id {
                continue;
            }
            for vertex in &triangle.vertices {
                let Some(screen) = project(
                    vertex.position,
                    center,
                    projection_scale,
                    self.camera_yaw,
                    self.camera_pitch,
                    0.0,
                    self.camera_target,
                    self.projection_mode,
                    self.grid_spacing,
                ) else {
                    continue;
                };
                let distance = screen.distance(pointer);
                if distance <= self.paint_radius_pixels {
                    let normalized = distance / self.paint_radius_pixels.max(1.0);
                    let influence = 1.0 - normalized * normalized * (3.0 - 2.0 * normalized);
                    affected
                        .entry(vertex.source_index)
                        .and_modify(|current| *current = current.max(influence))
                        .or_insert(influence);
                }
            }
        }
        if affected.is_empty() {
            return;
        }
        let Some(field) = self.dynamics_fields.get_mut(&id) else {
            return;
        };
        field.paint(
            affected,
            self.paint_value,
            self.paint_strength,
            self.paint_mode,
        );
        if let Some(cloth) = self.dynamics_cloth.get_mut(&id) {
            cloth.sync_mobility(field);
        }
        self.scene_revision = self.scene_revision.wrapping_add(1);
        self.project_dirty = true;
    }

    fn ensure_painted_texture(&mut self, id: NodeId) {
        self.painted_masks
            .entry(id)
            .or_insert_with(|| PaintedMask::new(512));
        let Some(object_index) = self
            .scene
            .tree
            .iter()
            .enumerate()
            .find_map(|(index, (candidate, _))| (candidate == id).then_some(index))
        else {
            return;
        };
        if self.compositor_nodes.iter().any(|node| {
            node.object_index == object_index
                && matches!(node.settings, NodeSettings::PaintedMask { .. })
        }) {
            return;
        }
        let node_id = self.compositor_next_id;
        self.compositor_next_id = self.compositor_next_id.wrapping_add(1);
        self.compositor_nodes.push(CompositorNode {
            id: node_id,
            object_index,
            object_name: self
                .scene
                .tree
                .node(id)
                .map(|object| object.name.clone())
                .unwrap_or_default(),
            settings_object_name: self
                .scene
                .tree
                .node(id)
                .map(|object| object.name.clone())
                .ok(),
            settings: NodeSettings::PaintedMask { object_index },
            position: Vec2::new(80.0, 330.0),
        });
        self.project_dirty = true;
    }

    fn paint_selected_texture(&mut self, pointer: Pos2, viewport: Rect) {
        let Some(id) = self.scene.selected else {
            return;
        };
        let Some(presented) = self.presented_view.as_ref() else {
            return;
        };
        let center = viewport.center();
        let projection_scale = viewport.width().min(viewport.height()) * 0.18 * self.camera_zoom;
        let (right, up, forward) = camera_basis(self.camera_yaw, self.camera_pitch, 0.0);
        let camera_distance = PERSPECTIVE_CAMERA_DISTANCE * self.grid_spacing.max(1.0e-4);
        let camera_origin = self.camera_target - forward * camera_distance;
        let screen_x = (pointer.x - center.x) / projection_scale;
        let screen_y = -(pointer.y - center.y) / projection_scale;
        let (ray_origin, ray_direction) = match self.projection_mode {
            ProjectionMode::Perspective => (
                camera_origin,
                (forward
                    + right * (screen_x / camera_distance)
                    + up * (screen_y / camera_distance))
                    .normalized(),
            ),
            ProjectionMode::Orthographic => (
                self.camera_target + right * screen_x + up * screen_y - forward * camera_distance,
                forward,
            ),
        };
        let mut hit: Option<([f32; 2], f32)> = None;
        for triangle in presented
            .triangles
            .iter()
            .filter(|triangle| triangle.object_id == id)
        {
            let points = triangle.vertices.map(|vertex| {
                CoreVec3::new(vertex.position[0], vertex.position[1], vertex.position[2])
            });
            let Some((distance, weights)) = ray_triangle_hit(ray_origin, ray_direction, points)
            else {
                continue;
            };
            let uv = [
                (0..3)
                    .map(|index| triangle.vertices[index].uv[0] * weights[index])
                    .sum(),
                (0..3)
                    .map(|index| triangle.vertices[index].uv[1] * weights[index])
                    .sum(),
            ];
            if hit.map_or(true, |(_, best_distance)| distance < best_distance) {
                hit = Some((uv, distance));
            }
        }
        let Some((uv, _)) = hit else {
            return;
        };
        let Some(mask) = self.painted_masks.get_mut(&id) else {
            return;
        };
        let radius = (self.paint_radius_pixels / viewport.width().min(viewport.height()).max(1.0)
            * mask.width.max(mask.height) as f32
            * 2.0)
            .clamp(1.0, mask.width.max(mask.height) as f32 * 0.25);
        let previous = self.texture_paint_last_uv.replace(uv);
        let interpolation_steps = previous
            .filter(|previous| {
                (previous[0] - uv[0]).abs() <= 0.5 && (previous[1] - uv[1]).abs() <= 0.5
            })
            .map(|previous| {
                let delta = Vec2::new(
                    (uv[0] - previous[0]) * mask.width as f32,
                    (uv[1] - previous[1]) * mask.height as f32,
                );
                (delta.length() / (radius * 0.35).max(1.0)).ceil() as usize
            })
            .unwrap_or(1)
            .clamp(1, 128);
        let target = (self.paint_value.clamp(0.0, 1.0) * 255.0).round();
        let smooth_source = (self.paint_mode == PaintMode::Smooth).then(|| mask.pixels.clone());
        let mut changed = false;
        for step in 1..=interpolation_steps {
            let amount_along = step as f32 / interpolation_steps as f32;
            let sample_uv = previous
                .map(|previous| {
                    [
                        previous[0] + (uv[0] - previous[0]) * amount_along,
                        previous[1] + (uv[1] - previous[1]) * amount_along,
                    ]
                })
                .unwrap_or(uv);
            let center_x = sample_uv[0].rem_euclid(1.0) * mask.width as f32;
            let center_y = sample_uv[1].rem_euclid(1.0) * mask.height as f32;
            let minimum_x = (center_x - radius).floor().max(0.0) as u32;
            let maximum_x = (center_x + radius).ceil().min(mask.width as f32 - 1.0) as u32;
            let minimum_y = (center_y - radius).floor().max(0.0) as u32;
            let maximum_y = (center_y + radius).ceil().min(mask.height as f32 - 1.0) as u32;
            for y in minimum_y..=maximum_y {
                for x in minimum_x..=maximum_x {
                    let distance = Vec2::new(x as f32 - center_x, y as f32 - center_y).length();
                    if distance > radius {
                        continue;
                    }
                    let falloff = 1.0 - (distance / radius).powi(2);
                    let amount = (falloff * self.paint_strength).clamp(0.0, 1.0);
                    let index = (y * mask.width + x) as usize;
                    let current = mask.pixels[index] as f32;
                    let next = match self.paint_mode {
                        PaintMode::Replace => current + (target - current) * amount,
                        PaintMode::Add => current + target * amount,
                        PaintMode::Subtract => current - target * amount,
                        PaintMode::Erase => {
                            if amount > 0.01 {
                                0.0
                            } else {
                                current
                            }
                        }
                        PaintMode::Smooth => {
                            let source = smooth_source.as_ref().expect("smooth snapshot exists");
                            let mut sum = 0u32;
                            let mut samples = 0u32;
                            for sample_y in y.saturating_sub(1)..=(y + 1).min(mask.height - 1) {
                                for sample_x in x.saturating_sub(1)..=(x + 1).min(mask.width - 1) {
                                    sum +=
                                        source[(sample_y * mask.width + sample_x) as usize] as u32;
                                    samples += 1;
                                }
                            }
                            let average = sum as f32 / samples.max(1) as f32;
                            current + (average - current) * amount
                        }
                    }
                    .clamp(0.0, 255.0)
                    .round() as u8;
                    changed |= next != mask.pixels[index];
                    mask.pixels[index] = next;
                }
            }
        }
        if changed {
            mask.revision = mask.revision.wrapping_add(1);
            self.scene_revision = self.scene_revision.wrapping_add(1);
            self.texture_revision = self.texture_revision.wrapping_add(1);
            self.compositor_eval_cache.clear();
            self.compositor_gpu_cache.clear();
            self.project_dirty = true;
        }
    }

    fn compositing_workspace(&mut self, ctx: &egui::Context) {
        let Some(object_index) = self.selected_object_index() else {
            egui::CentralPanel::default()
                .frame(panel_frame(Color32::from_rgb(18, 20, 26)))
                .show(ctx, |ui| {
                    ui.centered_and_justified(|ui| {
                        ui.label("Select an object in Scene to edit its node graph.");
                    });
                });
            return;
        };
        let object_choices = self
            .scene
            .tree
            .iter()
            .map(|(id, node)| (id, node.name.clone()))
            .collect::<Vec<_>>();
        let visible_ids: BTreeSet<usize> = self
            .compositor_nodes
            .iter()
            .filter_map(|node| (node.object_index == object_index).then_some(node.id))
            .collect();
        if !visible_ids.contains(&self.compositor_selected_node) {
            self.compositor_selected_node = self
                .compositor_nodes
                .iter()
                .find(|node| {
                    node.object_index == object_index
                        && matches!(node.settings, NodeSettings::ObjectMesh { .. })
                })
                .or_else(|| {
                    self.compositor_nodes
                        .iter()
                        .find(|node| node.object_index == object_index)
                })
                .map(|node| node.id)
                .unwrap_or(0);
            self.compositor_pending_output = None;
            self.compositor_dragging_node = None;
        }
        let debug_nodes: BTreeSet<usize> = self
            .compositor_nodes
            .iter()
            .filter_map(|node| {
                (node.object_index == object_index && matches!(node.settings, NodeSettings::Debug))
                    .then_some(node.id)
            })
            .collect();
        self.compositor_debug_textures
            .retain(|node, _| debug_nodes.contains(node));
        let previous_links = self.compositor_links.clone();
        egui::CentralPanel::default()
            .frame(panel_frame(Color32::from_rgb(18, 20, 26)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let object_name = self
                        .scene
                        .tree
                        .iter()
                        .nth(object_index)
                        .map(|(_, node)| node.name.clone())
                        .unwrap_or_else(|| "Object".into());
                    ui.strong(format!("Nodes — {object_name}"));
                    egui::ComboBox::from_id_salt("node_workspace_object")
                        .selected_text(&object_name)
                        .show_ui(ui, |ui| {
                            for (id, name) in &object_choices {
                                ui.selectable_value(&mut self.scene.selected, Some(*id), name);
                            }
                        });
                    ui.separator();
                    ui.menu_button("Add", |ui| {
                        ui.menu_button("Input", |ui| {
                            for (index, label) in [(0, "Object Texture"), (1, "Image Asset"), (2, "Constant Value"), (14, "Object Handle"), (15, "Time"), (19, "Painted Texture")] {
                                if compositor_add_button(ui, true, label) {
                                    self.activate_compositor_node(index);
                                }
                            }
                        });
                        ui.menu_button("Transform", |ui| {
                            if compositor_add_button(ui, true, "XYZ Position") {
                                self.activate_compositor_node(25);
                            }
                            if compositor_add_button(ui, true, "XYZ Rotation") {
                                self.activate_compositor_node(26);
                            }
                        });
                        ui.menu_button("Color", |ui| {
                            for (index, label) in [(3, "Remap"), (10, "Color Space Convert"), (11, "Color Decoder"), (13, "Color Encoder"), (12, "Grayscale")] {
                                if compositor_add_button(ui, true, label) {
                                    self.activate_compositor_node(index);
                                }
                            }
                        });
                        ui.menu_button("Converter", |ui| {
                            for (index, label) in [(4, "Texture Math"), (30, "Algebra"), (9, "Texture Combine"), (5, "Sharp Threshold"), (6, "Smooth Threshold")] {
                                if compositor_add_button(ui, true, label) {
                                    self.activate_compositor_node(index);
                                }
                            }
                        });
                        ui.menu_button("Filter", |ui| {
                            if compositor_add_button(ui, true, "Image Filter") {
                                self.activate_compositor_node(7);
                            }
                        });
                        ui.menu_button("Dynamics", |ui| {
                            if compositor_add_button(ui, true, "Object Simulator") {
                                self.activate_compositor_node(27);
                            }
                            if compositor_add_button(ui, true, "Mass Density") {
                                self.activate_compositor_node(20);
                            }
                            if compositor_add_button(ui, true, "Spring Mesh") {
                                self.activate_compositor_node(21);
                            }
                            if compositor_add_button(ui, true, "Force Field") {
                                self.activate_compositor_node(22);
                            }
                            if compositor_add_button(ui, true, "Velocity Field") {
                                self.activate_compositor_node(23);
                            }
                            if compositor_add_button(ui, true, "Simulator") {
                                self.activate_compositor_node(24);
                            }
                        });
                        ui.menu_button("Utility", |ui| {
                            if compositor_add_button(ui, true, "Debug Preview") {
                                self.activate_compositor_node(16);
                            }
                        });
                        ui.menu_button("Output", |ui| {
                            if compositor_add_button(ui, true, "Object Texture") {
                                self.activate_compositor_node(8);
                            }
                            ui.add_enabled(false, egui::Button::new("Object Transform"));
                            ui.add_enabled(false, egui::Button::new("Object Mesh"));
                            ui.add_enabled(false, egui::Button::new("Force Output"));
                            ui.add_enabled(false, egui::Button::new("Engine Joint"));
                        });
                        ui.separator();
                        ui.add_enabled(
                            false,
                            egui::Button::new("Output targets selected object"),
                        );
                    });
                    ui.menu_button("Node", |ui| {
                        let sel = self.compositor_selected_node;
                        let can_remove = !self.compositor_nodes.iter().any(|n| n.id == sel && matches!(n.settings, NodeSettings::ObjectTransform { .. } | NodeSettings::ObjectMesh { .. } | NodeSettings::ForceOutput { .. } | NodeSettings::Engine { .. }));
                        if ui.add_enabled(can_remove, egui::Button::new("Remove from graph")).clicked() {
                            self.compositor_nodes.retain(|n| n.id != sel);
                            self.compositor_links.retain(|&(fid, _, tid, _)| fid != sel && tid != sel);
                            self.compositor_pending_output = None;
                            self.compositor_selected_node = self.compositor_nodes.iter().find(|n| n.object_index == object_index && matches!(n.settings, NodeSettings::ObjectMesh { .. })).map(|n| n.id).unwrap_or(0);
                            self.project_dirty = true;
                            ui.close_menu();
                        }
                        ui.label("Mute");
                    });
                    ui.menu_button("View", |ui| {
                        if ui.button("Frame All").clicked() {
                            self.compositor_pan = Vec2::ZERO;
                            self.compositor_zoom = 1.0;
                            ui.close_menu();
                        }
                    });
                    ui.label(
                        RichText::new(if self.render_device == RenderDevice::Vulkan {
                            "Device: Vulkan"
                        } else {
                            "Device: CPU"
                        })
                            .weak()
                            .small(),
                    );
                });
                ui.separator();

                let available = ui.available_size();
                let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
                if response.dragged_by(egui::PointerButton::Middle) {
                    self.compositor_pan += response.drag_delta();
                }
                if response.hovered() {
                    let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        self.compositor_zoom = (self.compositor_zoom * (scroll * 0.0015).exp()).clamp(0.35, 2.5);
                    }
                }

                let rect = response.rect;
                painter.rect_filled(rect, 0.0, Color32::from_rgb(16, 18, 23));
                let palette_width = 176.0;
                let properties_width = 220.0;
                let canvas = Rect::from_min_max(
                    Pos2::new(rect.left() + palette_width, rect.top()),
                    Pos2::new(rect.right() - properties_width, rect.bottom()),
                );
                let grid_spacing = 32.0 * self.compositor_zoom;
                if grid_spacing >= 12.0 {
                    let offset_x = self.compositor_pan.x.rem_euclid(grid_spacing);
                    let offset_y = self.compositor_pan.y.rem_euclid(grid_spacing);
                    let mut x = canvas.left() + offset_x;
                    while x < canvas.right() {
                        painter.line_segment([Pos2::new(x, canvas.top()), Pos2::new(x, canvas.bottom())], Stroke::new(1.0, Color32::from_rgb(31, 34, 42)));
                        x += grid_spacing;
                    }
                    let mut y = canvas.top() + offset_y;
                    while y < canvas.bottom() {
                        painter.line_segment([Pos2::new(canvas.left(), y), Pos2::new(canvas.right(), y)], Stroke::new(1.0, Color32::from_rgb(31, 34, 42)));
                        y += grid_spacing;
                    }
                }

                let origin = canvas.min + Vec2::new(70.0, 100.0) + self.compositor_pan;
                let scale = self.compositor_zoom;
                let node_specs_by_kind: [(&str, &str, Color32); 31] = [
                    ("Object Texture", "Input", Color32::from_rgb(76, 122, 155)),
                    ("Image Asset", "Input", Color32::from_rgb(76, 122, 155)),
                    ("Constant Value", "Input", Color32::from_rgb(76, 122, 155)),
                    ("Remap", "Color", Color32::from_rgb(122, 88, 151)),
                    ("Texture Math", "Converter", Color32::from_rgb(105, 112, 122)),
                    ("Sharp Threshold", "Converter", Color32::from_rgb(105, 112, 122)),
                    ("Smooth Threshold", "Converter", Color32::from_rgb(105, 112, 122)),
                    ("Image Filter", "Filter", Color32::from_rgb(92, 128, 92)),
                    (
                        "Object Texture",
                        "Texture Output",
                        Color32::from_rgb(128, 113, 72),
                    ),
                    ("Texture Combine", "Converter", Color32::from_rgb(105, 112, 122)),
                    ("Color Space Convert", "Color", Color32::from_rgb(122, 88, 151)),
                    ("Color Decoder", "Color", Color32::from_rgb(122, 88, 151)),
                    ("Grayscale", "Color", Color32::from_rgb(122, 88, 151)),
                    ("Color Encoder", "Color", Color32::from_rgb(122, 88, 151)),
                    ("Object Handle", "Input", Color32::from_rgb(76, 122, 155)),
                    ("Time", "Input", Color32::from_rgb(76, 122, 155)),
                    ("Debug Preview", "Utility", Color32::from_rgb(173, 91, 117)),
                    ("Object Transform", "Object Output", Color32::from_rgb(128, 113, 72)),
                    ("Object Mesh", "Object Output", Color32::from_rgb(128, 113, 72)),
                    ("Painted Texture", "Texture Input", Color32::from_rgb(76, 122, 155)),
                    ("Mass Density", "Particle Field", Color32::from_rgb(76, 142, 153)),
                    ("Spring Mesh", "Dynamics", Color32::from_rgb(92, 128, 92)),
                    ("Force Field", "Dynamics", Color32::from_rgb(151, 95, 76)),
                    ("Velocity Field", "Dynamics", Color32::from_rgb(70, 145, 135)),
                    ("Simulator", "Simulation", Color32::from_rgb(166, 124, 57)),
                    ("XYZ Position", "Transform", Color32::from_rgb(91, 118, 166)),
                    ("XYZ Rotation", "Transform", Color32::from_rgb(91, 118, 166)),
                    (
                        "Object Simulator",
                        "Rigid Dynamics",
                        Color32::from_rgb(196, 139, 48),
                    ),
                    (
                        "Force Output",
                        "Object Output",
                        Color32::from_rgb(151, 95, 76),
                    ),
                    (
                        "Engine Joint",
                        "Object Output",
                        Color32::from_rgb(190, 93, 47),
                    ),
                    ("Algebra", "Converter", Color32::from_rgb(105, 112, 122)),
                ];
                let node_heights_by_kind: [f32; 31] = [
                    205.0, 165.0, 215.0, 390.0, 175.0, 150.0, 185.0, 175.0, 220.0, 220.0,
                    175.0, 140.0, 140.0, 165.0, 285.0, 205.0, 270.0, 190.0, 165.0, 150.0,
                    285.0, 235.0, 245.0, 285.0, 190.0,
                    205.0, 205.0, 360.0, 210.0, 190.0, 190.0,
                ];
                let node_width = 230.0;

                // Handle pending spawn: center the new node
                if let Some(id) = self.compositor_pending_spawn.take() {
                    if let Some(node) = self.compositor_nodes.iter_mut().find(|n| n.id == id) {
                        let kind = node.settings.kind();
                        let combine_mode = match &node.settings {
                            NodeSettings::TextureCombine { mode, .. } => *mode,
                            _ => 0,
                        };
                        let height = compositor_node_height(
                            node_heights_by_kind[kind],
                            kind,
                            combine_mode,
                        );
                        node.position = compositor_centered_position(canvas, origin, scale, Vec2::new(node_width, height));
                    }
                }

                // Build node rects indexed by node_id
                let node_id_rects: Vec<(usize, Rect)> = self
                    .compositor_nodes
                    .iter()
                    .filter(|node| node.object_index == object_index)
                    .map(|node| {
                        let kind = node.settings.kind();
                        let combine_mode = match &node.settings {
                            NodeSettings::TextureCombine { mode, .. } => *mode,
                            _ => 0,
                        };
                        let height = compositor_node_height(
                            node_heights_by_kind[kind],
                            kind,
                            combine_mode,
                        );
                        let rect = Rect::from_min_size(
                            origin + node.position * scale,
                            Vec2::new(node_width, height) * scale,
                        );
                        (node.id, rect)
                    })
                    .collect();
                let rect_by_id: std::collections::HashMap<usize, Rect> = node_id_rects.iter().cloned().collect();

                let output_socket = |node_rect: Rect, kind: usize, out_idx: usize| -> Pos2 {
                    let base_y = compositor_output_socket_y(kind, out_idx);
                    Pos2::new(node_rect.right(), node_rect.top() + base_y * scale)
                };
                let input_socket = |node_rect: Rect, kind: usize, input: usize| -> Pos2 {
                    let y = compositor_input_socket_y(kind, input);
                    Pos2::new(node_rect.left(), node_rect.top() + y * scale)
                };

                let (pointer, primary_pressed, primary_down, primary_released) = ui.input(|input| (
                    input.pointer.interact_pos(),
                    input.pointer.button_pressed(egui::PointerButton::Primary),
                    input.pointer.button_down(egui::PointerButton::Primary),
                    input.pointer.button_released(egui::PointerButton::Primary),
                ));

                if primary_pressed && let Some(ptr) = pointer && canvas.contains(ptr) {
                    let socket_radius = 12.0 * scale.max(0.7);
                    let mut handled = false;
                    // Check output sockets
                    'outer: for &(node_id, node_rect) in &node_id_rects {
                        let kind = self.compositor_nodes.iter().find(|n| n.id == node_id).map(|n| n.settings.kind()).unwrap_or(0);
                        for out_idx in 0..compositor_output_count(kind) {
                            if ptr.distance(output_socket(node_rect, kind, out_idx)) <= socket_radius {
                                self.compositor_pending_output = if self.compositor_pending_output == Some((node_id, out_idx)) {
                                    None
                                } else {
                                    Some((node_id, out_idx))
                                };
                                self.compositor_selected_node = node_id;
                                handled = true;
                                break 'outer;
                            }
                        }
                        // Check input sockets
                        let cm = if kind == 9 {
                            self.compositor_nodes.iter().find(|n| n.id == node_id)
                                .and_then(|n| if let NodeSettings::TextureCombine { mode, .. } = n.settings { Some(mode) } else { None })
                                .unwrap_or(0)
                        } else { 0 };
                        for input in 0..compositor_input_count(kind, cm) {
                            if ptr.distance(input_socket(node_rect, kind, input)) <= socket_radius {
                                if let Some((from_id, from_out)) = self.compositor_pending_output.take() {
                                    if from_id != node_id {
                                        // Toggle link: if exact link exists, remove it; otherwise replace
                                        if let Some(link) = self.compositor_links.iter().position(|&(fid, fo, tid, ti)| fid == from_id && fo == from_out && tid == node_id && ti == input) {
                                            self.compositor_links.remove(link);
                                        } else {
                                            self.compositor_links.retain(|&(_, _, to, ti)| to != node_id || ti != input);
                                            self.compositor_links.push((from_id, from_out, node_id, input));
                                        }
                                        self.project_dirty = true;
                                    }
                                } else {
                                    let old_len = self.compositor_links.len();
                                    self.compositor_links.retain(|&(_, _, to, ti)| to != node_id || ti != input);
                                    self.project_dirty |= old_len != self.compositor_links.len();
                                }
                                self.compositor_selected_node = node_id;
                                handled = true;
                                break 'outer;
                            }
                        }
                    }
                    // Check close buttons (non-Output nodes)
                    if !handled {
                        for &(node_id, node_rect) in &node_id_rects {
                            let kind = self.compositor_nodes.iter().find(|n| n.id == node_id).map(|n| n.settings.kind()).unwrap_or(0);
                            if matches!(kind, 17 | 18 | 28) { continue; }
                            let close_pos = Pos2::new(node_rect.right() - 13.0 * scale, node_rect.top() + 15.0 * scale);
                            if ptr.distance(close_pos) <= 10.0 * scale {
                                let id_to_remove = node_id;
                                self.compositor_nodes.retain(|n| n.id != id_to_remove);
                                self.compositor_links.retain(|&(fid, _, tid, _)| fid != id_to_remove && tid != id_to_remove);
                                self.compositor_pending_output = None;
                                self.compositor_selected_node = self.compositor_nodes.iter().find(|n| n.object_index == object_index && matches!(n.settings, NodeSettings::ObjectMesh { .. })).map(|n| n.id).unwrap_or(0);
                                self.project_dirty = true;
                                handled = true;
                                break;
                            }
                        }
                    }
                    // Check header drag
                    if !handled {
                        for &(node_id, node_rect) in node_id_rects.iter().rev() {
                            let header_rect = Rect::from_min_size(node_rect.min, Vec2::new(node_rect.width(), 30.0 * scale));
                            if header_rect.contains(ptr) {
                                self.compositor_selected_node = node_id;
                                self.compositor_dragging_node = Some((node_id, ptr - node_rect.min));
                                self.compositor_pending_output = None;
                                handled = true;
                                break;
                            }
                        }
                    }
                    if !handled {
                        self.compositor_pending_output = None;
                    }
                }

                if primary_down && let (Some(ptr), Some((drag_id, grab_offset))) = (pointer, self.compositor_dragging_node) {
                    if let Some(node) = self.compositor_nodes.iter_mut().find(|n| n.id == drag_id) {
                        node.position = (ptr - grab_offset - origin) / scale;
                    }
                    ctx.request_repaint();
                }
                if primary_released && self.compositor_dragging_node.take().is_some() {
                    self.project_dirty = true;
                }

                if ui.input_mut(|i| i.consume_key(egui::Modifiers::NONE, Key::Delete)) {
                    let sel = self.compositor_selected_node;
                    if let Some(node) = self.compositor_nodes.iter().find(|n| n.id == sel) {
                        if !matches!(node.settings, NodeSettings::ObjectTransform { .. } | NodeSettings::ObjectMesh { .. } | NodeSettings::ForceOutput { .. }) {
                            self.compositor_nodes.retain(|n| n.id != sel);
                            self.compositor_links.retain(|&(fid, _, tid, _)| fid != sel && tid != sel);
                            self.compositor_pending_output = None;
                            self.compositor_selected_node = self.compositor_nodes.iter().find(|n| n.object_index == object_index && matches!(n.settings, NodeSettings::ObjectMesh { .. })).map(|n| n.id).unwrap_or(0);
                            self.project_dirty = true;
                        }
                    }
                }

                // Draw links
                for &(from_id, from_out, to_id, to_input) in &self.compositor_links {
                    if let (Some(&from_rect), Some(&to_rect)) = (rect_by_id.get(&from_id), rect_by_id.get(&to_id)) {
                        let from_kind = self.compositor_nodes.iter().find(|n| n.id == from_id).map(|n| n.settings.kind()).unwrap_or(0);
                        let to_kind = self.compositor_nodes.iter().find(|n| n.id == to_id).map(|n| n.settings.kind()).unwrap_or(0);
                        let to_cm = if to_kind == 9 {
                            self.compositor_nodes.iter().find(|n| n.id == to_id)
                                .and_then(|n| if let NodeSettings::TextureCombine { mode, .. } = n.settings { Some(mode) } else { None })
                                .unwrap_or(0)
                        } else { 0 };
                        let _ = to_cm;
                        let start = output_socket(from_rect, from_kind, from_out);
                        let end = input_socket(to_rect, to_kind, to_input);
                        painter.add(egui::Shape::line(compositor_link_curve(start, end), Stroke::new(3.0, Color32::from_rgb(218, 190, 92))));
                    }
                }
                if let (Some((from_id, from_out)), Some(ptr)) = (self.compositor_pending_output, pointer) {
                    if let Some(&from_rect) = rect_by_id.get(&from_id) {
                        let from_kind = self.compositor_nodes.iter().find(|n| n.id == from_id).map(|n| n.settings.kind()).unwrap_or(0);
                        painter.add(egui::Shape::line(
                            compositor_link_curve(output_socket(from_rect, from_kind, from_out), ptr),
                            Stroke::new(2.0, Color32::from_rgb(108, 190, 255)),
                        ));
                    }
                }

                // Draw nodes
                for &(node_id, node_rect) in &node_id_rects {
                    let node = match self.compositor_nodes.iter().find(|n| n.id == node_id) {
                        Some(n) => n,
                        None => continue,
                    };
                    let kind = node.settings.kind();
                    let (title, _, header_color) = node_specs_by_kind[kind];
                    let is_output = matches!(kind, 17 | 18 | 28 | 29);
                    let selected = self.compositor_selected_node == node_id;
                    let cm = if let NodeSettings::TextureCombine { mode, .. } = node.settings { mode } else { 0 };

                    painter.rect_filled(node_rect, 6.0, if selected { Color32::from_rgb(47, 50, 61) } else { Color32::from_rgb(36, 39, 48) });
                    let header_rect = Rect::from_min_size(node_rect.min, Vec2::new(node_rect.width(), 30.0 * scale));
                    painter.rect_filled(header_rect, 6.0, header_color);
                    let border = if selected { Color32::from_rgb(108, 190, 255) } else { Color32::from_rgb(67, 71, 83) };
                    for (a, b) in [
                        (node_rect.left_top(), node_rect.right_top()),
                        (node_rect.right_top(), node_rect.right_bottom()),
                        (node_rect.right_bottom(), node_rect.left_bottom()),
                        (node_rect.left_bottom(), node_rect.left_top()),
                    ] {
                        painter.line_segment([a, b], Stroke::new(2.0, border));
                    }
                    painter.text(header_rect.left_center() + Vec2::new(9.0 * scale, 0.0), Align2::LEFT_CENTER, title, FontId::proportional(13.0 * scale), Color32::WHITE);
                    ui.interact(
                        header_rect,
                        Id::new(("compositor_node_help", node_id)),
                        Sense::hover(),
                    )
                    .on_hover_text(compositor_node_description(kind));
                    if !is_output {
                        painter.text(Pos2::new(node_rect.right() - 13.0 * scale, node_rect.top() + 15.0 * scale), Align2::CENTER_CENTER, "×", FontId::proportional(16.0 * scale), Color32::from_rgba_unmultiplied(255, 255, 255, 160));
                    }
                    if matches!(kind, 25 | 26 | 28) {
                        painter.text(
                            node_rect.left_top() + Vec2::new(10.0, 45.0) * scale,
                            Align2::LEFT_TOP,
                            match kind {
                                25 => "Position",
                                26 => "Rotation",
                                _ => "Force",
                            },
                            FontId::proportional(11.0 * scale),
                            Color32::from_gray(180),
                        );
                    }
                    // Draw input sockets
                    for input in 0..compositor_input_count(kind, cm) {
                        let pos = input_socket(node_rect, kind, input);
                        painter.circle_filled(pos, 6.0 * scale, Color32::from_rgb(218, 190, 92));
                        painter.text(pos + Vec2::new(10.0 * scale, 0.0), Align2::LEFT_CENTER, compositor_input_label(kind, input), FontId::proportional(10.0 * scale), Color32::from_gray(180));
                    }

                    // Draw output sockets
                    for out_idx in 0..compositor_output_count(kind) {
                        let pos = output_socket(node_rect, kind, out_idx);
                        painter.circle_filled(
                            pos,
                            6.0 * scale,
                            Color32::from_rgb(218, 190, 92),
                        );
                        painter.text(pos - Vec2::new(10.0 * scale, 0.0), Align2::RIGHT_CENTER, compositor_output_label(kind, out_idx), FontId::proportional(10.0 * scale), Color32::from_gray(180));
                    }

                    // Draw node controls UI
                    let layer_id = compositor_control_layer(ui.layer_id(), node_id);
                    ui.ctx().set_sublayer(ui.layer_id(), layer_id);
                    let controls_top = compositor_controls_top(kind, cm);
                    let controls_origin = node_rect.min + Vec2::new(10.0, controls_top) * scale;
                    let transform = egui::emath::TSTransform::from_translation(controls_origin.to_vec2()) * egui::emath::TSTransform::from_scaling(scale);
                    ui.ctx().set_transform_layer(layer_id, transform);
                    let local_rect = Rect::from_min_size(
                        Pos2::ZERO,
                        Vec2::new(
                            node_width - 20.0,
                            node_rect.height() / scale - controls_top - 10.0,
                        ),
                    );
                    let mut controls_ui = ui.new_child(egui::UiBuilder::new().layer_id(layer_id).max_rect(local_rect).layout(Layout::top_down(Align::Min)));
                    controls_ui.set_clip_rect(local_rect);
                    controls_ui.style_mut().spacing.item_spacing.y = 3.0;
                    self.compositor_node_controls_ui(node_id, &mut controls_ui);
                }

                // Palette sidebar
                let palette = Rect::from_min_max(rect.min, Pos2::new(canvas.left(), rect.bottom()));
                painter.rect_filled(palette, 0.0, Color32::from_rgb(25, 27, 34));
                painter.text(palette.left_top() + Vec2::new(12.0, 14.0), Align2::LEFT_TOP, "NODES", FontId::proportional(12.0), Color32::from_rgb(108, 190, 255));
                for (index, label) in [
                    "Input / Object Texture",
                    "Input / Image Asset",
                    "Input / Constant Value",
                    "Input / Object Handle",
                    "Input / Time",
                    "Color / Remap",
                    "Color / Color Space Convert",
                    "Color / Color Decoder",
                    "Color / Color Encoder",
                    "Color / Grayscale",
                    "Converter / Texture Math",
                    "Converter / Texture Combine",
                    "Converter / Sharp Threshold",
                    "Converter / Smooth Threshold",
                    "Filter / Image Filter",
                    "Utility / Debug Preview",
                    "Output / Object Texture",
                    "Output / Object Transform",
                    "Output / Object Mesh",
                    "Input / Painted Texture",
                    "Dynamics / Mass Density",
                    "Dynamics / Spring Mesh",
                    "Dynamics / Force Field",
                    "Dynamics / Velocity Field",
                    "Dynamics / Simulator",
                ].iter().enumerate() {
                    painter.text(palette.left_top() + Vec2::new(12.0, 46.0 + index as f32 * 27.0), Align2::LEFT_TOP, *label, FontId::proportional(12.0), Color32::from_gray(190));
                }

                // Properties sidebar
                let properties = Rect::from_min_max(Pos2::new(canvas.right(), rect.top()), rect.max);
                painter.rect_filled(properties, 0.0, Color32::from_rgb(25, 27, 34));
                let mut info_ui = ui.new_child(egui::UiBuilder::new().max_rect(properties.shrink(12.0)).layout(Layout::top_down(Align::Min)));
                info_ui.heading("Selection");
                let sel_id = self.compositor_selected_node;
                if let Some(node) = self.compositor_nodes.iter().find(|n| n.id == sel_id) {
                    let kind = node.settings.kind();
                    info_ui.label(node_specs_by_kind[kind].0);
                    info_ui.small(format!("Position: {:.0}, {:.0}", node.position.x, node.position.y));
                }
                info_ui.separator();
                info_ui.small("Edit settings directly inside the node. Drag its colored header to move it; click sockets to connect or disconnect.");
            });
        if self.compositor_links != previous_links {
            self.compositor_eval_cache.clear();
            self.compositor_gpu_cache.clear();
        }
    }

    fn dynamics_workspace(&mut self, ctx: &egui::Context) {
        let previous_wind = self.dynamics_wind.clone();
        let previous_settings = self.dynamics_settings.clone();
        let object_choices = self
            .scene
            .tree
            .iter()
            .map(|(id, node)| (id, node.name.clone()))
            .collect::<Vec<_>>();
        egui::CentralPanel::default()
            .frame(panel_frame(Color32::from_rgb(18, 20, 26)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    let object_name = self
                        .scene
                        .selected
                        .and_then(|id| self.scene.tree.node(id).ok())
                        .map(|node| node.name.clone())
                        .unwrap_or_else(|| "No object selected".into());
                    ui.strong(format!("Nodes — {object_name}"));
                    egui::ComboBox::from_id_salt("node_workspace_object")
                        .selected_text(&object_name)
                        .show_ui(ui, |ui| {
                            for (id, name) in &object_choices {
                                ui.selectable_value(&mut self.scene.selected, Some(*id), name);
                            }
                        });
                    ui.separator();
                    if ui
                        .button(if self.dynamics_running {
                            "Ⅱ Pause"
                        } else {
                            "▶ Simulate"
                        })
                        .clicked()
                    {
                        self.dynamics_running = !self.dynamics_running;
                        self.dynamics_last_tick = Instant::now();
                    }
                    if ui.button("Step").clicked() {
                        self.dynamics_time += 1.0 / 60.0;
                        for id in self.dynamics_enabled.clone() {
                            if let (Some(field), Some(cloth)) = (
                                self.dynamics_fields.get(&id),
                                self.dynamics_cloth.get_mut(&id),
                            ) {
                                cloth.settings = self.dynamics_settings.clone();
                                cloth.sync_mobility(field);
                                cloth.step(1.0 / 60.0, self.dynamics_time, &self.dynamics_wind);
                            }
                        }
                        self.scene_revision = self.scene_revision.wrapping_add(1);
                    }
                    if ui.button("Reset").clicked() {
                        self.reset_dynamics();
                    }
                    ui.separator();
                    ui.label(format!(
                        "{:.2} s · {} simulated object(s)",
                        self.dynamics_time,
                        self.dynamics_enabled.len()
                    ));
                });
                ui.separator();
                let available = ui.available_size();
                let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
                if response.dragged_by(egui::PointerButton::Middle) {
                    self.dynamics_pan += response.drag_delta();
                }
                if response.hovered() {
                    let scroll = ui.input(|input| input.smooth_scroll_delta.y);
                    if scroll != 0.0 {
                        self.dynamics_zoom =
                            (self.dynamics_zoom * (scroll * 0.0015).exp()).clamp(0.4, 2.2);
                    }
                }
                painter.rect_filled(response.rect, 0.0, Color32::from_rgb(16, 18, 23));
                let origin = response.rect.min + Vec2::new(50.0, 45.0) + self.dynamics_pan;
                let scale = self.dynamics_zoom;
                let node_size = Vec2::new(210.0, 150.0) * scale;
                let titles = [
                    (
                        "Object Geometry",
                        "Geometry",
                        Color32::from_rgb(76, 122, 155),
                    ),
                    (
                        "Mobility Field",
                        "Vertex Scalar",
                        Color32::from_rgb(122, 88, 151),
                    ),
                    (
                        "Wind Field",
                        "Spatial Vector",
                        Color32::from_rgb(76, 142, 153),
                    ),
                    ("Collision Set", "Geometry", Color32::from_rgb(151, 95, 76)),
                    (
                        "XPBD Cloth Solver",
                        "Simulation State",
                        Color32::from_rgb(92, 128, 92),
                    ),
                    (
                        "Geometry Output",
                        "Deformed Mesh",
                        Color32::from_rgb(128, 113, 72),
                    ),
                ];
                let rects: Vec<Rect> = self
                    .dynamics_node_positions
                    .iter()
                    .map(|position| Rect::from_min_size(origin + *position * scale, node_size))
                    .collect();
                let pointer = ui.input(|input| input.pointer.interact_pos());
                let pressed =
                    ui.input(|input| input.pointer.button_pressed(egui::PointerButton::Primary));
                if pressed {
                    if let Some(pointer) = pointer {
                        for (index, rect) in rects.iter().enumerate().rev() {
                            let header = Rect::from_min_size(
                                rect.min,
                                Vec2::new(rect.width(), 30.0 * scale),
                            );
                            if header.contains(pointer) {
                                self.dynamics_dragging_node = Some((index, pointer - rect.min));
                                break;
                            }
                        }
                    }
                }
                let primary_down =
                    ui.input(|input| input.pointer.button_down(egui::PointerButton::Primary));
                if primary_down {
                    if let (Some(pointer), Some((index, offset))) =
                        (pointer, self.dynamics_dragging_node)
                    {
                        self.dynamics_node_positions[index] = (pointer - offset - origin) / scale;
                        ctx.request_repaint();
                    }
                } else {
                    self.dynamics_dragging_node = None;
                }
                for (from, to) in [(0, 4), (1, 4), (2, 4), (3, 4), (4, 5)] {
                    let start = rects[from].right_center();
                    let end = rects[to].left_center();
                    painter.add(egui::Shape::line(
                        compositor_link_curve(start, end),
                        Stroke::new(3.0, Color32::from_rgb(218, 190, 92)),
                    ));
                }
                for (index, rect) in rects.iter().copied().enumerate() {
                    let (title, kind, color) = titles[index];
                    painter.rect_filled(rect, 6.0, Color32::from_rgb(36, 39, 48));
                    let header =
                        Rect::from_min_size(rect.min, Vec2::new(rect.width(), 30.0 * scale));
                    painter.rect_filled(header, 6.0, color);
                    painter.text(
                        header.left_center() + Vec2::new(9.0 * scale, 0.0),
                        Align2::LEFT_CENTER,
                        title,
                        FontId::proportional(13.0 * scale),
                        Color32::WHITE,
                    );
                    painter.text(
                        rect.left_top() + Vec2::new(12.0, 48.0) * scale,
                        Align2::LEFT_TOP,
                        kind,
                        FontId::proportional(11.0 * scale),
                        Color32::from_gray(174),
                    );
                    if index != 0 && index != 1 && index != 2 && index != 3 {
                        painter.circle_filled(
                            rect.left_center(),
                            6.0 * scale,
                            Color32::from_rgb(218, 190, 92),
                        );
                    }
                    if index != 5 {
                        painter.circle_filled(
                            rect.right_center(),
                            6.0 * scale,
                            Color32::from_rgb(218, 190, 92),
                        );
                    }
                    let detail = match index {
                        0 => self
                            .scene
                            .selected
                            .and_then(|id| self.scene.tree.node(id).ok())
                            .map(|node| node.name.as_str())
                            .unwrap_or("Select a mesh in Scene"),
                        1 => "dynamics.mobility",
                        2 => "World-space air velocity",
                        3 => "Pole capsule + ground",
                        4 => "Stretch · shear · bend",
                        _ => "Viewport deformation",
                    };
                    painter.text(
                        rect.left_top() + Vec2::new(12.0, 76.0) * scale,
                        Align2::LEFT_TOP,
                        detail,
                        FontId::proportional(12.0 * scale),
                        Color32::from_gray(215),
                    );
                }

                let controls = Rect::from_min_size(
                    response.rect.right_top() - Vec2::new(280.0, -12.0),
                    Vec2::new(265.0, 390.0),
                );
                painter.rect_filled(controls, 6.0, Color32::from_rgb(25, 27, 34));
                let mut controls_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(controls.shrink(12.0))
                        .layout(Layout::top_down(Align::Min)),
                );
                controls_ui.strong("Field & solver properties");
                controls_ui.checkbox(&mut self.dynamics_wind.enabled, "Wind enabled");
                controls_ui.add(
                    egui::DragValue::new(&mut self.dynamics_wind.velocity.x)
                        .speed(0.1)
                        .prefix("Wind X "),
                );
                controls_ui.add(
                    egui::DragValue::new(&mut self.dynamics_wind.velocity.y)
                        .speed(0.1)
                        .prefix("Wind Y "),
                );
                controls_ui.add(
                    egui::DragValue::new(&mut self.dynamics_wind.velocity.z)
                        .speed(0.1)
                        .prefix("Wind Z "),
                );
                controls_ui.add(
                    egui::Slider::new(&mut self.dynamics_wind.gust_strength, 0.0..=2.0)
                        .text("Gust"),
                );
                controls_ui.add(
                    egui::Slider::new(&mut self.dynamics_wind.turbulence, 0.0..=2.0)
                        .text("Turbulence"),
                );
                controls_ui.separator();
                controls_ui.add(
                    egui::Slider::new(&mut self.dynamics_settings.iterations, 1..=16)
                        .text("Iterations"),
                );
                controls_ui.add(
                    egui::DragValue::new(&mut self.dynamics_settings.stretch_compliance)
                        .speed(0.000_001)
                        .prefix("Stretch compliance "),
                );
                controls_ui.add(
                    egui::DragValue::new(&mut self.dynamics_settings.bend_compliance)
                        .speed(0.000_01)
                        .prefix("Bend compliance "),
                );
                controls_ui.add(
                    egui::Slider::new(&mut self.dynamics_settings.damping, 0.0..=0.2)
                        .text("Damping"),
                );
                controls_ui.small(
                    "Edit mobility in Scene → Field Paint. Wind and solver changes apply live.",
                );
            });
        if self.dynamics_wind != previous_wind || self.dynamics_settings != previous_settings {
            self.project_dirty = true;
        }
    }

    fn apply_editor_camera_input(&mut self, input: InputSample) {
        self.camera_yaw += input.orbit.x * 0.002;
        self.camera_pitch = (self.camera_pitch + input.orbit.y * 0.002).clamp(-1.2, 1.2);
        self.camera_zoom = (self.camera_zoom * input.zoom_log.exp()).clamp(0.001, 100.0);
        if input.pan != Vec2::ZERO {
            let pixels_per_unit = input.viewport_extent * 0.18 * self.camera_zoom;
            self.camera_target = pan_camera_target(
                self.camera_target,
                input.pan,
                pixels_per_unit,
                self.camera_yaw,
                self.camera_pitch,
            );
        }
    }

    fn viewport(&mut self, ctx: &egui::Context) {
        if self.active_tool == Tool::TexturePaint
            && let Some(id) = self.scene.selected
        {
            self.ensure_painted_texture(id);
        }
        egui::CentralPanel::default()
            .frame(panel_frame(Color32::from_rgb(19, 21, 27)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    egui::ComboBox::from_id_salt("projection_mode")
                        .selected_text(
                            RichText::new(self.projection_mode.label().to_uppercase())
                                .small()
                                .strong(),
                        )
                        .show_ui(ui, |ui| {
                            ui.selectable_value(
                                &mut self.projection_mode,
                                ProjectionMode::Perspective,
                                "Perspective",
                            );
                            ui.selectable_value(
                                &mut self.projection_mode,
                                ProjectionMode::Orthographic,
                                "Orthographic",
                            );
                        });
                    ui.separator();
                    ui.label(
                        RichText::new("EDITOR PREVIEW")
                            .small()
                            .color(Color32::from_rgb(104, 190, 255)),
                    );
                    ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
                        egui::ComboBox::from_id_salt("viewport_mode")
                            .selected_text(self.viewport_mode.label())
                            .show_ui(ui, |ui| {
                                ui.selectable_value(
                                    &mut self.viewport_mode,
                                    ViewportMode::Shaded,
                                    "Matte",
                                );
                                ui.selectable_value(
                                    &mut self.viewport_mode,
                                    ViewportMode::ShadedEdges,
                                    "Matte + Edges",
                                );
                                ui.selectable_value(
                                    &mut self.viewport_mode,
                                    ViewportMode::Wireframe,
                                    "Wireframe",
                                );
                            });
                        if ui
                            .selectable_label(self.camera_preview_visible, "📷")
                            .on_hover_text("Realtime camera preview on/off")
                            .clicked()
                        {
                            self.camera_preview_visible = !self.camera_preview_visible;
                        }
                        if ui
                            .selectable_label(self.global_light_enabled, "☀")
                            .on_hover_text("Global light on/off")
                            .clicked()
                        {
                            self.global_light_enabled = !self.global_light_enabled;
                            self.scene_revision = self.scene_revision.wrapping_add(1);
                            self.project_dirty = true;
                        }
                    });
                });
                let available = ui.available_size();
                let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
                let scene_camera = self
                    .camera_preview_visible
                    .then(|| self.active_scene_camera())
                    .flatten();
                let render_rect = scene_camera.as_ref().map_or(response.rect, |camera| {
                    fit_aspect_rect(response.rect, camera.aspect_ratio)
                });
                let render_camera = scene_camera.as_ref().map_or(
                    (
                        self.camera_yaw,
                        self.camera_pitch,
                        0.0,
                        self.camera_zoom,
                        self.camera_target,
                        self.grid_spacing,
                        self.projection_mode,
                    ),
                    |camera| {
                        scene_camera_render_view(camera, render_rect.size(), self.grid_spacing)
                    },
                );
                self.viewport_focused = response.hovered();
                let pointer_delta = ui.input(|input| input.pointer.delta());
                if matches!(self.active_tool, Tool::FieldPaint | Tool::TexturePaint) {
                    let (pointer, pressed, down, released) = ui.input(|input| {
                        (
                            input.pointer.hover_pos(),
                            input.pointer.button_pressed(egui::PointerButton::Primary),
                            input.pointer.button_down(egui::PointerButton::Primary),
                            input.pointer.button_released(egui::PointerButton::Primary),
                        )
                    });
                    if let Some(pointer) = pointer.filter(|point| response.rect.contains(*point)) {
                        painter.circle_stroke(
                            pointer,
                            self.paint_radius_pixels,
                            Stroke::new(2.0, Color32::from_rgb(255, 220, 96)),
                        );
                        if pressed {
                            if let Some(id) = self.scene.selected {
                                match self.active_tool {
                                    Tool::FieldPaint => {
                                        if let Some(field) = self.dynamics_fields.get(&id) {
                                            self.field_stroke_before =
                                                Some((id, field.values.clone()));
                                        }
                                    }
                                    Tool::TexturePaint => {
                                        self.texture_paint_last_uv = None;
                                        if let Some(mask) = self.painted_masks.get(&id) {
                                            self.texture_paint_stroke_before =
                                                Some((id, mask.pixels.clone()));
                                        }
                                    }
                                    _ => {}
                                }
                            }
                        }
                        if down {
                            match self.active_tool {
                                Tool::FieldPaint => {
                                    self.paint_selected_field(pointer, response.rect)
                                }
                                Tool::TexturePaint => {
                                    self.paint_selected_texture(pointer, response.rect)
                                }
                                _ => {}
                            }
                        }
                    }
                    if released {
                        match self.active_tool {
                            Tool::FieldPaint => {
                                if let Some((id, before)) = self.field_stroke_before.take() {
                                    let changed = self
                                        .dynamics_fields
                                        .get(&id)
                                        .is_some_and(|field| field.values != before);
                                    if changed {
                                        self.field_undo.push((id, before));
                                        self.field_redo.clear();
                                    }
                                }
                            }
                            Tool::TexturePaint => {
                                self.texture_paint_last_uv = None;
                                if let Some((id, before)) = self.texture_paint_stroke_before.take()
                                {
                                    let changed = self
                                        .painted_masks
                                        .get(&id)
                                        .is_some_and(|mask| mask.pixels != before);
                                    if changed {
                                        self.texture_paint_undo.push((id, before));
                                        self.texture_paint_redo.clear();
                                        if self.texture_paint_undo.len() > 16 {
                                            self.texture_paint_undo.remove(0);
                                        }
                                    }
                                }
                            }
                            _ => {}
                        }
                    }
                }
                let orbit = if response.dragged_by(egui::PointerButton::Secondary) {
                    pointer_delta
                } else {
                    Vec2::ZERO
                };
                let pan = if response.dragged_by(egui::PointerButton::Middle) {
                    pointer_delta
                } else {
                    Vec2::ZERO
                };
                let zoom_log = if response.hovered() {
                    ui.input(|input| input.smooth_scroll_delta.y) * 0.001
                } else {
                    0.0
                };
                if orbit != Vec2::ZERO || pan != Vec2::ZERO || zoom_log != 0.0 {
                    let sample = InputSample {
                        orbit,
                        pan,
                        zoom_log,
                        viewport_extent: response.rect.width().min(response.rect.height()),
                    };
                    self.apply_editor_camera_input(sample);
                    self.input_worker.submit(sample);
                }
                self.schedule_compositor_lod_update(ctx);
                self.refresh_preview_cache();
                let preview = Arc::clone(&self.cached_preview);
                let lighting = self.viewport_lighting(&preview);
                let frame_now = Instant::now();
                let viewport_frame_due = frame_now >= self.next_viewport_frame;
                if viewport_frame_due {
                    self.next_viewport_frame = advance_frame_deadline(
                        self.next_viewport_frame,
                        frame_now,
                        self.target_fps,
                    );
                }
                ctx.request_repaint_after(
                    self.next_viewport_frame
                        .saturating_duration_since(Instant::now()),
                );
                let viewport_texture = {
                    let key = DepthCacheKey {
                        size: [
                            render_rect.width().round().max(1.0) as usize,
                            render_rect.height().round().max(1.0) as usize,
                        ],
                        yaw: render_camera.0,
                        pitch: render_camera.1,
                        roll: render_camera.2,
                        zoom: render_camera.3,
                        target: render_camera.4,
                        grid_spacing: self.grid_spacing,
                        projection: render_camera.6,
                        scene_revision: self.scene_revision,
                        texture_revision: self.texture_revision,
                        global_light_enabled: self.global_light_enabled,
                        global_shadow_resolution: self.global_shadow_resolution,
                        shadow_quality: self.shadow_quality,
                        shadow_blur_radius: self.shadow_blur_radius,
                        show_grid: self.show_grid && scene_camera.is_none(),
                        mode: self.viewport_mode,
                        tool: self.active_tool,
                        device: self.render_device,
                    };
                    let mut newest_completed = None;
                    while let Ok(result) = self.display_worker.results.try_recv() {
                        newest_completed = Some(result);
                    }
                    if let Some(result) = newest_completed {
                        self.viewport_render_in_flight = false;
                        self.viewport_requested_key = None;
                        let presented_assets = completed_asset_loads(
                            &self.asset_loading_present_revision,
                            result.key.scene_revision,
                        );
                        for path in presented_assets {
                            self.asset_loading_present_revision.remove(&path);
                            self.asset_loading.remove(&path);
                        }
                        self.performance
                            .viewport_queue_wait
                            .record(result.queue_wait);
                        self.performance
                            .shadow_prepare
                            .record(result.shadow_prepare_time);
                        self.performance
                            .viewport_resource_upload
                            .record(result.resource_upload_time);
                        self.performance
                            .viewport_vertex_upload
                            .record(result.vertex_upload_time);
                        self.performance
                            .viewport_texture_upload
                            .record(result.texture_upload_time);
                        self.performance
                            .viewport_target_allocation
                            .record(result.viewport_target_allocation_time);
                        self.performance
                            .shadow_target_allocation
                            .record(result.shadow_target_allocation_time);
                        if !result.renderer_initialization_time.is_zero() {
                            self.performance
                                .viewport_initialization
                                .record(result.renderer_initialization_time);
                        }
                        if let Some((revision, started)) = self.compositor_present_revision {
                            if result.key.texture_revision >= revision {
                                self.performance
                                    .control_to_present
                                    .record(started.elapsed());
                                self.compositor_present_revision = None;
                                self.compositor_control_started = None;
                            }
                        }
                        match result.device {
                            RenderDevice::Vulkan => {
                                self.performance.viewport_vulkan.record(result.render_time);
                                self.performance
                                    .viewport_prepare
                                    .record(result.prepare_time);
                            }
                            RenderDevice::Cpu => {
                                self.performance.viewport_cpu.record(result.render_time)
                            }
                        }
                        let presentation_started = Instant::now();
                        self.viewport_depth = result.frame.linear_depth;
                        match result.frame.color {
                            FrameColor::Cpu(color) => {
                                if let Some(texture) = &mut self.viewport_color {
                                    texture.set(color, TextureOptions::NEAREST);
                                } else {
                                    self.viewport_color = Some(ctx.load_texture(
                                        "viewport-depth-color",
                                        color,
                                        TextureOptions::NEAREST,
                                    ));
                                }
                                self.viewport_native_texture = None;
                                self.viewport_native_view = None;
                            }
                            FrameColor::Vulkan(image) => {
                                if let Some(render_state) = &self.wgpu_render_state {
                                    let view_key = Arc::as_ptr(&image.view) as usize;
                                    if self.viewport_native_view != Some(view_key) {
                                        let mut renderer = render_state.renderer.write();
                                        if let Some(id) = self.viewport_native_texture {
                                            renderer.update_egui_texture_from_wgpu_texture(
                                                &render_state.device,
                                                &image.view,
                                                wgpu::FilterMode::Nearest,
                                                id,
                                            );
                                        } else {
                                            self.viewport_native_texture =
                                                Some(renderer.register_native_texture(
                                                    &render_state.device,
                                                    &image.view,
                                                    wgpu::FilterMode::Nearest,
                                                ));
                                        }
                                        self.viewport_native_view = Some(view_key);
                                    }
                                }
                            }
                        }
                        self.performance
                            .viewport_present
                            .record(presentation_started.elapsed());
                        self.viewport_depth_key = Some(result.key);
                        self.presented_view = Some(PresentedView {
                            camera: result.camera,
                            triangles: result.triangles,
                            show_grid: result.show_grid,
                            mode: result.mode,
                            tool: result.tool,
                        });
                    }
                    if self.viewport_depth_key != Some(key)
                        && self.viewport_requested_key != Some(key)
                        && !self.viewport_render_in_flight
                        && viewport_frame_due
                    {
                        self.display_worker.submit_latest(RenderJob {
                            key,
                            viewport_size: render_rect.size(),
                            triangles: Arc::clone(&preview),
                            camera: render_camera,
                            lighting: lighting.clone(),
                            show_grid: self.show_grid && scene_camera.is_none(),
                            mode: self.viewport_mode,
                            tool: self.active_tool,
                            reusable_depth: std::mem::take(&mut self.viewport_depth),
                            device: self.render_device,
                            queued_at: Instant::now(),
                        });
                        self.viewport_requested_key = Some(key);
                        self.viewport_render_in_flight = true;
                    }
                    self.viewport_native_texture
                        .or_else(|| self.viewport_color.as_ref().map(TextureHandle::id))
                };
                if let Some(presented) = &self.presented_view {
                    painter.rect_filled(response.rect, 0.0, Color32::BLACK);
                    let collider_wireframe = scene_camera
                        .is_none()
                        .then(|| self.selected_collider_wireframe())
                        .unwrap_or_default();
                    let camera_wireframes = scene_camera
                        .is_none()
                        .then(|| self.camera_wireframes())
                        .unwrap_or_default();
                    draw_viewport(
                        &painter,
                        render_rect,
                        presented.show_grid,
                        presented.tool,
                        presented.camera,
                        &presented.triangles,
                        presented.mode,
                        viewport_texture,
                        &collider_wireframe,
                        &camera_wireframes,
                        if scene_camera.is_some() {
                            &[]
                        } else {
                            &lighting.points
                        },
                    );
                    if self.camera_preview_visible && scene_camera.is_none() {
                        painter.rect_filled(response.rect, 0.0, Color32::BLACK);
                        painter.text(
                            response.rect.center(),
                            Align2::CENTER_CENTER,
                            "No visible Camera",
                            FontId::proportional(14.0),
                            Color32::from_rgb(235, 190, 80),
                        );
                    }
                } else {
                    painter.rect_filled(response.rect, 0.0, Color32::from_rgb(21, 24, 31));
                }
            });
    }

    fn selected_collider_wireframe(&self) -> Vec<[[f32; 3]; 2]> {
        let Some(id) = self.scene.selected else {
            return Vec::new();
        };
        let Ok(node) = self.scene.tree.node(id) else {
            return Vec::new();
        };
        let Some(Component::Collider {
            shape,
            center,
            half_extents,
            radius,
            height,
            ..
        }) = node
            .components
            .iter()
            .find(|component| matches!(component, Component::Collider { .. }))
        else {
            return Vec::new();
        };
        let shape = ColliderShape::from_storage(shape);
        let transform = node.global_transform();
        collider_wireframe_segments(shape, *center, *half_extents, *radius, *height)
            .into_iter()
            .map(|segment| {
                segment.map(|point| {
                    let local = CoreVec3::new(point[0], point[1], point[2]);
                    let world = transform.translation
                        + transform
                            .rotation
                            .rotate(transform.scale.component_mul(local));
                    [world.x, world.y, world.z]
                })
            })
            .collect()
    }

    fn camera_wireframes(&self) -> Vec<[[f32; 3]; 2]> {
        let mut output = Vec::new();
        for (id, node) in self.scene.tree.iter() {
            if !self.scene.visible(id) {
                continue;
            }
            let Some(Component::Camera { .. }) = node
                .components
                .iter()
                .find(|component| matches!(component, Component::Camera { .. }))
            else {
                continue;
            };
            // Camera geometry is an editor gizmo, not a renderable frustum. Its
            // base shape ignores camera and grid settings; only authored object
            // scale is allowed to resize it.
            let depth = 1.0;
            let half_height = 0.35;
            let half_width = 0.55;
            let apex = CoreVec3::ZERO;
            let corners = [
                CoreVec3::new(-half_width, depth, -half_height),
                CoreVec3::new(half_width, depth, -half_height),
                CoreVec3::new(half_width, depth, half_height),
                CoreVec3::new(-half_width, depth, half_height),
            ];
            let transform = node.global_transform();
            let world = |point: CoreVec3| {
                let point = transform.translation
                    + transform
                        .rotation
                        .rotate(transform.scale.component_mul(point));
                [point.x, point.y, point.z]
            };
            for corner in corners {
                output.push([world(apex), world(corner)]);
            }
            for index in 0..4 {
                output.push([world(corners[index]), world(corners[(index + 1) % 4])]);
            }
            // A small roof chevron makes the camera's up direction readable.
            let roof_center = CoreVec3::new(0.0, depth, half_height * 1.3);
            output.push([world(corners[2]), world(roof_center)]);
            output.push([world(roof_center), world(corners[3])]);
        }
        output
    }

    fn active_scene_camera(&self) -> Option<SceneCameraPreview> {
        self.scene.tree.iter().find_map(|(id, node)| {
            if !self.scene.visible(id) {
                return None;
            }
            node.components.iter().find_map(|component| {
                let Component::Camera {
                    field_of_view_degrees,
                    projection,
                    aspect_ratio,
                    near_clip: _,
                    far_clip: _,
                    orthographic_size,
                } = component
                else {
                    return None;
                };
                Some(SceneCameraPreview {
                    transform: node.global_transform(),
                    perspective: projection != "orthographic",
                    field_of_view_degrees: *field_of_view_degrees,
                    aspect_ratio: aspect_ratio.clamp(0.1, 10.0),
                    orthographic_size: orthographic_size.max(0.001),
                })
            })
        })
    }

    fn status_bar(&self, ctx: &egui::Context) {
        egui::Area::new(Id::new("status"))
            .anchor(Align2::LEFT_BOTTOM, [0.0, 0.0])
            .show(ctx, |ui| {
                egui::Frame::new()
                    .fill(Color32::from_rgb(18, 20, 25))
                    .inner_margin(egui::Margin::symmetric(9, 4))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            let object_count = self.scene.tree.iter().count();
                            let triangle_count =
                                self.scene
                                    .tree
                                    .iter()
                                    .filter_map(|(_, node)| {
                                        node.components.iter().find_map(|component| match component
                                        {
                                            Component::Model { asset, .. } => self
                                                .imported_assets
                                                .iter()
                                                .find(|imported| imported.path == *asset)
                                                .map(|imported| imported.mesh.triangle_count()),
                                            _ => None,
                                        })
                                    })
                                    .sum::<usize>();
                            ui.small("ZeroFPS 0.1");
                            ui.separator();
                            ui.small(format!(
                                "{object_count} objects  •  {triangle_count} preview tris"
                            ));
                            ui.separator();
                            let timing = match self.render_device {
                                RenderDevice::Vulkan => self.performance.viewport_vulkan,
                                RenderDevice::Cpu => self.performance.viewport_cpu,
                            };
                            let fps = if timing.latest_ms > f64::EPSILON {
                                (1_000.0 / timing.latest_ms).min(self.target_fps as f64)
                            } else {
                                0.0
                            };
                            ui.small(format!(
                                "{} viewport {:.2} ms  •  {:.1} FPS ({} cap)",
                                self.render_device.label(),
                                timing.latest_ms,
                                fps,
                                self.target_fps
                            ));
                            ui.separator();
                            if !self.asset_loading.is_empty() {
                                ui.small(
                                    RichText::new(format!(
                                        "Loading {} asset{}",
                                        self.asset_loading.len(),
                                        if self.asset_loading.len() == 1 {
                                            ""
                                        } else {
                                            "s"
                                        }
                                    ))
                                    .color(Color32::from_rgb(235, 167, 88)),
                                );
                                ui.separator();
                            }
                            ui.small(if self.viewport_focused {
                                "Viewport focused  •  RMB orbit  •  Wheel zoom"
                            } else {
                                "Ready"
                            });
                        });
                    });
            });
    }

    fn loading_overlay(&self, ctx: &egui::Context) {
        let Some((path, phase)) = self.asset_loading.first_key_value() else {
            return;
        };
        let name = std::path::Path::new(path)
            .file_name()
            .and_then(|name| name.to_str())
            .unwrap_or(path);
        egui::Area::new(Id::new("asset_loading_overlay"))
            .anchor(Align2::CENTER_TOP, [0.0, 72.0])
            .order(egui::Order::Foreground)
            .show(ctx, |ui| {
                egui::Frame::popup(ui.style())
                    .fill(Color32::from_rgb(28, 30, 37))
                    .inner_margin(egui::Margin::symmetric(14, 10))
                    .show(ui, |ui| {
                        ui.horizontal(|ui| {
                            ui.add(egui::Spinner::new().size(18.0));
                            ui.vertical(|ui| {
                                ui.strong(format!("Loading {name}"));
                                ui.small(*phase);
                                if self.asset_loading.len() > 1 {
                                    ui.weak(format!(
                                        "{} additional assets queued",
                                        self.asset_loading.len() - 1
                                    ));
                                }
                            });
                        });
                    });
            });
    }
}

impl eframe::App for EditorApp {
    fn update(&mut self, ctx: &egui::Context, _frame: &mut eframe::Frame) {
        ctx.send_viewport_cmd(egui::ViewportCommand::Title(format!(
            "{} — ZeroFPS Project",
            self.project_display_name()
        )));
        self.poll_asset_imports();
        self.poll_save_as();
        self.poll_load_project();
        self.poll_compositor_image_import();
        self.sync_compositor_outputs();
        self.poll_vulkan_compositor(ctx);
        self.tick_compositor_time(ctx);
        self.apply_object_transform_graphs();
        self.poll_compositor_apply(ctx);
        self.tick_dynamics(ctx);
        self.poll_build(ctx);
        self.shortcuts(ctx);
        self.top_bar(ctx);
        match self.workspace_tab {
            WorkspaceTab::Scene => {
                self.hierarchy(ctx);
                self.inspector(ctx);
                self.bottom_panel(ctx);
                self.viewport(ctx);
            }
            WorkspaceTab::Compositing => self.compositing_workspace(ctx),
        }
        self.loading_overlay(ctx);
        self.status_bar(ctx);
        self.project_error_popup(ctx);
    }
}

fn configure_style(ctx: &egui::Context) {
    ctx.set_visuals(egui::Visuals::dark());
    let mut style = (*ctx.style()).clone();
    style.spacing.item_spacing = Vec2::new(7.0, 6.0);
    style.spacing.button_padding = Vec2::new(9.0, 5.0);
    style.visuals.selection.bg_fill = Color32::from_rgb(45, 101, 145);
    style.visuals.widgets.hovered.bg_fill = Color32::from_rgb(48, 53, 65);
    style.visuals.widgets.active.bg_fill = Color32::from_rgb(54, 111, 151);
    ctx.set_style(style);
}

fn panel_frame(fill: Color32) -> egui::Frame {
    egui::Frame::new()
        .fill(fill)
        .inner_margin(egui::Margin::same(8))
        .stroke(Stroke::new(1.0, Color32::from_rgb(45, 48, 57)))
}

fn compositor_link_curve(start: Pos2, end: Pos2) -> Vec<Pos2> {
    let handle = ((end.x - start.x).abs() * 0.5).max(35.0);
    let control_a = start + Vec2::new(handle, 0.0);
    let control_b = end - Vec2::new(handle, 0.0);
    (0..=24)
        .map(|step| {
            let t = step as f32 / 24.0;
            let inverse = 1.0 - t;
            let weights = [
                inverse * inverse * inverse,
                3.0 * inverse * inverse * t,
                3.0 * inverse * t * t,
                t * t * t,
            ];
            Pos2::new(
                start.x * weights[0]
                    + control_a.x * weights[1]
                    + control_b.x * weights[2]
                    + end.x * weights[3],
                start.y * weights[0]
                    + control_a.y * weights[1]
                    + control_b.y * weights[2]
                    + end.y * weights[3],
            )
        })
        .collect()
}

fn remap_curve_editor(ui: &mut egui::Ui, points: &mut [[f32; 2]], bezier: bool) -> bool {
    let desired = Vec2::new(ui.available_width().max(80.0), 170.0);
    let (response, painter) = ui.allocate_painter(desired, Sense::click_and_drag());
    let rect = response.rect.shrink(8.0);
    painter.rect_filled(rect, 2.0, Color32::from_rgb(17, 19, 24));
    for step in 0..=4 {
        let fraction = step as f32 / 4.0;
        let x = egui::lerp(rect.left()..=rect.right(), fraction);
        let y = egui::lerp(rect.bottom()..=rect.top(), fraction);
        painter.line_segment(
            [Pos2::new(x, rect.top()), Pos2::new(x, rect.bottom())],
            Stroke::new(1.0, Color32::from_rgb(42, 45, 53)),
        );
        painter.line_segment(
            [Pos2::new(rect.left(), y), Pos2::new(rect.right(), y)],
            Stroke::new(1.0, Color32::from_rgb(42, 45, 53)),
        );
    }
    let to_screen = |point: [f32; 2]| {
        Pos2::new(
            egui::lerp(rect.left()..=rect.right(), point[0]),
            egui::lerp(rect.bottom()..=rect.top(), point[1]),
        )
    };
    let mut changed = false;
    if response.dragged()
        && let Some(pointer) = response.interact_pointer_pos()
    {
        let index = [1_usize, 2]
            .into_iter()
            .min_by(|left, right| {
                pointer
                    .distance_sq(to_screen(points[*left]))
                    .total_cmp(&pointer.distance_sq(to_screen(points[*right])))
            })
            .unwrap_or(1);
        let x = ((pointer.x - rect.left()) / rect.width()).clamp(0.0, 1.0);
        let y = ((rect.bottom() - pointer.y) / rect.height()).clamp(0.0, 1.0);
        let lower = if index == 1 { 0.0 } else { points[1][0] };
        let upper = if index == 1 { points[2][0] } else { 1.0 };
        points[index] = [x.clamp(lower, upper), y];
        changed = true;
    }
    let curve: Vec<_> = if bezier {
        (0..=48)
            .map(|step| {
                let t = step as f32 / 48.0;
                let inverse = 1.0 - t;
                let weights = [
                    inverse.powi(3),
                    3.0 * inverse.powi(2) * t,
                    3.0 * inverse * t.powi(2),
                    t.powi(3),
                ];
                to_screen([
                    points
                        .iter()
                        .zip(weights)
                        .map(|(point, weight)| point[0] * weight)
                        .sum(),
                    points
                        .iter()
                        .zip(weights)
                        .map(|(point, weight)| point[1] * weight)
                        .sum(),
                ])
            })
            .collect()
    } else {
        points.iter().copied().map(to_screen).collect()
    };
    painter.line(curve, Stroke::new(2.5, Color32::from_rgb(108, 190, 255)));
    if bezier {
        for pair in points.windows(2) {
            painter.line_segment(
                [to_screen(pair[0]), to_screen(pair[1])],
                Stroke::new(1.0, Color32::from_gray(105)),
            );
        }
    }
    for (index, point) in points.iter().enumerate() {
        painter.circle_filled(
            to_screen(*point),
            if index == 0 || index + 1 == points.len() {
                4.0
            } else {
                6.0
            },
            if index == 0 || index + 1 == points.len() {
                Color32::from_gray(150)
            } else {
                Color32::from_rgb(218, 190, 92)
            },
        );
    }
    changed
}

fn parse_compositor_position(value: &str) -> Option<Vec2> {
    let (x, y) = value.split_once(',')?;
    Some(Vec2::new(x.parse().ok()?, y.parse().ok()?))
}

fn object_index_by_name(names: &[String], target: &str) -> Option<usize> {
    names.iter().position(|name| name == target)
}

fn encode_hex(bytes: &[u8]) -> String {
    const HEX: &[u8; 16] = b"0123456789abcdef";
    let mut encoded = String::with_capacity(bytes.len() * 2);
    for &byte in bytes {
        encoded.push(HEX[(byte >> 4) as usize] as char);
        encoded.push(HEX[(byte & 0x0f) as usize] as char);
    }
    encoded
}

fn decode_hex(encoded: &str) -> Option<Vec<u8>> {
    if encoded.len() % 2 != 0 {
        return None;
    }
    encoded
        .as_bytes()
        .chunks_exact(2)
        .map(|pair| {
            let high = (pair[0] as char).to_digit(16)?;
            let low = (pair[1] as char).to_digit(16)?;
            Some(((high << 4) | low) as u8)
        })
        .collect()
}

fn barycentric_2d(point: Pos2, triangle: [Pos2; 3]) -> Option<[f32; 3]> {
    let edge0 = triangle[1] - triangle[0];
    let edge1 = triangle[2] - triangle[0];
    let relative = point - triangle[0];
    let denominator = edge0.x * edge1.y - edge1.x * edge0.y;
    if !denominator.is_finite() || denominator.abs() < 1.0e-8 {
        return None;
    }
    let weight1 = (relative.x * edge1.y - edge1.x * relative.y) / denominator;
    let weight2 = (edge0.x * relative.y - relative.x * edge0.y) / denominator;
    let weight0 = 1.0 - weight1 - weight2;
    let weights = [weight0, weight1, weight2];
    (weights
        .iter()
        .all(|weight| *weight >= -1.0e-4 && weight.is_finite()))
    .then_some(weights)
}

#[allow(clippy::too_many_arguments)]
fn area_weighted_particle_masses(
    mesh: &MeshAsset,
    mask: &PaintedMask,
    object_scale: CoreVec3,
    base_density: f32,
    density_scale: f32,
    minimum_mass: f32,
    normalize: bool,
    total_mass: f32,
) -> Vec<f32> {
    let minimum_mass = minimum_mass.max(1.0e-6);
    let density_factor = base_density.max(0.0) * density_scale.max(0.0);
    let densities = mesh
        .vertices
        .iter()
        .map(|vertex| sample_painted_mask(mask, vertex.uv) * density_factor)
        .collect::<Vec<_>>();
    let mut masses = vec![0.0; mesh.vertices.len()];
    for triangle in mesh
        .primitives
        .iter()
        .flat_map(|primitive| primitive.indices.chunks_exact(3))
    {
        let indices = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        if indices.iter().any(|index| *index >= mesh.vertices.len()) {
            continue;
        }
        let point = |index: usize| {
            let position = mesh.vertices[index].position;
            object_scale.component_mul(CoreVec3::new(position[0], position[1], position[2]))
        };
        let area = (point(indices[1]) - point(indices[0]))
            .cross(point(indices[2]) - point(indices[0]))
            .length()
            * 0.5;
        let density = (densities[indices[0]] + densities[indices[1]] + densities[indices[2]]) / 3.0;
        let share = (area * density / 3.0).max(0.0);
        for index in indices {
            masses[index] += share;
        }
    }
    for mass in &mut masses {
        *mass = mass.max(minimum_mass);
    }
    if normalize {
        let current_total = masses.iter().sum::<f32>();
        if current_total > f32::EPSILON {
            let factor = total_mass.max(minimum_mass) / current_total;
            for mass in &mut masses {
                *mass = (*mass * factor).max(minimum_mass);
            }
        }
    }
    masses
}

fn sample_painted_mask(mask: &PaintedMask, uv: [f32; 2]) -> f32 {
    if mask.width == 0 || mask.height == 0 {
        return 0.0;
    }
    let x = (uv[0].rem_euclid(1.0) * mask.width as f32).floor() as usize % mask.width as usize;
    let y = (uv[1].rem_euclid(1.0) * mask.height as f32).floor() as usize % mask.height as usize;
    mask.pixels
        .get(y * mask.width as usize + x)
        .copied()
        .unwrap_or(0) as f32
        / 255.0
}

fn evaluate_force_formula(formula: &str, x: f32, y: f32, z: f32, t: f32) -> Result<f32, String> {
    let mut parser = FormulaParser {
        input: formula.as_bytes(),
        cursor: 0,
        variables: [x, y, z, t],
    };
    let value = parser.expression()?;
    parser.whitespace();
    if parser.cursor != parser.input.len() {
        return Err("unexpected trailing expression".into());
    }
    value
        .is_finite()
        .then_some(value)
        .ok_or_else(|| "formula produced a non-finite value".into())
}

struct FormulaParser<'a> {
    input: &'a [u8],
    cursor: usize,
    variables: [f32; 4],
}

impl FormulaParser<'_> {
    fn expression(&mut self) -> Result<f32, String> {
        let mut value = self.term()?;
        loop {
            self.whitespace();
            if self.consume(b'+') {
                value += self.term()?;
            } else if self.consume(b'-') {
                value -= self.term()?;
            } else {
                return Ok(value);
            }
        }
    }

    fn term(&mut self) -> Result<f32, String> {
        let mut value = self.power()?;
        loop {
            self.whitespace();
            if self.consume(b'*') {
                value *= self.power()?;
            } else if self.consume(b'/') {
                value /= self.power()?;
            } else {
                return Ok(value);
            }
        }
    }

    fn power(&mut self) -> Result<f32, String> {
        let value = self.unary()?;
        self.whitespace();
        if self.consume(b'^') {
            Ok(value.powf(self.power()?))
        } else {
            Ok(value)
        }
    }

    fn unary(&mut self) -> Result<f32, String> {
        self.whitespace();
        if self.consume(b'+') {
            self.unary()
        } else if self.consume(b'-') {
            Ok(-self.unary()?)
        } else {
            self.primary()
        }
    }

    fn primary(&mut self) -> Result<f32, String> {
        self.whitespace();
        if self.consume(b'(') {
            let value = self.expression()?;
            self.whitespace();
            return self
                .consume(b')')
                .then_some(value)
                .ok_or_else(|| "missing closing parenthesis".into());
        }
        if self
            .input
            .get(self.cursor)
            .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'.')
        {
            return self.number();
        }
        let identifier = self.identifier()?;
        match identifier.as_str() {
            "x" => Ok(self.variables[0]),
            "y" => Ok(self.variables[1]),
            "z" => Ok(self.variables[2]),
            "t" => Ok(self.variables[3]),
            "pi" => Ok(std::f32::consts::PI),
            "e" => Ok(std::f32::consts::E),
            function => {
                self.whitespace();
                if !self.consume(b'(') {
                    return Err(format!("unknown variable `{function}`"));
                }
                let argument = self.expression()?;
                self.whitespace();
                if !self.consume(b')') {
                    return Err("missing function parenthesis".into());
                }
                match function {
                    "sin" => Ok(argument.sin()),
                    "cos" => Ok(argument.cos()),
                    "abs" => Ok(argument.abs()),
                    "sqrt" if argument >= 0.0 => Ok(argument.sqrt()),
                    "sqrt" => Err("sqrt requires a non-negative value".into()),
                    _ => Err(format!("unknown function `{function}`")),
                }
            }
        }
    }

    fn number(&mut self) -> Result<f32, String> {
        let start = self.cursor;
        while self.input.get(self.cursor).is_some_and(|byte| {
            byte.is_ascii_digit() || matches!(*byte, b'.' | b'e' | b'E' | b'+' | b'-')
        }) {
            if self.cursor > start
                && matches!(self.input[self.cursor], b'+' | b'-')
                && !matches!(self.input[self.cursor - 1], b'e' | b'E')
            {
                break;
            }
            self.cursor += 1;
        }
        std::str::from_utf8(&self.input[start..self.cursor])
            .ok()
            .and_then(|number| number.parse().ok())
            .ok_or_else(|| "invalid number".into())
    }

    fn identifier(&mut self) -> Result<String, String> {
        let start = self.cursor;
        while self
            .input
            .get(self.cursor)
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        {
            self.cursor += 1;
        }
        (self.cursor > start)
            .then(|| String::from_utf8_lossy(&self.input[start..self.cursor]).into_owned())
            .ok_or_else(|| "expected a value".into())
    }

    fn whitespace(&mut self) {
        while self
            .input
            .get(self.cursor)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            self.cursor += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.input.get(self.cursor) == Some(&expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }
}

fn resolve_object_handle_value(nodes: &[CompositorNode], node_id: usize) -> Option<f32> {
    fn resolve(
        nodes: &[CompositorNode],
        node_id: usize,
        visiting: &mut BTreeSet<usize>,
    ) -> Option<f32> {
        if !visiting.insert(node_id) {
            return None;
        }
        let value = match &nodes.iter().find(|node| node.id == node_id)?.settings {
            NodeSettings::ObjectHandle {
                value,
                source_handle,
                ..
            } => match source_handle {
                Some(source) => resolve(nodes, *source, visiting),
                None => Some(*value),
            },
            _ => None,
        };
        visiting.remove(&node_id);
        value.filter(|value| value.is_finite())
    }
    resolve(nodes, node_id, &mut BTreeSet::new())
}

fn compositor_input_count(kind: usize, combine_mode: usize) -> usize {
    match kind {
        0 | 1 | 2 | 14 | 15 | 19 => 0,
        9 => {
            if combine_mode == 1 {
                3
            } else {
                2
            }
        }
        13 => 4,
        17 | 25 | 26 | 28 | 30 => 3,
        18 => 1,
        22 => 3,
        23 => 1,
        24 | 27 | 29 => 2,
        _ => 1,
    }
}

fn compositor_input_socket_y(kind: usize, input: usize) -> f32 {
    match kind {
        9 => 85.0 + input as f32 * 30.0,
        13 => 75.0 + input as f32 * 22.0,
        17 | 22 | 24 | 25 | 26 | 27 | 28 | 29 | 30 => 74.0 + input as f32 * 26.0,
        _ => 70.0,
    }
}

fn compositor_output_socket_y(kind: usize, output: usize) -> f32 {
    match kind {
        11 => 70.0 + output as f32 * 22.0,
        27 => 74.0 + output as f32 * 26.0,
        _ => 70.0,
    }
}

fn compositor_controls_top(kind: usize, combine_mode: usize) -> f32 {
    let input_bottom = compositor_input_count(kind, combine_mode)
        .checked_sub(1)
        .map(|input| compositor_input_socket_y(kind, input));
    let output_bottom = compositor_output_count(kind)
        .checked_sub(1)
        .map(|output| compositor_output_socket_y(kind, output));
    input_bottom
        .into_iter()
        .chain(output_bottom)
        .fold(48.0_f32, f32::max)
        + 22.0
}

fn compositor_node_height(base_height: f32, kind: usize, combine_mode: usize) -> f32 {
    base_height + (compositor_controls_top(kind, combine_mode) - 62.0).max(0.0)
}

fn compositor_output_count(kind: usize) -> usize {
    match kind {
        8 | 16 | 17 | 18 | 28 | 29 => 0,
        11 => 4,
        27 => 2,
        _ => 1,
    }
}

fn compositor_input_label(kind: usize, input: usize) -> &'static str {
    match kind {
        3 => "Value",
        4 => "Value",
        5 | 6 => "Value",
        7 => "Image",
        8 => "Texture",
        9 => ["A", "B", "Alpha"][input],
        10..=12 => "Color",
        13 => ["R", "G", "B", "A"][input],
        16 => "Value",
        17 => ["Position", "Rotation", "Scale"][input],
        18 => "Mesh",
        20 => "Density Texture",
        21 => "Particle Mass",
        22 => ["Strength Texture", "Mass Field", "Spring Mesh"][input],
        23 => "Strength Texture",
        24 => ["Velocity Field", "Force Field"][input],
        25 | 26 => ["X", "Y", "Z"][input],
        27 => ["Initial Position", "Initial Rotation"][input],
        28 => ["X", "Y", "Z"][input],
        29 => ["Throttle", "Torque"][input],
        30 => ["X", "Y", "Z"][input],
        _ => "Input",
    }
}

fn compositor_output_label(kind: usize, output: usize) -> &'static str {
    match kind {
        11 => ["R", "G", "B", "A"][output],
        20 => "Mass Field",
        21 => "Spring Mesh",
        22 => "Force Field",
        23 => "Velocity Field",
        24 => "Mesh",
        25 => "Position",
        26 => "Rotation",
        27 => ["Position", "Rotation"][output],
        0 | 1 | 7 | 8 | 9 | 10 | 12 | 13 | 19 => "Texture",
        3 | 4 | 5 | 6 | 14 | 15 | 30 => "Value",
        2 => "Value / Color",
        16 => "Preview",
        _ => "Output",
    }
}

fn compositor_node_description(kind: usize) -> &'static str {
    match kind {
        0 => "Reads a texture channel from a scene object.",
        1 => "Loads an external image asset.",
        2 => "Provides a constant scalar or RGB color.",
        3 => "Remaps values through a polyline or Bézier curve.",
        4 => "Applies scalar or texture algebra.",
        5 => "Converts values below the threshold to 0 and values above it to 1.",
        6 => "Applies a threshold with a smooth transition.",
        7 => "Applies a selectable image-space filter.",
        8 => "Overwrites one texture channel on the current object.",
        9 => "Combines two textures using algebra or alpha mixing.",
        10 => "Converts between supported color spaces.",
        11 => "Splits an RGBA texture into four channels.",
        12 => "Converts a color texture to grayscale.",
        13 => "Joins four scalar channels into an RGBA texture.",
        14 => "Exposes a live scalar control in Scene mode.",
        15 => "Outputs simulation or wall-clock time.",
        16 => "Displays an intermediate node result.",
        17 => "Applies position, rotation, and scale to the object.",
        18 => "Applies the final generated mesh to the object.",
        19 => "Creates a scalar texture with the in-app mesh painter.",
        20 => "Converts a density texture into particle masses.",
        21 => "Creates stretch and bending constraints from a mesh.",
        22 => "Evaluates a Cartesian force field over mesh particles.",
        23 => "Evaluates a Cartesian velocity field over mesh particles.",
        24 => "Runs deformable mesh simulation.",
        25 => "Builds a position vector from X, Y, and Z values.",
        26 => "Builds an Euler rotation from X, Y, and Z angles.",
        27 => "Runs fixed-step rigid position and rotation simulation.",
        28 => "Collects an object's force and propagates it to simulated ancestors.",
        29 => "Drives an Engine cylinder around its local Z axle.",
        30 => "Evaluates an algebraic expression using up to three inputs.",
        _ => "Node",
    }
}

fn compositor_control_layer(parent: egui::LayerId, node: usize) -> egui::LayerId {
    egui::LayerId::new(parent.order, Id::new(("compositor_node_controls", node)))
}

fn compositor_add_button(ui: &mut egui::Ui, enabled: bool, label: &str) -> bool {
    let clicked = ui.add_enabled(enabled, egui::Button::new(label)).clicked();
    if clicked {
        ui.close_menu();
    }
    clicked
}

fn compositor_centered_position(canvas: Rect, origin: Pos2, scale: f32, node_size: Vec2) -> Vec2 {
    (canvas.center() - origin) / scale - node_size * 0.5
}

fn select_compositor_lod(projected_extent: f32, current: u32) -> u32 {
    let demand = (projected_extent.max(1.0) * 1.5).ceil() as u32;
    let desired = demand.next_power_of_two().clamp(128, 4096);
    if current == u32::MAX {
        return desired;
    }
    if desired > current && demand as f32 <= current as f32 * 1.15 {
        current
    } else if desired < current && demand as f32 >= current as f32 * 0.45 {
        current
    } else {
        desired
    }
}

fn scaled_modulated_time(seconds: f32, scale: f32, modulus: f32) -> f32 {
    let scaled = seconds * scale;
    if modulus > 0.0 {
        scaled.rem_euclid(modulus)
    } else {
        scaled
    }
}

fn select_compositor_lod_for_backend(
    projected_extent: f32,
    current: u32,
    backend: RenderDevice,
) -> u32 {
    match backend {
        RenderDevice::Vulkan => u32::MAX,
        RenderDevice::Cpu => select_compositor_lod(projected_extent, current),
    }
}

fn resize_texture_for_lod(texture: &Arc<TextureAsset>, maximum: u32) -> Arc<TextureAsset> {
    let largest = texture.width.max(texture.height);
    if maximum == u32::MAX || largest <= maximum || texture.width == 0 || texture.height == 0 {
        return Arc::clone(texture);
    }
    let scale = maximum as f32 / largest as f32;
    let width = ((texture.width as f32 * scale).round() as u32).max(1);
    let height = ((texture.height as f32 * scale).round() as u32).max(1);
    let Some(image) =
        image::RgbaImage::from_raw(texture.width, texture.height, texture.pixels.clone())
    else {
        return Arc::clone(texture);
    };
    let resized =
        image::imageops::resize(&image, width, height, image::imageops::FilterType::Triangle);
    Arc::new(TextureAsset {
        name: format!("{}-lod-{maximum}", texture.name),
        width,
        height,
        pixels: resized.into_raw(),
        cached_mips: Vec::new(),
    })
}

fn combine_compositor_textures(
    a: &TextureAsset,
    b: &TextureAsset,
    alpha_texture: Option<&TextureAsset>,
    mode: usize,
    operation: usize,
    fallback_alpha: f32,
) -> TextureAsset {
    let width = a.width.max(b.width).max(1);
    let height = a.height.max(b.height).max(1);
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            let uv = [
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            ];
            let left = sample_texture_nearest(a, uv);
            let right = sample_texture_nearest(b, uv);
            let alpha = alpha_texture
                .map(|texture| sample_texture_nearest(texture, uv)[0])
                .unwrap_or(fallback_alpha)
                .clamp(0.0, 1.0);
            for channel in 0..4 {
                let value = if mode == 1 {
                    alpha * left[channel] + (1.0 - alpha) * right[channel]
                } else {
                    match operation {
                        0 => left[channel] + right[channel],
                        1 => left[channel] - right[channel],
                        2 => left[channel] * right[channel],
                        3 => {
                            if right[channel].abs() <= f32::EPSILON {
                                0.0
                            } else {
                                left[channel] / right[channel]
                            }
                        }
                        4 => left[channel].max(0.0).powf(right[channel]),
                        5 => left[channel].min(right[channel]),
                        6 => left[channel].max(right[channel]),
                        _ => (left[channel] - right[channel]).abs(),
                    }
                };
                pixels.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    TextureAsset {
        name: "compositor-combine".into(),
        width,
        height,
        pixels,
        cached_mips: Vec::new(),
    }
}

fn apply_compositor_algebra(
    inputs: &[Option<Arc<TextureAsset>>; 3],
    program: &[compositor_graph::AlgebraInstruction],
) -> TextureAsset {
    let width = inputs
        .iter()
        .flatten()
        .map(|texture| texture.width)
        .max()
        .unwrap_or(1);
    let height = inputs
        .iter()
        .flatten()
        .map(|texture| texture.height)
        .max()
        .unwrap_or(1);
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            let uv = [
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            ];
            let samples = inputs.each_ref().map(|input| {
                input
                    .as_deref()
                    .map(|texture| sample_texture_nearest(texture, uv))
                    .unwrap_or([0.0; 4])
            });
            for channel in 0..4 {
                let variables = [
                    samples[0][channel],
                    samples[1][channel],
                    samples[2][channel],
                ];
                let value =
                    compositor_graph::evaluate_algebra_program(program, variables).unwrap_or(0.0);
                pixels.push((value.clamp(0.0, 1.0) * 255.0).round() as u8);
            }
        }
    }
    TextureAsset {
        name: "compositor-algebra".into(),
        width,
        height,
        pixels,
        cached_mips: Vec::new(),
    }
}

fn apply_compositor_math(
    mut texture: TextureAsset,
    operation: usize,
    fallback: f32,
) -> TextureAsset {
    for_each_rgba_parallel(&mut texture.pixels, |pixel| {
        for channel in &mut pixel[..3] {
            let value = *channel as f32 / 255.0;
            let result = match operation {
                0 => value + fallback,
                1 => value - fallback,
                2 => value * fallback,
                3 => {
                    if fallback.abs() <= f32::EPSILON {
                        0.0
                    } else {
                        value / fallback
                    }
                }
                4 => value.max(0.0).powf(fallback),
                5 => value.min(fallback),
                6 => value.max(fallback),
                _ => (value - fallback).abs(),
            };
            *channel = (result.clamp(0.0, 1.0) * 255.0).round() as u8;
        }
    });
    texture.name = "compositor-math".into();
    texture
}

fn convert_compositor_color_space(
    mut texture: TextureAsset,
    from: usize,
    to: usize,
) -> TextureAsset {
    if from != to {
        for_each_rgba_parallel(&mut texture.pixels, |pixel| {
            for channel in &mut pixel[..3] {
                let value = *channel as f32 / 255.0;
                let converted = if from == 0 {
                    if value <= 0.04045 {
                        value / 12.92
                    } else {
                        ((value + 0.055) / 1.055).powf(2.4)
                    }
                } else if value <= 0.003_130_8 {
                    value * 12.92
                } else {
                    1.055 * value.powf(1.0 / 2.4) - 0.055
                };
                *channel = (converted.clamp(0.0, 1.0) * 255.0).round() as u8;
            }
        });
    }
    texture.name = "compositor-color-space".into();
    texture
}

fn extract_compositor_channel(mut texture: TextureAsset, channel: usize) -> TextureAsset {
    for_each_rgba_parallel(&mut texture.pixels, |pixel| {
        let value = pixel[channel.min(3)];
        pixel[0] = value;
        pixel[1] = value;
        pixel[2] = value;
        pixel[3] = 255;
    });
    texture.name = "compositor-channel".into();
    texture
}

fn grayscale_compositor_texture(mut texture: TextureAsset, mode: usize) -> TextureAsset {
    for_each_rgba_parallel(&mut texture.pixels, |pixel| {
        let [red, green, blue] = [pixel[0] as f32, pixel[1] as f32, pixel[2] as f32];
        let gray = match mode {
            1 => (red + green + blue) / 3.0,
            2 => {
                let maximum = red.max(green).max(blue);
                let minimum = red.min(green).min(blue);
                (maximum + minimum) * 0.5
            }
            _ => 0.2126 * red + 0.7152 * green + 0.0722 * blue,
        }
        .round() as u8;
        pixel[0] = gray;
        pixel[1] = gray;
        pixel[2] = gray;
    });
    texture.name = "compositor-grayscale".into();
    texture
}

fn for_each_rgba_parallel(pixels: &mut [u8], operation: impl Fn(&mut [u8]) + Sync) {
    let pixel_count = pixels.len() / 4;
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(8);
    if pixel_count < 65_536 || workers == 1 {
        for pixel in pixels.chunks_exact_mut(4) {
            operation(pixel);
        }
        return;
    }
    let chunk_pixels = pixel_count.div_ceil(workers);
    let chunk_bytes = chunk_pixels * 4;
    std::thread::scope(|scope| {
        for chunk in pixels.chunks_mut(chunk_bytes) {
            let operation = &operation;
            scope.spawn(move || {
                for pixel in chunk.chunks_exact_mut(4) {
                    operation(pixel);
                }
            });
        }
    });
}

fn join_compositor_channels(rgb: [&TextureAsset; 3], alpha: Option<&TextureAsset>) -> TextureAsset {
    let width = rgb.iter().map(|texture| texture.width).max().unwrap_or(1);
    let height = rgb.iter().map(|texture| texture.height).max().unwrap_or(1);
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            let uv = [
                (x as f32 + 0.5) / width as f32,
                (y as f32 + 0.5) / height as f32,
            ];
            for texture in rgb {
                pixels.push(
                    (sample_texture_nearest(texture, uv)[0].clamp(0.0, 1.0) * 255.0).round() as u8,
                );
            }
            pixels.push(
                alpha
                    .map(|texture| {
                        (sample_texture_nearest(texture, uv)[0].clamp(0.0, 1.0) * 255.0).round()
                            as u8
                    })
                    .unwrap_or(255),
            );
        }
    }
    TextureAsset {
        name: "compositor-channel-join".into(),
        width,
        height,
        pixels,
        cached_mips: Vec::new(),
    }
}

fn parse_compositor_link(value: &str) -> Option<(usize, usize, usize, usize)> {
    let mut fields = value.split(',');
    let first: usize = fields.next()?.parse().ok()?;
    let second: usize = fields.next()?.parse().ok()?;
    let third_str = fields.next();
    let fourth_str = fields.next();
    if fields.next().is_some() {
        return None;
    }
    match (third_str, fourth_str) {
        (Some(third), Some(fourth)) => {
            // 4-number format: from_id, from_output, to_id, to_input
            let from_output: usize = third.parse().ok()?;
            let to_id: usize = second; // wait, reorder: first=from_id, second=from_output, third=to_id, fourth=to_input
            let _ = to_id;
            let from_id = first;
            let from_out: usize = second;
            let to: usize = third.parse().ok()?;
            let input: usize = fourth.parse().ok()?;
            let _ = from_output;
            Some((from_id, from_out, to, input))
        }
        (Some(third), None) => {
            // 3-number format (old): from, to, input — from_output=0
            let to: usize = second;
            let input: usize = third.parse().ok()?;
            Some((first, 0, to, input))
        }
        (None, _) => {
            // 2-number format (very old): from, to — input=0, from_output=0
            Some((first, 0, second, 0))
        }
    }
}

fn parse_compositor_color(value: &str) -> Option<[f32; 3]> {
    let mut channels = value.split(',');
    let color = [
        channels.next()?.parse().ok()?,
        channels.next()?.parse().ok()?,
        channels.next()?.parse().ok()?,
    ];
    channels.next().is_none().then_some(color)
}

fn parse_compositor_bezier(value: &str) -> Option<Vec<[f32; 2]>> {
    let points: Option<Vec<_>> = value
        .split(';')
        .map(|point| {
            let (x, y) = point.split_once(',')?;
            Some([x.parse().ok()?, y.parse().ok()?])
        })
        .collect();
    points.filter(|points| points.len() == 4)
}

fn safe_bundle_filename(name: &str) -> String {
    name.chars()
        .map(|character| {
            if character.is_ascii_alphanumeric() || matches!(character, '.' | '-' | '_') {
                character
            } else {
                '_'
            }
        })
        .collect()
}

fn rewrite_asset_paths<T: AsRef<std::path::Path>>(
    project: &mut ProjectFile,
    mapping: &BTreeMap<String, T>,
) {
    let ids: Vec<_> = project.scene.geometry.iter().map(|(id, _)| id).collect();
    for id in ids {
        let Ok(node) = project.scene.geometry.node_mut(id) else {
            continue;
        };
        for component in &mut node.components {
            if let Component::Model { asset, .. } = component
                && let Some(replacement) = mapping.get(asset)
            {
                *asset = replacement.as_ref().to_string_lossy().into_owned();
            }
        }
    }
}

fn rewrite_compositor_image_paths<T: AsRef<std::path::Path>>(
    project: &mut ProjectFile,
    mapping: &BTreeMap<String, T>,
) {
    for (key, path) in &mut project.project.properties {
        if (key.starts_with("compositor.image.") || key.ends_with(".image_archive"))
            && let Some(replacement) = mapping.get(path)
        {
            *path = replacement.as_ref().to_string_lossy().into_owned();
        }
    }
}

fn section_title(ui: &mut egui::Ui, name: &str, action: &str) {
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(name)
                .small()
                .strong()
                .color(Color32::from_gray(190)),
        );
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.small(action);
        });
    });
    ui.separator();
}

fn tool_button(ui: &mut egui::Ui, active: &mut Tool, value: Tool, key: &str, title: &str) {
    if ui
        .selectable_label(*active == value, format!("{key}  {title}"))
        .clicked()
    {
        *active = value;
    }
}

fn vector_editor(ui: &mut egui::Ui, label: &str, value: &mut [f32; 3], speed: f64) -> bool {
    ui.label(RichText::new(label).small().weak());
    ui.horizontal(|ui| {
        let [x, y, z] = value;
        let mut changed = false;
        for (axis, color, component) in [
            ("X", Color32::from_rgb(235, 91, 91), x),
            ("Y", Color32::from_rgb(96, 205, 125), y),
            ("Z", Color32::from_rgb(91, 145, 235), z),
        ] {
            ui.label(RichText::new(axis).strong().color(color));
            changed |= ui
                .add(
                    egui::DragValue::new(component)
                        .speed(speed)
                        .min_decimals(3)
                        .max_decimals(3),
                )
                .changed();
        }
        changed
    })
    .inner
}

fn property(ui: &mut egui::Ui, name: &str, value: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(name).weak());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.add(egui::Label::new(value).truncate())
                .on_hover_text(value);
        });
    });
}

fn inherited_property(ui: &mut egui::Ui, name: &str, value: &str, source: &str) {
    ui.horizontal(|ui| {
        ui.label(RichText::new(name).weak());
        ui.with_layout(Layout::right_to_left(Align::Center), |ui| {
            ui.label(value)
                .on_hover_text(format!("Inherited from {source}"));
            ui.label(RichText::new("↳").color(Color32::from_rgb(112, 174, 220)))
                .on_hover_text(format!("Inherited from {source}"));
        });
    });
}

fn tab(ui: &mut egui::Ui, active: &mut BottomTab, value: BottomTab, title: &str) {
    if ui.selectable_label(*active == value, title).clicked() {
        *active = value;
    }
}

fn scripts_panel(ui: &mut egui::Ui) {
    ui.horizontal(|ui| {
        for (name, detail) in [
            ("spin_controller.rs", "Component · clean"),
            ("player_input.rs", "Component · clean"),
            ("game.rs", "Entry point · clean"),
        ] {
            ui.group(|ui| {
                ui.set_min_size(Vec2::new(190.0, 62.0));
                ui.label(RichText::new("◇  ").color(Color32::from_rgb(235, 167, 88)));
                ui.strong(name);
                ui.small(RichText::new(detail).weak());
            });
        }
    });
}

fn console_panel(ui: &mut egui::Ui, logs: &[LogEntry]) {
    egui::ScrollArea::vertical()
        .stick_to_bottom(true)
        .show(ui, |ui| {
            for log in logs {
                ui.horizontal(|ui| {
                    ui.label(
                        RichText::new(log.level)
                            .monospace()
                            .color(log.color)
                            .strong(),
                    );
                    ui.label(RichText::new(&log.message).monospace());
                });
            }
        });
}

fn telemetry_panel(ui: &mut egui::Ui, state: PlayState, performance: &EditorPerformanceTelemetry) {
    let live = matches!(state, PlayState::Running | PlayState::Paused);
    ui.horizontal(|ui| {
        ui.label(
            RichText::new(if live { "● CONNECTED" } else { "○ OFFLINE" })
                .color(if live {
                    Color32::from_rgb(97, 219, 141)
                } else {
                    Color32::from_gray(130)
                })
                .strong(),
        );
        ui.separator();
        ui.label(format!(
            "Viewport worker  {:.3} ms",
            performance.viewport_vulkan.latest_ms
        ));
        ui.label(format!(
            "Graph submit  {:.3} ms",
            performance.compositor_vulkan_submit.latest_ms
        ));
        ui.label("Socket  loopback");
        ui.separator();
        if ui.button("Copy telemetry").clicked() {
            ui.ctx().copy_text(format_telemetry_report(performance));
        }
    });
    ui.add_space(8.0);
    egui::Grid::new("editor_execution_telemetry")
        .striped(true)
        .show(ui, |ui| {
            ui.strong("Stage");
            ui.strong("Latest");
            ui.strong("EMA");
            ui.strong("Maximum");
            ui.strong("Samples");
            ui.end_row();
            for (name, metric) in telemetry_metrics(performance) {
                ui.label(name);
                ui.monospace(format!("{:.3} ms", metric.latest_ms));
                ui.monospace(format!("{:.3} ms", metric.average_ms));
                ui.monospace(format!("{:.3} ms", metric.maximum_ms));
                ui.monospace(metric.samples.to_string());
                ui.end_row();
            }
        });
    ui.add_space(8.0);
    ui.small(
        "Vulkan viewport timing covers batch preparation plus command encoding/submission. The \
         resident color target is sampled directly by egui; depth is not read back during normal rendering.",
    );
}

fn telemetry_metrics(
    performance: &EditorPerformanceTelemetry,
) -> [(&'static str, TimingMetric); 18] {
    [
        ("Vulkan viewport worker", performance.viewport_vulkan),
        ("GPU batch preparation", performance.viewport_prepare),
        (
            "egui native texture presentation",
            performance.viewport_present,
        ),
        ("Viewport queue wait", performance.viewport_queue_wait),
        (
            "Shadow preparation / GPU encoding",
            performance.shadow_prepare,
        ),
        (
            "GPU resource loading / upload",
            performance.viewport_resource_upload,
        ),
        ("GPU mesh-buffer upload", performance.viewport_vertex_upload),
        (
            "GPU texture + mip upload",
            performance.viewport_texture_upload,
        ),
        (
            "Viewport target allocation",
            performance.viewport_target_allocation,
        ),
        (
            "Shadow target allocation",
            performance.shadow_target_allocation,
        ),
        (
            "Vulkan renderer initialization",
            performance.viewport_initialization,
        ),
        ("Control → graph apply", performance.control_to_graph_apply),
        (
            "Control → composite ready",
            performance.control_to_composite_ready,
        ),
        ("Control → presented frame", performance.control_to_present),
        ("Graph compilation", performance.graph_compile),
        ("Graph evaluation", performance.graph_evaluation),
        ("CPU viewport", performance.viewport_cpu),
        (
            "Vulkan graph encode + submission",
            performance.compositor_vulkan_submit,
        ),
    ]
}

fn format_telemetry_report(performance: &EditorPerformanceTelemetry) -> String {
    let mut report = String::from("Stage\tLatest\tEMA\tMaximum\tSamples\n");
    for (name, metric) in telemetry_metrics(performance) {
        report.push_str(&format!(
            "{name}\t{:.3} ms\t{:.3} ms\t{:.3} ms\t{}\n",
            metric.latest_ms, metric.average_ms, metric.maximum_ms, metric.samples
        ));
    }
    report
}

fn completed_asset_loads(pending: &BTreeMap<String, u64>, presented_revision: u64) -> Vec<String> {
    pending
        .iter()
        .filter_map(|(path, revision)| (presented_revision >= *revision).then_some(path.clone()))
        .collect()
}

#[derive(Clone, Copy)]
struct RasterVertex {
    position: Pos2,
    camera_depth: f32,
    world_position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

#[derive(Clone)]
struct PreparedRasterTriangle {
    object_id: NodeId,
    vertices: [RasterVertex; 3],
    base_color: [f32; 4],
    light: f32,
    face_normal: CoreVec3,
    shader: ShaderMode,
    smooth_normals: bool,
    transmission: f32,
    ior: f32,
    texture_mips: Option<Arc<Vec<TextureAsset>>>,
    texture_lod: usize,
    min_y: usize,
    max_y: usize,
}

fn rasterize_depth_frame(
    viewport_size: Vec2,
    triangles: &[PreviewTriangle],
    camera: (f32, f32, f32, f32, CoreVec3, f32, ProjectionMode),
    lighting: &ViewportLighting,
    mut linear_depth: Vec<f32>,
    workspace: &mut RasterWorkspace,
) -> DepthFrame {
    let width = viewport_size.x.round().max(1.0) as usize;
    let height = viewport_size.y.round().max(1.0) as usize;
    let size = [width, height];
    let mut color = ColorImage::new(size, Color32::TRANSPARENT);
    linear_depth.resize(width * height, f32::INFINITY);
    linear_depth.fill(f32::INFINITY);
    let live_textures = triangles
        .iter()
        .filter_map(|triangle| triangle.texture.as_ref())
        .map(|texture| Arc::as_ptr(texture) as usize)
        .collect::<HashSet<_>>();
    workspace
        .texture_mips
        .retain(|key, _| live_textures.contains(key));
    let (yaw, pitch, roll, zoom, camera_target, grid_spacing, projection_mode) = camera;
    let center = Pos2::new(width as f32 * 0.5, height as f32 * 0.5 + 25.0);
    let scale = perspective_view_scale(
        viewport_size.x.min(viewport_size.y) * 0.18 * zoom,
        projection_mode,
        grid_spacing,
    );
    let camera_position = perspective_camera_position(
        yaw,
        pitch,
        camera_target,
        PERSPECTIVE_CAMERA_DISTANCE * grid_spacing,
    );
    workspace.prepared.clear();
    workspace.clipped.clear();
    workspace.projected.clear();
    workspace.prepared.reserve(triangles.len());
    workspace.clipped.reserve(4);
    workspace.projected.reserve(4);
    for triangle in triangles {
        let world = triangle.vertices.map(|vertex| {
            CoreVec3::new(vertex.position[0], vertex.position[1], vertex.position[2])
        });
        let normal = (world[1] - world[0])
            .cross(world[2] - world[0])
            .normalized();
        let centroid = (world[0] + world[1] + world[2]) * (1.0 / 3.0);
        let band = viewport_light_factor(
            normal,
            centroid,
            triangle.shader,
            lighting,
            Some(triangle.object_id),
        );
        clip_preview_polygon_to_near_into(
            triangle,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            grid_spacing,
            &mut workspace.clipped,
        );
        workspace.projected.clear();
        workspace
            .projected
            .extend(workspace.clipped.iter().filter_map(|vertex| {
                let position = project(
                    vertex.position,
                    center,
                    scale,
                    yaw,
                    pitch,
                    roll,
                    camera_target,
                    projection_mode,
                    grid_spacing,
                )?;
                let camera_depth = PERSPECTIVE_CAMERA_DISTANCE * grid_spacing
                    + view_depth(vertex.position, yaw, pitch, roll, camera_target);
                (position.x.is_finite()
                    && position.y.is_finite()
                    && camera_depth.is_finite()
                    && camera_depth > 0.0)
                    .then_some(RasterVertex {
                        position,
                        camera_depth,
                        world_position: vertex.position,
                        normal: vertex.normal,
                        uv: vertex.uv,
                        color: vertex.color,
                    })
            }));
        for offset in 1..workspace.projected.len().saturating_sub(1) {
            let raster_vertices = [
                workspace.projected[0],
                workspace.projected[offset],
                workspace.projected[offset + 1],
            ];
            let min_y = raster_vertices
                .iter()
                .map(|vertex| vertex.position.y)
                .fold(f32::INFINITY, f32::min)
                .floor()
                .clamp(0.0, height.saturating_sub(1) as f32) as usize;
            let max_y = raster_vertices
                .iter()
                .map(|vertex| vertex.position.y)
                .fold(f32::NEG_INFINITY, f32::max)
                .ceil()
                .clamp(0.0, height.saturating_sub(1) as f32) as usize;
            let texture_mips = triangle.texture.as_ref().map(|texture| {
                let key = Arc::as_ptr(texture) as usize;
                Arc::clone(
                    &workspace
                        .texture_mips
                        .entry(key)
                        .or_insert_with(|| (Arc::clone(texture), Arc::new(texture.mip_chain(3))))
                        .1,
                )
            });
            let texture_lod = triangle.texture.as_deref().map_or(0, |texture| {
                triangle_texture_lod(raster_vertices, texture, 3)
            });
            workspace.prepared.push(PreparedRasterTriangle {
                object_id: triangle.object_id,
                vertices: raster_vertices,
                base_color: triangle.base_color,
                light: band,
                face_normal: normal,
                shader: triangle.shader,
                smooth_normals: triangle.smooth_normals,
                transmission: triangle.transmission,
                ior: triangle.ior,
                texture_mips,
                texture_lod,
                min_y,
                max_y,
            });
        }
    }
    workspace.prepared.sort_by(|left, right| {
        left.transmission
            .partial_cmp(&right.transmission)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    let workers = std::thread::available_parallelism()
        .map(usize::from)
        .unwrap_or(1)
        .min(height)
        .max(1);
    let rows_per_worker = height.div_ceil(workers);
    let pixels_per_worker = rows_per_worker * width;
    workspace.bands.resize_with(workers, Vec::new);
    for band in &mut workspace.bands {
        band.clear();
    }
    for triangle in workspace.prepared.iter().cloned() {
        let first_band = triangle.min_y / rows_per_worker;
        let last_band = (triangle.max_y / rows_per_worker).min(workers - 1);
        for band in &mut workspace.bands[first_band..=last_band] {
            band.push(triangle.clone());
        }
    }
    std::thread::scope(|scope| {
        for (band, (color_band, depth_band)) in color
            .pixels
            .chunks_mut(pixels_per_worker)
            .zip(linear_depth.chunks_mut(pixels_per_worker))
            .enumerate()
        {
            let y_start = band * rows_per_worker;
            let y_end = (y_start + color_band.len().div_ceil(width)).min(height);
            let prepared = &workspace.bands[band];
            scope.spawn(move || {
                for triangle in prepared {
                    rasterize_triangle_band(
                        triangle.vertices,
                        triangle.base_color,
                        triangle.light,
                        triangle.face_normal,
                        triangle.shader,
                        triangle.smooth_normals,
                        lighting,
                        triangle.transmission,
                        triangle.ior,
                        camera_position,
                        triangle
                            .texture_mips
                            .as_deref()
                            .and_then(|levels| levels.get(triangle.texture_lod)),
                        triangle.object_id,
                        projection_mode,
                        size,
                        y_start,
                        y_end,
                        color_band,
                        depth_band,
                    );
                }
            });
        }
    });
    DepthFrame {
        color: FrameColor::Cpu(color),
        linear_depth,
    }
}

fn triangle_texture_lod(
    vertices: [RasterVertex; 3],
    texture: &TextureAsset,
    maximum_levels: usize,
) -> usize {
    if maximum_levels <= 1 || texture.width == 0 || texture.height == 0 {
        return 0;
    }
    let mut texels_per_pixel = 0.0_f32;
    for edge in 0..3 {
        let from = vertices[edge];
        let to = vertices[(edge + 1) % 3];
        let screen_pixels = from.position.distance(to.position).max(1.0e-4);
        let texel_u = (to.uv[0] - from.uv[0]) * texture.width as f32;
        let texel_v = (to.uv[1] - from.uv[1]) * texture.height as f32;
        texels_per_pixel = texels_per_pixel.max(texel_u.hypot(texel_v) / screen_pixels);
    }
    texels_per_pixel
        .max(1.0)
        .log2()
        .floor()
        .max(0.0)
        .min(maximum_levels.saturating_sub(1) as f32) as usize
}

#[allow(clippy::too_many_arguments)]
fn rasterize_triangle_band(
    vertices: [RasterVertex; 3],
    base_color: [f32; 4],
    light: f32,
    face_normal: CoreVec3,
    shader: ShaderMode,
    smooth_normals: bool,
    lighting: &ViewportLighting,
    transmission: f32,
    ior: f32,
    camera_position: CoreVec3,
    texture: Option<&TextureAsset>,
    object_id: NodeId,
    projection_mode: ProjectionMode,
    size: [usize; 2],
    y_start: usize,
    y_end: usize,
    color: &mut [Color32],
    linear_depth: &mut [f32],
) {
    let area = edge_function(
        vertices[0].position,
        vertices[1].position,
        vertices[2].position,
    );
    if area.abs() <= 1.0e-6 {
        return;
    }
    let [width, height] = size;
    let axis_bounds = |coordinate: fn(Pos2) -> f32, upper: usize| {
        let minimum = vertices
            .iter()
            .map(|vertex| coordinate(vertex.position))
            .fold(f32::INFINITY, f32::min)
            .floor()
            .clamp(0.0, upper.saturating_sub(1) as f32) as usize;
        let maximum = vertices
            .iter()
            .map(|vertex| coordinate(vertex.position))
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil()
            .clamp(0.0, upper.saturating_sub(1) as f32) as usize;
        (minimum, maximum)
    };
    let (min_x, max_x) = axis_bounds(|point| point.x, width);
    let (mut min_y, mut max_y) = axis_bounds(|point| point.y, height);
    min_y = min_y.max(y_start);
    max_y = max_y.min(y_end.saturating_sub(1));
    if min_y > max_y {
        return;
    }
    let inverse_area = area.recip();
    let edges = [
        (vertices[1].position, vertices[2].position),
        (vertices[2].position, vertices[0].position),
        (vertices[0].position, vertices[1].position),
    ];
    let edge_step_x = edges.map(|(start, end)| end.y - start.y);
    let edge_step_y = edges.map(|(start, end)| -(end.x - start.x));
    let first_pixel = Pos2::new(min_x as f32 + 0.5, min_y as f32 + 0.5);
    let first_edges = edges.map(|(start, end)| edge_function(start, end, first_pixel));
    let depth_coefficients = match projection_mode {
        ProjectionMode::Perspective => vertices.map(|vertex| vertex.camera_depth.recip()),
        ProjectionMode::Orthographic => vertices.map(|vertex| vertex.camera_depth),
    };
    let first_depth_value = first_edges
        .iter()
        .zip(depth_coefficients)
        .map(|(edge, coefficient)| edge * inverse_area * coefficient)
        .sum::<f32>();
    let depth_step_x = edge_step_x
        .iter()
        .zip(depth_coefficients)
        .map(|(edge, coefficient)| edge * inverse_area * coefficient)
        .sum::<f32>();
    let depth_step_y = edge_step_y
        .iter()
        .zip(depth_coefficients)
        .map(|(edge, coefficient)| edge * inverse_area * coefficient)
        .sum::<f32>();
    let mut row_edges = first_edges;
    let mut row_depth_value = first_depth_value;
    for y in min_y..=max_y {
        let mut pixel_edges = row_edges;
        let mut depth_value = row_depth_value;
        for x in min_x..=max_x {
            if pixel_edges
                .iter()
                .all(|edge| edge * inverse_area >= -1.0e-5)
            {
                let depth = match projection_mode {
                    ProjectionMode::Perspective if depth_value > f32::EPSILON => {
                        depth_value.recip()
                    }
                    ProjectionMode::Perspective => f32::INFINITY,
                    ProjectionMode::Orthographic => depth_value,
                };
                let pixel = (y - y_start) * width + x;
                if depth < linear_depth[pixel] {
                    let weights = pixel_edges.map(|edge| edge * inverse_area);
                    let interpolate = |component: fn(&RasterVertex) -> f32| match projection_mode {
                        ProjectionMode::Perspective => {
                            weights
                                .iter()
                                .zip(vertices)
                                .map(|(weight, vertex)| {
                                    weight * component(&vertex) / vertex.camera_depth
                                })
                                .sum::<f32>()
                                / depth_value
                        }
                        ProjectionMode::Orthographic => weights
                            .iter()
                            .zip(vertices)
                            .map(|(weight, vertex)| weight * component(&vertex))
                            .sum(),
                    };
                    let uv = [
                        interpolate(|vertex| vertex.uv[0]),
                        interpolate(|vertex| vertex.uv[1]),
                    ];
                    let vertex_color = [
                        interpolate(|vertex| vertex.color[0]),
                        interpolate(|vertex| vertex.color[1]),
                        interpolate(|vertex| vertex.color[2]),
                        interpolate(|vertex| vertex.color[3]),
                    ];
                    let shading_normal = if smooth_normals {
                        CoreVec3::new(
                            interpolate(|vertex| vertex.normal[0]),
                            interpolate(|vertex| vertex.normal[1]),
                            interpolate(|vertex| vertex.normal[2]),
                        )
                        .normalized()
                    } else {
                        face_normal
                    };
                    let world_position = CoreVec3::new(
                        interpolate(|vertex| vertex.world_position[0]),
                        interpolate(|vertex| vertex.world_position[1]),
                        interpolate(|vertex| vertex.world_position[2]),
                    );
                    let pixel_light = if smooth_normals {
                        viewport_light_factor(
                            shading_normal,
                            world_position,
                            shader,
                            lighting,
                            Some(object_id),
                        )
                    } else {
                        light
                    };
                    let texel = texture
                        .map(|texture| sample_texture_nearest(texture, uv))
                        .unwrap_or([1.0; 4]);
                    let rgba = [0, 1, 2, 3].map(|channel| {
                        let lighting = if channel == 3 { 1.0 } else { pixel_light };
                        (base_color[channel] * vertex_color[channel] * texel[channel] * lighting)
                            .clamp(0.0, 1.0)
                    });
                    let surface = Color32::from_rgba_unmultiplied(
                        (rgba[0] * 255.0).round() as u8,
                        (rgba[1] * 255.0).round() as u8,
                        (rgba[2] * 255.0).round() as u8,
                        (rgba[3] * 255.0).round() as u8,
                    );
                    if transmission > 0.0 && color[pixel] != Color32::TRANSPARENT {
                        let view_direction = (camera_position - world_position).normalized();
                        let fresnel =
                            schlick_fresnel(shading_normal.dot(view_direction).abs(), ior);
                        let background_share = (transmission * (1.0 - fresnel)).clamp(0.0, 1.0);
                        color[pixel] =
                            blend_preview_surface(color[pixel], surface, background_share);
                    } else {
                        color[pixel] = surface;
                    }
                    if transmission <= 0.0 {
                        linear_depth[pixel] = depth;
                    }
                }
            }
            for edge in 0..3 {
                pixel_edges[edge] += edge_step_x[edge];
            }
            depth_value += depth_step_x;
        }
        for edge in 0..3 {
            row_edges[edge] += edge_step_y[edge];
        }
        row_depth_value += depth_step_y;
    }
}

fn sample_texture_nearest(texture: &TextureAsset, uv: [f32; 2]) -> [f32; 4] {
    if texture.width == 0 || texture.height == 0 || texture.pixels.len() < 4 {
        return [1.0; 4];
    }
    let u = uv[0].rem_euclid(1.0);
    let v = uv[1].rem_euclid(1.0);
    let x = (u * texture.width as f32).floor() as usize % texture.width as usize;
    let y = (v * texture.height as f32).floor() as usize % texture.height as usize;
    let offset = (y * texture.width as usize + x) * 4;
    if offset + 3 >= texture.pixels.len() {
        return [1.0; 4];
    }
    [
        texture.pixels[offset] as f32 / 255.0,
        texture.pixels[offset + 1] as f32 / 255.0,
        texture.pixels[offset + 2] as f32 / 255.0,
        texture.pixels[offset + 3] as f32 / 255.0,
    ]
}

fn blend_preview_surface(background: Color32, surface: Color32, background_share: f32) -> Color32 {
    let mix = |behind: u8, front: u8| {
        (f32::from(front) * (1.0 - background_share) + f32::from(behind) * background_share).round()
            as u8
    };
    Color32::from_rgba_unmultiplied(
        mix(background.r(), surface.r()),
        mix(background.g(), surface.g()),
        mix(background.b(), surface.b()),
        surface.a(),
    )
}

fn schlick_fresnel(normal_view_cosine: f32, ior: f32) -> f32 {
    let safe_ior = ior.clamp(1.0, 2.5);
    let f0 = ((safe_ior - 1.0) / (safe_ior + 1.0)).powi(2);
    f0 + (1.0 - f0) * (1.0 - normal_view_cosine.clamp(0.0, 1.0)).powi(5)
}

fn perspective_camera_position(yaw: f32, pitch: f32, target: CoreVec3, distance: f32) -> CoreVec3 {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let forward = CoreVec3::new(sin_yaw * cos_pitch, cos_yaw * cos_pitch, sin_pitch);
    target - forward * distance
}

fn edge_function(start: Pos2, end: Pos2, point: Pos2) -> f32 {
    (point.x - start.x) * (end.y - start.y) - (point.y - start.y) * (end.x - start.x)
}

struct RaycastTriangle {
    points: [CoreVec3; 3],
    color: Color32,
    centroid_depth: f32,
    screen_bounds: Option<ScreenBounds>,
    screen_polygon: Vec<Pos2>,
}

#[derive(Clone, Copy)]
struct ScreenBounds {
    min: Pos2,
    max: Pos2,
}

impl ScreenBounds {
    fn contains(self, point: Pos2) -> bool {
        point.x >= self.min.x
            && point.x <= self.max.x
            && point.y >= self.min.y
            && point.y <= self.max.y
    }
}

#[allow(dead_code)]
fn raycast_depth_frame(
    viewport_size: Vec2,
    triangles: &[[[f32; 3]; 3]],
    camera: (f32, f32, f32, f32, CoreVec3, f32, ProjectionMode),
) -> DepthFrame {
    const TILE_SIZE: usize = 16;
    let width = viewport_size.x.round().max(1.0) as usize;
    let height = viewport_size.y.round().max(1.0) as usize;
    let size = [width, height];
    let mut color = ColorImage::new(size, Color32::TRANSPARENT);
    let mut linear_depth = vec![f32::INFINITY; width * height];
    if triangles.is_empty() {
        return DepthFrame {
            color: FrameColor::Cpu(color),
            linear_depth,
        };
    }

    let (yaw, pitch, roll, zoom, camera_target, grid_spacing, projection_mode) = camera;
    let center = Pos2::new(width as f32 * 0.5, height as f32 * 0.5 + 25.0);
    let scale = perspective_view_scale(
        viewport_size.x.min(viewport_size.y) * 0.18 * zoom,
        projection_mode,
        grid_spacing,
    );
    let light = global_light_direction();
    let mut raster_triangles = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        let points = triangle.map(|point| CoreVec3::new(point[0], point[1], point[2]));
        let normal = (points[1] - points[0])
            .cross(points[2] - points[0])
            .normalized();
        let diffuse = normal.dot(light).max(0.0);
        let band = ((0.25 + diffuse * 0.75) * 3.0).round() / 3.0;
        raster_triangles.push(RaycastTriangle {
            points,
            color: Color32::from_rgb(
                (61.0 + 72.0 * band) as u8,
                (92.0 + 96.0 * band) as u8,
                (118.0 + 103.0 * band) as u8,
            ),
            centroid_depth: triangle
                .iter()
                .map(|point| view_depth(*point, yaw, pitch, roll, camera_target))
                .sum::<f32>()
                / 3.0,
            screen_bounds: None,
            screen_polygon: Vec::new(),
        });
    }

    let mut order: Vec<usize> = (0..raster_triangles.len()).collect();
    order.sort_unstable_by(|left, right| {
        raster_triangles[*left]
            .centroid_depth
            .total_cmp(&raster_triangles[*right].centroid_depth)
    });
    let tile_columns = width.div_ceil(TILE_SIZE);
    let tile_rows = height.div_ceil(TILE_SIZE);
    let mut tiles = vec![Vec::<usize>::new(); tile_columns * tile_rows];
    for index in order {
        let source = triangles[index];
        let clipped = clip_polygon_to_near(
            &source,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            grid_spacing,
        );
        let projected = clipped
            .iter()
            .filter_map(|point| {
                project(
                    *point,
                    center,
                    scale,
                    yaw,
                    pitch,
                    roll,
                    camera_target,
                    projection_mode,
                    grid_spacing,
                )
            })
            .filter(|point| point.x.is_finite() && point.y.is_finite())
            .collect::<Vec<_>>();
        if projected.len() < 3 {
            continue;
        }
        let projected_min_x = projected
            .iter()
            .map(|point| point.x)
            .fold(f32::INFINITY, f32::min);
        let projected_max_x = projected
            .iter()
            .map(|point| point.x)
            .fold(f32::NEG_INFINITY, f32::max);
        let projected_min_y = projected
            .iter()
            .map(|point| point.y)
            .fold(f32::INFINITY, f32::min);
        let projected_max_y = projected
            .iter()
            .map(|point| point.y)
            .fold(f32::NEG_INFINITY, f32::max);
        raster_triangles[index].screen_bounds = Some(ScreenBounds {
            min: Pos2::new(projected_min_x, projected_min_y),
            max: Pos2::new(projected_max_x, projected_max_y),
        });
        raster_triangles[index].screen_polygon = projected;
        let min_x = projected_min_x
            .floor()
            .clamp(0.0, width.saturating_sub(1) as f32) as usize;
        let max_x = projected_max_x
            .ceil()
            .clamp(0.0, width.saturating_sub(1) as f32) as usize;
        let min_y = projected_min_y
            .floor()
            .clamp(0.0, height.saturating_sub(1) as f32) as usize;
        let max_y = projected_max_y
            .ceil()
            .clamp(0.0, height.saturating_sub(1) as f32) as usize;
        if min_x > max_x || min_y > max_y {
            continue;
        }
        for tile_y in min_y / TILE_SIZE..=max_y / TILE_SIZE {
            for tile_x in min_x / TILE_SIZE..=max_x / TILE_SIZE {
                tiles[tile_y * tile_columns + tile_x].push(index);
            }
        }
    }

    let (right, up, forward) = camera_basis(yaw, pitch, roll);
    let camera_distance = PERSPECTIVE_CAMERA_DISTANCE * grid_spacing;
    let camera_origin = camera_target - forward * camera_distance;
    let near = PERSPECTIVE_NEAR * grid_spacing;
    for tile_y in 0..tile_rows {
        for tile_x in 0..tile_columns {
            let candidates = &tiles[tile_y * tile_columns + tile_x];
            if candidates.is_empty() {
                continue;
            }
            let x_end = ((tile_x + 1) * TILE_SIZE).min(width);
            let y_end = ((tile_y + 1) * TILE_SIZE).min(height);
            for y in tile_y * TILE_SIZE..y_end {
                for x in tile_x * TILE_SIZE..x_end {
                    let screen_x = (x as f32 + 0.5 - center.x) / scale;
                    let screen_y = -(y as f32 + 0.5 - center.y) / scale;
                    let (origin, direction) = match projection_mode {
                        ProjectionMode::Perspective => (
                            camera_origin,
                            (forward
                                + right * (screen_x / camera_distance)
                                + up * (screen_y / camera_distance))
                                .normalized(),
                        ),
                        ProjectionMode::Orthographic => (
                            camera_target + right * screen_x + up * screen_y
                                - forward * camera_distance,
                            forward,
                        ),
                    };
                    let pixel = y * width + x;
                    let screen_point = Pos2::new(x as f32 + 0.5, y as f32 + 0.5);
                    let mut covering_count = 0usize;
                    let mut sole_candidate = 0usize;
                    for candidate in candidates {
                        let triangle = &raster_triangles[*candidate];
                        if triangle
                            .screen_bounds
                            .is_some_and(|bounds| bounds.contains(screen_point))
                            && point_in_convex_polygon(screen_point, &triangle.screen_polygon)
                        {
                            covering_count += 1;
                            sole_candidate = *candidate;
                        }
                    }
                    if covering_count == 1 {
                        let triangle = &raster_triangles[sole_candidate];
                        if let Some(distance) =
                            ray_plane_distance(origin, direction, triangle.points)
                            && distance >= near
                        {
                            linear_depth[pixel] = distance;
                            color.pixels[pixel] = triangle.color;
                        }
                    } else if covering_count > 1 {
                        for candidate in candidates {
                            let triangle = &raster_triangles[*candidate];
                            if !triangle
                                .screen_bounds
                                .is_some_and(|bounds| bounds.contains(screen_point))
                                || !point_in_convex_polygon(screen_point, &triangle.screen_polygon)
                            {
                                continue;
                            }
                            let Some(distance) =
                                ray_triangle_distance(origin, direction, triangle.points)
                            else {
                                continue;
                            };
                            if distance >= near && distance < linear_depth[pixel] {
                                linear_depth[pixel] = distance;
                                color.pixels[pixel] = triangle.color;
                            }
                        }
                    }
                }
            }
        }
    }
    DepthFrame {
        color: FrameColor::Cpu(color),
        linear_depth,
    }
}

fn camera_basis(yaw: f32, pitch: f32, roll: f32) -> (CoreVec3, CoreVec3, CoreVec3) {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let base_right = CoreVec3::new(cos_yaw, -sin_yaw, 0.0);
    let base_up = CoreVec3::new(-sin_yaw * sin_pitch, -cos_yaw * sin_pitch, cos_pitch);
    let (sin_roll, cos_roll) = roll.sin_cos();
    (
        base_right * cos_roll - base_up * sin_roll,
        base_right * sin_roll + base_up * cos_roll,
        CoreVec3::new(sin_yaw * cos_pitch, cos_yaw * cos_pitch, sin_pitch),
    )
}

fn point_in_convex_polygon(point: Pos2, polygon: &[Pos2]) -> bool {
    if polygon.len() < 3 {
        return false;
    }
    let mut orientation = 0.0f32;
    for index in 0..polygon.len() {
        let start = polygon[index];
        let end = polygon[(index + 1) % polygon.len()];
        let edge = end - start;
        let offset = point - start;
        let cross = edge.x * offset.y - edge.y * offset.x;
        if cross.abs() <= 1.0e-4 {
            continue;
        }
        if orientation == 0.0 {
            orientation = cross.signum();
        } else if orientation * cross < 0.0 {
            return false;
        }
    }
    true
}

fn ray_plane_distance(
    origin: CoreVec3,
    direction: CoreVec3,
    triangle: [CoreVec3; 3],
) -> Option<f32> {
    let normal = (triangle[1] - triangle[0]).cross(triangle[2] - triangle[0]);
    let denominator = normal.dot(direction);
    if denominator.abs() <= 1.0e-7 {
        return None;
    }
    let distance = normal.dot(triangle[0] - origin) / denominator;
    (distance > 0.0 && distance.is_finite()).then_some(distance)
}

fn ray_triangle_distance(
    origin: CoreVec3,
    direction: CoreVec3,
    triangle: [CoreVec3; 3],
) -> Option<f32> {
    let edge_ab = triangle[1] - triangle[0];
    let edge_ac = triangle[2] - triangle[0];
    let cross = direction.cross(edge_ac);
    let determinant = edge_ab.dot(cross);
    if determinant.abs() <= 1.0e-7 {
        return None;
    }
    let inverse = determinant.recip();
    let origin_offset = origin - triangle[0];
    let barycentric_b = origin_offset.dot(cross) * inverse;
    if !(0.0..=1.0).contains(&barycentric_b) {
        return None;
    }
    let barycentric_cross = origin_offset.cross(edge_ab);
    let barycentric_c = direction.dot(barycentric_cross) * inverse;
    if barycentric_c < 0.0 || barycentric_b + barycentric_c > 1.0 {
        return None;
    }
    let distance = edge_ac.dot(barycentric_cross) * inverse;
    (distance > 0.0 && distance.is_finite()).then_some(distance)
}

fn ray_triangle_hit(
    origin: CoreVec3,
    direction: CoreVec3,
    triangle: [CoreVec3; 3],
) -> Option<(f32, [f32; 3])> {
    let edge_ab = triangle[1] - triangle[0];
    let edge_ac = triangle[2] - triangle[0];
    let cross = direction.cross(edge_ac);
    let determinant = edge_ab.dot(cross);
    if determinant.abs() <= 1.0e-7 {
        return None;
    }
    let inverse = determinant.recip();
    let origin_offset = origin - triangle[0];
    let weight_b = origin_offset.dot(cross) * inverse;
    if !(0.0..=1.0).contains(&weight_b) {
        return None;
    }
    let barycentric_cross = origin_offset.cross(edge_ab);
    let weight_c = direction.dot(barycentric_cross) * inverse;
    if weight_c < 0.0 || weight_b + weight_c > 1.0 {
        return None;
    }
    let distance = edge_ac.dot(barycentric_cross) * inverse;
    let weight_a = 1.0 - weight_b - weight_c;
    (distance > 0.0
        && distance.is_finite()
        && [weight_a, weight_b, weight_c]
            .iter()
            .all(|weight| weight.is_finite()))
    .then_some((distance, [weight_a, weight_b, weight_c]))
}

#[allow(clippy::too_many_arguments)]
fn fit_aspect_rect(outer: Rect, aspect: f32) -> Rect {
    let aspect = aspect.clamp(0.1, 10.0);
    let outer_aspect = outer.width() / outer.height().max(1.0);
    let size = if outer_aspect > aspect {
        Vec2::new(outer.height() * aspect, outer.height())
    } else {
        Vec2::new(outer.width(), outer.width() / aspect)
    };
    Rect::from_center_size(outer.center(), size)
}

fn scene_camera_render_view(
    camera: &SceneCameraPreview,
    viewport_size: Vec2,
    grid_spacing: f32,
) -> (f32, f32, f32, f32, CoreVec3, f32, ProjectionMode) {
    let forward = camera.transform.rotation.rotate(CoreVec3::Y).normalized();
    let yaw = forward.x.atan2(forward.y);
    let pitch = forward.z.clamp(-1.0, 1.0).asin();
    let (base_right, base_up, _) = camera_basis(yaw, pitch, 0.0);
    let actual_right = camera.transform.rotation.rotate(CoreVec3::X).normalized();
    let roll = (-actual_right.dot(base_up)).atan2(actual_right.dot(base_right));
    let grid_spacing = grid_spacing.max(1.0e-4);
    let distance = PERSPECTIVE_CAMERA_DISTANCE * grid_spacing;
    let target = camera.transform.translation + forward * distance;
    let minimum_extent = viewport_size.x.min(viewport_size.y).max(1.0);
    let (zoom, projection) = if camera.perspective {
        let tan_half = (camera.field_of_view_degrees.clamp(1.0, 179.0).to_radians() * 0.5)
            .tan()
            .max(1.0e-4);
        (
            viewport_size.y.max(1.0) / (7.2 * tan_half * minimum_extent),
            ProjectionMode::Perspective,
        )
    } else {
        (
            viewport_size.y.max(1.0)
                / (camera.orthographic_size.max(0.001) * minimum_extent * 0.18),
            ProjectionMode::Orthographic,
        )
    };
    (yaw, pitch, roll, zoom, target, grid_spacing, projection)
}

fn draw_viewport(
    painter: &egui::Painter,
    rect: Rect,
    grid: bool,
    tool: Tool,
    camera: (f32, f32, f32, f32, CoreVec3, f32, ProjectionMode),
    triangles: &[PreviewTriangle],
    mode: ViewportMode,
    viewport_texture: Option<TextureId>,
    collider_wireframe: &[[[f32; 3]; 2]],
    camera_wireframes: &[[[f32; 3]; 2]],
    point_lights: &[ViewportLight],
) {
    let (yaw, pitch, roll, zoom, camera_target, grid_spacing, projection_mode) = camera;
    painter.rect_filled(rect, 0.0, Color32::from_rgb(21, 24, 31));
    let center = rect.center() + Vec2::new(0.0, 25.0);
    let scale = perspective_view_scale(
        rect.width().min(rect.height()) * 0.18 * zoom,
        projection_mode,
        grid_spacing,
    );

    if grid {
        let grid_center_x = (camera_target.x / grid_spacing).round() * grid_spacing;
        let grid_center_y = (camera_target.y / grid_spacing).round() * grid_spacing;
        let origin = [grid_center_x, grid_center_y, 0.0];
        let projected_origin = project(
            origin,
            center,
            scale,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            grid_spacing,
        );
        let projected_x = project(
            [grid_center_x + grid_spacing, grid_center_y, 0.0],
            center,
            scale,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            grid_spacing,
        );
        let projected_y = project(
            [grid_center_x, grid_center_y + grid_spacing, 0.0],
            center,
            scale,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            grid_spacing,
        );
        let half_cells = match (projected_origin, projected_x, projected_y) {
            (Some(origin), Some(x), Some(y)) => {
                visible_grid_half_cells(rect, x - origin, y - origin)
            }
            _ => 200,
        };
        let radius = half_cells as f32 * grid_spacing;
        let alpha = (110.0 * grid_distance_alpha(zoom, grid_spacing)) as u8;
        let color = Color32::from_rgba_unmultiplied(110, 110, 110, alpha);
        let project_grid_line = |segment| match projection_mode {
            ProjectionMode::Orthographic => project_segment(
                segment,
                center,
                scale,
                yaw,
                pitch,
                roll,
                camera_target,
                projection_mode,
                grid_spacing,
            ),
            ProjectionMode::Perspective => projective_ground_line(
                canonical_ground_line(segment),
                rect,
                center,
                scale,
                yaw,
                pitch,
                camera_target,
                grid_spacing,
            ),
        };
        for index in -half_cells..=half_cells {
            let offset = index as f32 * grid_spacing;
            for segment in [
                [
                    [grid_center_x + offset, grid_center_y - radius, 0.0],
                    [grid_center_x + offset, grid_center_y + radius, 0.0],
                ],
                [
                    [grid_center_x - radius, grid_center_y + offset, 0.0],
                    [grid_center_x + radius, grid_center_y + offset, 0.0],
                ],
            ] {
                if let Some(projected) = project_grid_line(segment) {
                    painter.line_segment(projected, Stroke::new(0.6, color));
                }
            }
        }
        for (axis, color) in [
            (
                [[-1.0, 0.0, 0.0], [1.0, 0.0, 0.0]],
                Color32::from_rgb(235, 91, 91),
            ),
            (
                [[0.0, -1.0, 0.0], [0.0, 1.0, 0.0]],
                Color32::from_rgb(96, 205, 125),
            ),
            (
                [[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]],
                Color32::from_rgb(91, 145, 235),
            ),
        ] {
            let projected = if projection_mode == ProjectionMode::Perspective {
                projective_ground_line(
                    axis,
                    rect,
                    center,
                    scale,
                    yaw,
                    pitch,
                    camera_target,
                    grid_spacing,
                )
            } else {
                analytical_projected_line(
                    axis,
                    rect,
                    center,
                    scale,
                    (yaw, pitch, camera_target, projection_mode, grid_spacing),
                )
            };
            if let Some(projected) = projected {
                painter.line_segment(projected, Stroke::new(1.8, color));
            }
        }
    }

    if !triangles.is_empty() {
        if let Some(texture) = viewport_texture {
            painter.image(
                texture,
                rect,
                Rect::from_min_max(Pos2::ZERO, Pos2::new(1.0, 1.0)),
                Color32::WHITE,
            );
        }

        if mode != ViewportMode::Shaded {
            let edge_color = if mode == ViewportMode::Wireframe {
                Color32::from_rgb(112, 180, 226)
            } else {
                Color32::from_rgb(28, 43, 54)
            };
            for triangle in triangles {
                for (a, b) in [(0, 1), (1, 2), (2, 0)] {
                    if let Some(projected) = project_segment(
                        [triangle.vertices[a].position, triangle.vertices[b].position],
                        center,
                        scale,
                        yaw,
                        pitch,
                        roll,
                        camera_target,
                        projection_mode,
                        grid_spacing,
                    ) {
                        painter.line_segment(projected, Stroke::new(0.7, edge_color));
                    }
                }
            }
        }

        let pivot = project(
            [0.0, 0.0, 0.7],
            center,
            scale,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            grid_spacing,
        )
        .unwrap_or(center);
        if tool == Tool::Rotate {
            painter.circle_stroke(
                pivot,
                38.0,
                Stroke::new(3.0, Color32::from_rgb(96, 205, 125)),
            );
            painter.circle_stroke(
                pivot,
                29.0,
                Stroke::new(2.0, Color32::from_rgb(91, 145, 235)),
            );
        } else {
            for (offset, color, label) in [
                (Vec2::new(62.0, 0.0), Color32::from_rgb(235, 91, 91), "X"),
                (Vec2::new(-38.0, 35.0), Color32::from_rgb(96, 205, 125), "Y"),
                (Vec2::new(0.0, -62.0), Color32::from_rgb(91, 145, 235), "Z"),
            ] {
                painter.arrow(pivot, offset, Stroke::new(3.0, color));
                painter.text(
                    pivot + offset,
                    Align2::CENTER_CENTER,
                    label,
                    FontId::proportional(12.0),
                    color,
                );
            }
        }
    }
    let collider_stroke = Stroke::new(2.0, Color32::from_rgb(255, 210, 48));
    for &segment in collider_wireframe {
        if let Some(projected) = project_segment(
            segment,
            center,
            scale,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            grid_spacing,
        ) {
            painter.line_segment(projected, collider_stroke);
        }
    }
    let camera_stroke = Stroke::new(2.0, Color32::from_rgb(255, 220, 64));
    for &segment in camera_wireframes {
        if let Some(projected) = project_segment(
            segment,
            center,
            scale,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            grid_spacing,
        ) {
            painter.line_segment(projected, camera_stroke);
        }
    }
    for light in point_lights {
        if let Some(position) = project(
            [light.position.x, light.position.y, light.position.z],
            center,
            scale,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            grid_spacing,
        ) {
            let yellow = Color32::from_rgb(255, 210, 48);
            painter.circle_filled(position, 4.0, yellow);
            painter.circle_stroke(position, 8.0, Stroke::new(2.0, yellow));
        }
    }
    painter.text(
        rect.left_top() + Vec2::new(12.0, 14.0),
        Align2::LEFT_TOP,
        "Scene 01",
        FontId::proportional(13.0),
        Color32::from_gray(175),
    );
}

fn format_grid_spacing(spacing: f32) -> String {
    if spacing < 1.0 {
        format!("{spacing} units")
    } else {
        format!("{spacing:.0} units")
    }
}

fn shader_light_factor(diffuse: f32, shader: ShaderMode) -> f32 {
    let continuous = 0.25 + diffuse.clamp(0.0, 1.0) * 0.75;
    match shader {
        ShaderMode::Toon => (continuous * 3.0).round() / 3.0,
        ShaderMode::Diffuse => continuous,
    }
}

fn safe_normal_reciprocal(scale: f32) -> f32 {
    if scale.abs() <= 1.0e-6 {
        0.0
    } else {
        scale.recip()
    }
}

fn transform_normal(normal: CoreVec3, scale: CoreVec3, rotation: Quat) -> CoreVec3 {
    let reciprocal_scale = CoreVec3::new(
        safe_normal_reciprocal(scale.x),
        safe_normal_reciprocal(scale.y),
        safe_normal_reciprocal(scale.z),
    );
    rotation
        .rotate(reciprocal_scale.component_mul(normal))
        .normalized()
}

/// Editor directional light in world coordinates. Translation and camera
/// orientation must never participate in this value.
fn global_light_direction() -> CoreVec3 {
    CoreVec3::new(-0.35, 0.8, 0.45).normalized()
}

fn intensity_to_exponent(intensity: f32) -> f32 {
    if intensity.is_finite() && intensity > 0.0 {
        intensity.log10()
    } else if intensity == f32::INFINITY {
        f32::MAX.log10()
    } else {
        f32::MIN_POSITIVE.log10()
    }
}

fn light_intensity_from_exponent(exponent: f32) -> f32 {
    if exponent.is_nan() {
        return 1.0;
    }
    let intensity = 10.0_f32.powf(exponent);
    if intensity.is_infinite() {
        f32::MAX
    } else {
        intensity.max(0.0)
    }
}

fn sanitize_shadow_resolution(value: u32) -> u32 {
    match value {
        0 => 0,
        1..=191 => 128,
        192..=383 => 256,
        384..=767 => 512,
        768..=1535 => 1024,
        _ => 2048,
    }
}

fn sanitize_target_fps(value: u32) -> u32 {
    value.clamp(15, 360)
}

fn target_frame_period(target_fps: u32) -> Duration {
    Duration::from_secs_f64(1.0 / sanitize_target_fps(target_fps) as f64)
}

fn advance_frame_deadline(deadline: Instant, now: Instant, target_fps: u32) -> Instant {
    let period = target_frame_period(target_fps);
    if deadline > now {
        return deadline;
    }
    let elapsed = now.saturating_duration_since(deadline);
    let missed_periods = (elapsed.as_secs_f64() / period.as_secs_f64()).floor() as u32;
    deadline + period * missed_periods.saturating_add(1)
}

fn shadow_quality_label(quality: usize) -> &'static str {
    ["Potato", "Low", "Medium", "High", "Ultra"][quality.min(4)]
}

fn effective_shadow_resolution(requested: u32, quality: usize) -> u32 {
    if requested == 0 {
        return 0;
    }
    let divisor = [4, 2, 4, 1, 1][quality.min(4)];
    let numerator = [1, 1, 3, 1, 1][quality.min(4)];
    (requested.saturating_mul(numerator) / divisor).max(32)
}

fn shadow_filter_radius(quality: usize) -> usize {
    [2, 2, 1, 1, 2][quality.min(4)]
}

fn build_directional_shadow_map(
    triangles: &[PreviewTriangle],
    resolution: usize,
    filter_radius: usize,
) -> DirectionalShadowMap {
    build_directional_shadow_map_with_blur(triangles, resolution, filter_radius, 0)
}

fn build_directional_shadow_map_with_blur(
    triangles: &[PreviewTriangle],
    resolution: usize,
    filter_radius: usize,
    blur_radius: usize,
) -> DirectionalShadowMap {
    let resolution = resolution.clamp(1, 2048);
    let direction = global_light_direction();
    let forward = direction * -1.0;
    let reference_up = if forward.z.abs() < 0.95 {
        CoreVec3::Z
    } else {
        CoreVec3::Y
    };
    let right = reference_up.cross(forward).normalized();
    let up = forward.cross(right).normalized();
    let mut minimum = CoreVec3::new(f32::INFINITY, f32::INFINITY, f32::INFINITY);
    let mut maximum = CoreVec3::new(f32::NEG_INFINITY, f32::NEG_INFINITY, f32::NEG_INFINITY);
    for vertex in triangles.iter().flat_map(|triangle| triangle.vertices) {
        let point = CoreVec3::new(vertex.position[0], vertex.position[1], vertex.position[2]);
        minimum.x = minimum.x.min(point.x);
        minimum.y = minimum.y.min(point.y);
        minimum.z = minimum.z.min(point.z);
        maximum.x = maximum.x.max(point.x);
        maximum.y = maximum.y.max(point.y);
        maximum.z = maximum.z.max(point.z);
    }
    if !minimum.x.is_finite() {
        return DirectionalShadowMap {
            resolution,
            depth: vec![f32::INFINITY; resolution * resolution],
            origin: CoreVec3::ZERO,
            right,
            up,
            forward,
            extent: 1.0,
            bias: 1.0e-3,
            filter_radius,
        };
    }
    let origin = (minimum + maximum) * 0.5;
    let mut extent = 0.0_f32;
    for triangle in triangles {
        for vertex in triangle.vertices {
            let point =
                CoreVec3::new(vertex.position[0], vertex.position[1], vertex.position[2]) - origin;
            extent = extent.max(point.dot(right).abs()).max(point.dot(up).abs());
        }
    }
    extent = (extent * 1.05).max(1.0e-3);
    let mut map = DirectionalShadowMap {
        resolution,
        depth: vec![f32::INFINITY; resolution * resolution],
        origin,
        right,
        up,
        forward,
        extent,
        bias: (extent * 2.0 / resolution as f32).max(1.0e-5) * 1.5,
        filter_radius,
    };
    for triangle in triangles.iter().filter(|triangle| triangle.casts_shadows) {
        let projected = triangle.vertices.map(|vertex| {
            let point =
                CoreVec3::new(vertex.position[0], vertex.position[1], vertex.position[2]) - origin;
            let normalized_x = point.dot(right) / (extent * 2.0) + 0.5;
            let normalized_y = 0.5 - point.dot(up) / (extent * 2.0);
            (
                Pos2::new(
                    normalized_x * resolution as f32,
                    normalized_y * resolution as f32,
                ),
                point.dot(forward),
            )
        });
        rasterize_shadow_triangle(&mut map.depth, resolution, projected);
    }
    fast_blur_shadow_region(
        &mut map.depth,
        resolution,
        resolution,
        0,
        0,
        resolution,
        resolution,
        blur_radius,
    );
    map
}

fn fast_blur_shadow_region(
    depth: &mut [f32],
    stride: usize,
    full_height: usize,
    origin_x: usize,
    origin_y: usize,
    width: usize,
    height: usize,
    radius: usize,
) {
    let radius = radius.min(4);
    if radius == 0 || width == 0 || height == 0 || depth.len() < stride * full_height {
        return;
    }
    let mut scratch = vec![f32::INFINITY; width * height];
    for y in origin_y..origin_y + height {
        for x in origin_x..origin_x + width {
            let center = depth[y * stride + x];
            if !center.is_finite() {
                continue;
            }
            let mut total = 0.0;
            let mut samples = 0;
            for sample_x in
                x.saturating_sub(radius).max(origin_x)..=(x + radius).min(origin_x + width - 1)
            {
                let sample = depth[y * stride + sample_x];
                if sample.is_finite() {
                    total += sample;
                    samples += 1;
                }
            }
            scratch[(y - origin_y) * width + x - origin_x] = total / samples.max(1) as f32;
        }
    }
    for y in origin_y..origin_y + height {
        for x in origin_x..origin_x + width {
            let center = scratch[(y - origin_y) * width + x - origin_x];
            if !center.is_finite() {
                continue;
            }
            let mut total = 0.0;
            let mut samples = 0;
            for sample_y in
                y.saturating_sub(radius).max(origin_y)..=(y + radius).min(origin_y + height - 1)
            {
                let sample = scratch[(sample_y - origin_y) * width + x - origin_x];
                if sample.is_finite() {
                    total += sample;
                    samples += 1;
                }
            }
            depth[y * stride + x] = total / samples.max(1) as f32;
        }
    }
}

fn rasterize_shadow_triangle(depth: &mut [f32], resolution: usize, vertices: [(Pos2, f32); 3]) {
    let area = edge_function(vertices[0].0, vertices[1].0, vertices[2].0);
    if area.abs() <= 1.0e-8 {
        return;
    }
    let min_x = vertices
        .iter()
        .map(|vertex| vertex.0.x)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, resolution.saturating_sub(1) as f32) as usize;
    let max_x = vertices
        .iter()
        .map(|vertex| vertex.0.x)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, resolution.saturating_sub(1) as f32) as usize;
    let min_y = vertices
        .iter()
        .map(|vertex| vertex.0.y)
        .fold(f32::INFINITY, f32::min)
        .floor()
        .clamp(0.0, resolution.saturating_sub(1) as f32) as usize;
    let max_y = vertices
        .iter()
        .map(|vertex| vertex.0.y)
        .fold(f32::NEG_INFINITY, f32::max)
        .ceil()
        .clamp(0.0, resolution.saturating_sub(1) as f32) as usize;
    let inverse_area = area.recip();
    for y in min_y..=max_y {
        for x in min_x..=max_x {
            let sample = Pos2::new(x as f32 + 0.5, y as f32 + 0.5);
            let weights = [
                edge_function(vertices[1].0, vertices[2].0, sample) * inverse_area,
                edge_function(vertices[2].0, vertices[0].0, sample) * inverse_area,
                edge_function(vertices[0].0, vertices[1].0, sample) * inverse_area,
            ];
            if weights.iter().all(|weight| *weight >= -1.0e-5) {
                let sample_depth = weights
                    .iter()
                    .zip(vertices)
                    .map(|(weight, vertex)| weight * vertex.1)
                    .sum::<f32>();
                let target = &mut depth[y * resolution + x];
                *target = target.min(sample_depth);
            }
        }
    }
}

fn directional_shadow_visibility(
    surface: CoreVec3,
    normal: CoreVec3,
    map: &DirectionalShadowMap,
) -> f32 {
    let relative = surface - map.origin;
    let u = relative.dot(map.right) / (map.extent * 2.0) + 0.5;
    let v = 0.5 - relative.dot(map.up) / (map.extent * 2.0);
    if !(0.0..=1.0).contains(&u) || !(0.0..=1.0).contains(&v) {
        return 1.0;
    }
    let receiver_depth = relative.dot(map.forward);
    let normal_light = normal.dot(global_light_direction()).clamp(0.0, 1.0);
    let receiver_bias = map.bias * (1.0 + (1.0 - normal_light) * 5.0);
    let center_x = (u * map.resolution as f32).floor() as isize;
    let center_y = (v * map.resolution as f32).floor() as isize;
    let mut visible = 0.0;
    let mut samples = 0.0;
    let radius = map.filter_radius as isize;
    for offset_y in -radius..=radius {
        for offset_x in -radius..=radius {
            let x = center_x + offset_x;
            let y = center_y + offset_y;
            if x < 0 || y < 0 || x >= map.resolution as isize || y >= map.resolution as isize {
                visible += 1.0;
            } else {
                let caster_depth = map.depth[y as usize * map.resolution + x as usize];
                visible += f32::from(
                    !caster_depth.is_finite() || receiver_depth <= caster_depth + receiver_bias,
                );
            }
            samples += 1.0;
        }
    }
    visible / samples
}

fn cube_face_basis(face: usize) -> (CoreVec3, CoreVec3, CoreVec3) {
    match face {
        0 => (CoreVec3::X, CoreVec3::new(0.0, -1.0, 0.0), CoreVec3::Z),
        1 => (CoreVec3::new(-1.0, 0.0, 0.0), CoreVec3::Y, CoreVec3::Z),
        2 => (CoreVec3::Y, CoreVec3::X, CoreVec3::Z),
        3 => (
            CoreVec3::new(0.0, -1.0, 0.0),
            CoreVec3::new(-1.0, 0.0, 0.0),
            CoreVec3::Z,
        ),
        4 => (CoreVec3::Z, CoreVec3::X, CoreVec3::new(0.0, -1.0, 0.0)),
        _ => (CoreVec3::new(0.0, 0.0, -1.0), CoreVec3::X, CoreVec3::Y),
    }
}

fn cube_shadow_coordinate(direction: CoreVec3) -> (usize, f32, f32) {
    (0..6)
        .map(|face| {
            let (forward, right, up) = cube_face_basis(face);
            let depth = direction.dot(forward);
            (
                face,
                direction.dot(right) / depth * 0.5 + 0.5,
                0.5 - direction.dot(up) / depth * 0.5,
                depth,
            )
        })
        .max_by(|left, right| {
            left.3
                .partial_cmp(&right.3)
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .map(|(face, u, v, _)| (face, u, v))
        .unwrap()
}

fn build_point_shadow_atlas(
    triangles: &[PreviewTriangle],
    lights: &[ViewportLight],
    filter_radius: usize,
) -> PointShadowAtlas {
    build_point_shadow_atlas_with_blur(triangles, lights, filter_radius, 0)
}

fn build_point_shadow_atlas_with_blur(
    triangles: &[PreviewTriangle],
    lights: &[ViewportLight],
    filter_radius: usize,
    blur_radius: usize,
) -> PointShadowAtlas {
    let mut regions = [PointShadowRegion::default(); MAX_VIEWPORT_LIGHTS];
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
    let width = max_resolution * 6;
    let mut atlas = PointShadowAtlas {
        width,
        height,
        depth: vec![f32::INFINITY; width * height],
        regions,
    };
    let mut row = 0;
    for (light_index, light) in lights.iter().take(MAX_VIEWPORT_LIGHTS).enumerate() {
        let resolution = light.shadow_resolution as usize;
        if resolution == 0 {
            continue;
        }
        let bias = (1.0 / resolution as f32).max(1.0e-5) * 2.0;
        regions[light_index] = PointShadowRegion {
            row,
            resolution,
            bias,
            filter_radius,
        };
        for face in 0..6 {
            let (forward, right, up) = cube_face_basis(face);
            for triangle in triangles.iter().filter(|triangle| triangle.casts_shadows) {
                let face_vertices = triangle.vertices.map(|vertex| {
                    let point =
                        CoreVec3::new(vertex.position[0], vertex.position[1], vertex.position[2])
                            - light.position;
                    CoreVec3::new(point.dot(right), point.dot(up), point.dot(forward))
                });
                let clipped = clip_point_shadow_polygon(&face_vertices);
                for offset in 1..clipped.len().saturating_sub(1) {
                    let clipped_triangle = [clipped[0], clipped[offset], clipped[offset + 1]];
                    let projected = clipped_triangle.map(|point| {
                        (
                            Pos2::new(
                                face as f32 * resolution as f32
                                    + (point.x / point.z * 0.5 + 0.5) * resolution as f32,
                                row as f32 + (0.5 - point.y / point.z * 0.5) * resolution as f32,
                            ),
                            point.z,
                        )
                    });
                    rasterize_point_shadow_triangle(
                        &mut atlas.depth,
                        width,
                        height,
                        projected,
                        face * resolution,
                        row,
                        resolution,
                    );
                }
            }
            fast_blur_shadow_region(
                &mut atlas.depth,
                width,
                height,
                face * resolution,
                row,
                resolution,
                resolution,
                blur_radius,
            );
        }
        row += resolution;
    }
    atlas.regions = regions;
    atlas
}

fn clip_point_shadow_polygon(vertices: &[CoreVec3; 3]) -> Vec<CoreVec3> {
    let mut polygon = vertices.to_vec();
    for plane in [
        CoreVec3::new(0.0, 0.0, 1.0),
        CoreVec3::new(1.0, 0.0, 1.0),
        CoreVec3::new(-1.0, 0.0, 1.0),
        CoreVec3::new(0.0, 1.0, 1.0),
        CoreVec3::new(0.0, -1.0, 1.0),
    ] {
        if polygon.is_empty() {
            break;
        }
        let input = std::mem::take(&mut polygon);
        let mut previous = *input.last().unwrap();
        let mut previous_distance =
            previous.dot(plane) - if plane == CoreVec3::Z { 1.0e-5 } else { 0.0 };
        for current in input {
            let current_distance =
                current.dot(plane) - if plane == CoreVec3::Z { 1.0e-5 } else { 0.0 };
            let previous_inside = previous_distance >= 0.0;
            let current_inside = current_distance >= 0.0;
            if previous_inside != current_inside {
                let denominator = previous_distance - current_distance;
                if denominator.abs() > f32::EPSILON {
                    polygon
                        .push(previous + (current - previous) * (previous_distance / denominator));
                }
            }
            if current_inside {
                polygon.push(current);
            }
            previous = current;
            previous_distance = current_distance;
        }
    }
    polygon
}

#[allow(clippy::too_many_arguments)]
fn rasterize_point_shadow_triangle(
    atlas: &mut [f32],
    atlas_width: usize,
    atlas_height: usize,
    vertices: [(Pos2, f32); 3],
    tile_x: usize,
    tile_y: usize,
    resolution: usize,
) {
    let area = edge_function(vertices[0].0, vertices[1].0, vertices[2].0);
    if area.abs() <= 1.0e-8 {
        return;
    }
    let clamp_x =
        |value: f32| value.clamp(tile_x as f32, (tile_x + resolution - 1) as f32) as usize;
    let clamp_y =
        |value: f32| value.clamp(tile_y as f32, (tile_y + resolution - 1) as f32) as usize;
    let min_x = clamp_x(
        vertices
            .iter()
            .map(|vertex| vertex.0.x)
            .fold(f32::INFINITY, f32::min)
            .floor(),
    );
    let max_x = clamp_x(
        vertices
            .iter()
            .map(|vertex| vertex.0.x)
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil(),
    );
    let min_y = clamp_y(
        vertices
            .iter()
            .map(|vertex| vertex.0.y)
            .fold(f32::INFINITY, f32::min)
            .floor(),
    );
    let max_y = clamp_y(
        vertices
            .iter()
            .map(|vertex| vertex.0.y)
            .fold(f32::NEG_INFINITY, f32::max)
            .ceil(),
    );
    let inverse_area = area.recip();
    for y in min_y..=max_y.min(atlas_height - 1) {
        for x in min_x..=max_x.min(atlas_width - 1) {
            let sample = Pos2::new(x as f32 + 0.5, y as f32 + 0.5);
            let weights = [
                edge_function(vertices[1].0, vertices[2].0, sample) * inverse_area,
                edge_function(vertices[2].0, vertices[0].0, sample) * inverse_area,
                edge_function(vertices[0].0, vertices[1].0, sample) * inverse_area,
            ];
            if weights.iter().all(|weight| *weight >= -1.0e-5) {
                let inverse_face_depth = weights
                    .iter()
                    .zip(vertices)
                    .map(|(weight, vertex)| weight / vertex.1)
                    .sum::<f32>();
                if inverse_face_depth > f32::EPSILON {
                    let face_depth = inverse_face_depth.recip();
                    let face_x = ((sample.x - tile_x as f32) / resolution as f32) * 2.0 - 1.0;
                    let face_y = 1.0 - ((sample.y - tile_y as f32) / resolution as f32) * 2.0;
                    let radial_depth =
                        face_depth * (1.0 + face_x * face_x + face_y * face_y).sqrt();
                    let target = &mut atlas[y * atlas_width + x];
                    *target = target.min(radial_depth);
                }
            }
        }
    }
}

fn point_shadow_visibility(
    surface: CoreVec3,
    normal: CoreVec3,
    light: ViewportLight,
    light_index: usize,
    atlas: &PointShadowAtlas,
) -> f32 {
    let region = atlas.regions[light_index];
    if region.resolution == 0 {
        return 1.0;
    }
    let point = surface - light.position;
    let distance = point.length().max(1.0e-5);
    let direction = point * distance.recip();
    let surface_to_light = point * (-1.0 / distance);
    let normal_light = normal.dot(surface_to_light).clamp(0.0, 1.0);
    let receiver_bias = region.bias * distance * (1.0 + (1.0 - normal_light) * 5.0);
    let kernel = ((light.radius / distance) * region.resolution as f32)
        .ceil()
        .clamp(region.filter_radius as f32, 3.0) as isize;
    let reference = if direction.z.abs() < 0.9 {
        CoreVec3::Z
    } else {
        CoreVec3::Y
    };
    let tangent = reference.cross(direction).normalized();
    let bitangent = direction.cross(tangent).normalized();
    let angular_step = 2.0 / region.resolution as f32;
    let mut visible = 0.0;
    let mut samples = 0.0;
    for offset_y in -kernel..=kernel {
        for offset_x in -kernel..=kernel {
            let sample_direction = (direction
                + tangent * (offset_x as f32 * angular_step)
                + bitangent * (offset_y as f32 * angular_step))
                .normalized();
            let (face, u, v) = cube_shadow_coordinate(sample_direction);
            let local_x = (u * region.resolution as f32)
                .floor()
                .clamp(0.0, region.resolution.saturating_sub(1) as f32)
                as usize;
            let local_y = (v * region.resolution as f32)
                .floor()
                .clamp(0.0, region.resolution.saturating_sub(1) as f32)
                as usize;
            let x = face * region.resolution + local_x;
            let y = region.row + local_y;
            let caster_depth = atlas.depth[y * atlas.width + x];
            visible +=
                f32::from(!caster_depth.is_finite() || distance <= caster_depth + receiver_bias);
            samples += 1.0;
        }
    }
    visible / samples
}

fn viewport_light_factor(
    normal: CoreVec3,
    world_position: CoreVec3,
    shader: ShaderMode,
    lighting: &ViewportLighting,
    _receiver: Option<NodeId>,
) -> f32 {
    let mut diffuse = if lighting.global_enabled {
        let direction = global_light_direction();
        normal.dot(direction).max(0.0)
            * lighting.directional_shadow.as_deref().map_or(1.0, |map| {
                directional_shadow_visibility(world_position, normal, map)
            })
    } else {
        0.0
    };
    for (light_index, light) in lighting.points.iter().take(MAX_VIEWPORT_LIGHTS).enumerate() {
        let offset = light.position - world_position;
        let distance_squared = offset.dot(offset);
        if distance_squared <= 1.0e-8 {
            continue;
        }
        let luminance =
            (light.color[0].max(0.0) + light.color[1].max(0.0) + light.color[2].max(0.0)) / 3.0;
        let attenuation = light.intensity.max(0.0) * luminance / (1.0 + distance_squared);
        diffuse += spherical_light_lambert(
            normal.dot(offset.normalized()),
            distance_squared.sqrt(),
            light.radius,
        ) * attenuation
            * lighting.point_shadows.as_deref().map_or(1.0, |atlas| {
                point_shadow_visibility(world_position, normal, *light, light_index, atlas)
            });
    }
    shader_light_factor(diffuse.clamp(0.0, 2.0), shader).clamp(0.0, 2.0)
}

/// Smooth approximation of diffuse irradiance from a spherical emitter.
///
/// `angular_radius` is the sine of the emitter's apparent half-angle. The
/// square-root form converges exactly to max(N·L, 0) for a zero-radius light,
/// while finite radii round the terminator as an area light would.
fn spherical_light_lambert(normal_dot_light: f32, distance: f32, radius: f32) -> f32 {
    let angular_radius = if distance > f32::EPSILON {
        (radius.max(0.0) / distance).clamp(0.0, 1.0)
    } else {
        1.0
    };
    let rounded = 0.5
        * (normal_dot_light
            + (normal_dot_light * normal_dot_light + angular_radius * angular_radius).sqrt());
    let normalization = 2.0 / (1.0 + (1.0 + angular_radius * angular_radius).sqrt());
    (rounded * normalization).clamp(0.0, 1.0)
}

fn grid_distance_alpha(zoom: f32, grid_spacing: f32) -> f32 {
    (zoom.max(0.0) * grid_spacing.max(f32::EPSILON))
        .powf(1.5)
        .clamp(0.01, 1.0)
}

fn perspective_view_scale(
    base_scale: f32,
    projection_mode: ProjectionMode,
    grid_spacing: f32,
) -> f32 {
    match projection_mode {
        ProjectionMode::Perspective => base_scale / grid_spacing.max(f32::EPSILON),
        ProjectionMode::Orthographic => base_scale,
    }
}

fn canonical_ground_line(segment: [[f32; 3]; 2]) -> [[f32; 3]; 2] {
    if (segment[0][0] - segment[1][0]).abs() <= (segment[0][1] - segment[1][1]).abs() {
        [[segment[0][0], 0.0, 0.0], [segment[0][0], 1.0, 0.0]]
    } else {
        [[0.0, segment[0][1], 0.0], [1.0, segment[0][1], 0.0]]
    }
}

fn visible_grid_half_cells(rect: Rect, x_step: Vec2, y_step: Vec2) -> i32 {
    const MAX_HALF_CELLS: i32 = 2_000;
    let half_diagonal = rect.size().length() * 0.5;
    let smallest_step = x_step.length().min(y_step.length()).max(0.01);
    (half_diagonal * 1.5 / smallest_step)
        .ceil()
        .max(1.0)
        .min(MAX_HALF_CELLS as f32) as i32
}

#[allow(clippy::too_many_arguments)]
fn homogeneous_projection(
    point: [f32; 3],
    center: Pos2,
    scale: f32,
    yaw: f32,
    pitch: f32,
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
) -> [f32; 3] {
    let (view_x, view_y, depth) = view_coordinates(point, yaw, pitch, 0.0, camera_target);
    match projection_mode {
        ProjectionMode::Perspective => {
            let camera_distance = PERSPECTIVE_CAMERA_DISTANCE * perspective_unit;
            let denominator = camera_distance + depth;
            [
                center.x * denominator + scale * camera_distance * view_x,
                center.y * denominator - scale * camera_distance * view_y,
                denominator,
            ]
        }
        ProjectionMode::Orthographic => [center.x + scale * view_x, center.y - scale * view_y, 1.0],
    }
}

fn cross_homogeneous(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn perspective_ground_horizon(
    center: Pos2,
    scale: f32,
    yaw: f32,
    pitch: f32,
    perspective_unit: f32,
) -> [f32; 3] {
    let camera_distance = PERSPECTIVE_CAMERA_DISTANCE * perspective_unit;
    let direction_projection = |direction| {
        let (view_x, view_y, depth) = view_coordinates(direction, yaw, pitch, 0.0, CoreVec3::ZERO);
        [
            center.x * depth + scale * camera_distance * view_x,
            center.y * depth - scale * camera_distance * view_y,
            depth,
        ]
    };
    cross_homogeneous(
        direction_projection([1.0, 0.0, 0.0]),
        direction_projection([0.0, 1.0, 0.0]),
    )
}

fn clip_segment_to_half_plane(
    segment: [Pos2; 2],
    boundary: [f32; 3],
    visible_point: Pos2,
) -> Option<[Pos2; 2]> {
    let [a, b, c] = boundary;
    let normal_length = a.hypot(b);
    if normal_length <= f32::EPSILON {
        return Some(segment);
    }
    let visible_sign = (a * visible_point.x + b * visible_point.y + c).signum();
    if visible_sign == 0.0 {
        return None;
    }
    let distances =
        segment.map(|point| visible_sign * (a * point.x + b * point.y + c) / normal_length);
    let inside = distances.map(|distance| distance >= 0.0);
    match inside {
        [false, false] => None,
        [true, true] => Some(segment),
        _ => {
            let amount = distances[0] / (distances[0] - distances[1]);
            let intersection = segment[0].lerp(segment[1], amount);
            if inside[0] {
                Some([segment[0], intersection])
            } else {
                Some([intersection, segment[1]])
            }
        }
    }
}

fn clip_infinite_line_to_rect(line: [f32; 3], rect: Rect) -> Option<[Pos2; 2]> {
    let [a, b, c] = line;
    let mut intersections = Vec::with_capacity(4);
    if b.abs() > 1.0e-6 {
        for x in [rect.left(), rect.right()] {
            let point = Pos2::new(x, -(a * x + c) / b);
            if point.y >= rect.top() - 1.0e-3 && point.y <= rect.bottom() + 1.0e-3 {
                intersections.push(point);
            }
        }
    }
    if a.abs() > 1.0e-6 {
        for y in [rect.top(), rect.bottom()] {
            let point = Pos2::new(-(b * y + c) / a, y);
            if point.x >= rect.left() - 1.0e-3 && point.x <= rect.right() + 1.0e-3 {
                intersections.push(point);
            }
        }
    }
    intersections.dedup_by(|left, right| left.distance(*right) < 1.0e-3);
    if intersections.len() < 2 {
        return None;
    }
    let mut best = [intersections[0], intersections[1]];
    let mut best_distance = best[0].distance_sq(best[1]);
    for left in 0..intersections.len() {
        for right in left + 1..intersections.len() {
            let distance = intersections[left].distance_sq(intersections[right]);
            if distance > best_distance {
                best = [intersections[left], intersections[right]];
                best_distance = distance;
            }
        }
    }
    Some(best)
}

#[allow(clippy::too_many_arguments)]
fn projective_ground_line(
    segment: [[f32; 3]; 2],
    rect: Rect,
    center: Pos2,
    scale: f32,
    yaw: f32,
    pitch: f32,
    camera_target: CoreVec3,
    perspective_unit: f32,
) -> Option<[Pos2; 2]> {
    let viewport_segment = analytical_projected_line(
        segment,
        rect,
        center,
        scale,
        (
            yaw,
            pitch,
            camera_target,
            ProjectionMode::Perspective,
            perspective_unit,
        ),
    )?;
    let horizon = perspective_ground_horizon(center, scale, yaw, pitch, perspective_unit);
    let (_, _, base_depth) = view_coordinates(
        [camera_target.x, camera_target.y, 0.0],
        yaw,
        pitch,
        0.0,
        camera_target,
    );
    let forward_ground = CoreVec3::new(yaw.sin() * pitch.cos(), yaw.cos() * pitch.cos(), 0.0);
    let depth_gain = pitch.cos().powi(2).max(1.0e-4);
    let camera_distance = PERSPECTIVE_CAMERA_DISTANCE * perspective_unit;
    let near = PERSPECTIVE_NEAR * perspective_unit;
    let advance = ((near - camera_distance - base_depth) / depth_gain + perspective_unit).max(0.0);
    let visible_world = [
        camera_target.x + forward_ground.x * advance,
        camera_target.y + forward_ground.y * advance,
        0.0,
    ];
    let visible = homogeneous_projection(
        visible_world,
        center,
        scale,
        yaw,
        pitch,
        camera_target,
        ProjectionMode::Perspective,
        perspective_unit,
    );
    if visible[2] <= 0.0 {
        return None;
    }
    let visible_point = Pos2::new(visible[0] / visible[2], visible[1] / visible[2]);
    clip_segment_to_half_plane(viewport_segment, horizon, visible_point)
}

fn analytical_projected_line(
    line_points: [[f32; 3]; 2],
    rect: Rect,
    center: Pos2,
    scale: f32,
    camera: (f32, f32, CoreVec3, ProjectionMode, f32),
) -> Option<[Pos2; 2]> {
    let (yaw, pitch, camera_target, projection_mode, perspective_unit) = camera;
    let first = homogeneous_projection(
        line_points[0],
        center,
        scale,
        yaw,
        pitch,
        camera_target,
        projection_mode,
        perspective_unit,
    );
    let second = homogeneous_projection(
        line_points[1],
        center,
        scale,
        yaw,
        pitch,
        camera_target,
        projection_mode,
        perspective_unit,
    );
    clip_infinite_line_to_rect(cross_homogeneous(first, second), rect)
}

fn builtin_primitive_mesh(kind: BuiltinPrimitive) -> MeshAsset {
    match kind {
        BuiltinPrimitive::Cube => builtin_cube(),
        BuiltinPrimitive::Sphere => builtin_sphere(24, 16),
        BuiltinPrimitive::Floor => builtin_floor(),
    }
}

fn builtin_imported_assets() -> Vec<ImportedAsset> {
    [
        BuiltinPrimitive::Cube,
        BuiltinPrimitive::Sphere,
        BuiltinPrimitive::Floor,
    ]
    .into_iter()
    .map(|kind| {
        let mesh = builtin_primitive_mesh(kind);
        let bounds = mesh_bounds(&mesh);
        ImportedAsset {
            path: kind.asset_path().into(),
            autofixed_mesh: mesh.clone(),
            mesh,
            autofix_report: MeshAutofixReport::default(),
            bounds,
        }
    })
    .collect()
}

fn builtin_cube() -> MeshAsset {
    let mut vertices = Vec::with_capacity(24);
    let mut indices = Vec::with_capacity(36);
    let mut face = |corners: [[f32; 3]; 4], normal: [f32; 3]| {
        let start = vertices.len() as u32;
        for (position, uv) in
            corners
                .into_iter()
                .zip([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]])
        {
            vertices.push(Vertex {
                position,
                normal,
                uv,
                color: [1.0; 4],
            });
        }
        indices.extend_from_slice(&[start, start + 1, start + 2, start, start + 2, start + 3]);
    };
    face(
        [
            [0.5, -0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, 0.5, 0.5],
            [0.5, -0.5, 0.5],
        ],
        [1.0, 0.0, 0.0],
    );
    face(
        [
            [-0.5, 0.5, -0.5],
            [-0.5, -0.5, -0.5],
            [-0.5, -0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        [-1.0, 0.0, 0.0],
    );
    face(
        [
            [-0.5, 0.5, -0.5],
            [-0.5, 0.5, 0.5],
            [0.5, 0.5, 0.5],
            [0.5, 0.5, -0.5],
        ],
        [0.0, 1.0, 0.0],
    );
    face(
        [
            [-0.5, -0.5, 0.5],
            [-0.5, -0.5, -0.5],
            [0.5, -0.5, -0.5],
            [0.5, -0.5, 0.5],
        ],
        [0.0, -1.0, 0.0],
    );
    face(
        [
            [-0.5, -0.5, 0.5],
            [0.5, -0.5, 0.5],
            [0.5, 0.5, 0.5],
            [-0.5, 0.5, 0.5],
        ],
        [0.0, 0.0, 1.0],
    );
    face(
        [
            [-0.5, 0.5, -0.5],
            [0.5, 0.5, -0.5],
            [0.5, -0.5, -0.5],
            [-0.5, -0.5, -0.5],
        ],
        [0.0, 0.0, -1.0],
    );
    canonical_builtin_mesh("Cube", vertices, indices)
}

fn builtin_sphere(segments: usize, stacks: usize) -> MeshAsset {
    let mut vertices = Vec::with_capacity(2 + segments * (stacks - 1));
    vertices.push(Vertex {
        position: [0.0, 0.0, -0.5],
        normal: [0.0, 0.0, -1.0],
        uv: [0.5, 1.0],
        color: [1.0; 4],
    });
    for stack in 1..stacks {
        let v = stack as f32 / stacks as f32;
        let latitude = -std::f32::consts::FRAC_PI_2 + v * std::f32::consts::PI;
        let (z, radius) = latitude.sin_cos();
        for segment in 0..segments {
            let u = segment as f32 / segments as f32;
            let longitude = u * std::f32::consts::TAU;
            let (y, x) = longitude.sin_cos();
            let normal = [radius * x, radius * y, z];
            vertices.push(Vertex {
                position: normal.map(|value| value * 0.5),
                normal,
                uv: [u, 1.0 - v],
                color: [1.0; 4],
            });
        }
    }
    let top = vertices.len() as u32;
    vertices.push(Vertex {
        position: [0.0, 0.0, 0.5],
        normal: [0.0, 0.0, 1.0],
        uv: [0.5, 0.0],
        color: [1.0; 4],
    });

    let mut indices = Vec::with_capacity(segments * (stacks - 1) * 6);
    for segment in 0..segments {
        let current = 1 + segment as u32;
        let next = 1 + ((segment + 1) % segments) as u32;
        indices.extend_from_slice(&[0, next, current]);
    }
    for ring in 0..stacks - 2 {
        for segment in 0..segments {
            let a = 1 + (ring * segments + segment) as u32;
            let b = 1 + (ring * segments + (segment + 1) % segments) as u32;
            let d = 1 + ((ring + 1) * segments + segment) as u32;
            let c = 1 + ((ring + 1) * segments + (segment + 1) % segments) as u32;
            indices.extend_from_slice(&[a, b, c, a, c, d]);
        }
    }
    let last_ring = 1 + ((stacks - 2) * segments) as u32;
    for segment in 0..segments {
        let current = last_ring + segment as u32;
        let next = last_ring + ((segment + 1) % segments) as u32;
        indices.extend_from_slice(&[current, next, top]);
    }
    canonical_builtin_mesh("Sphere", vertices, indices)
}

fn builtin_floor() -> MeshAsset {
    canonical_builtin_mesh(
        "Floor",
        vec![
            Vertex {
                position: [-5.0, -5.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 0.0],
                color: [1.0; 4],
            },
            Vertex {
                position: [5.0, -5.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [10.0, 0.0],
                color: [1.0; 4],
            },
            Vertex {
                position: [5.0, 5.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [10.0, 10.0],
                color: [1.0; 4],
            },
            Vertex {
                position: [-5.0, 5.0, 0.0],
                normal: [0.0, 0.0, 1.0],
                uv: [0.0, 10.0],
                color: [1.0; 4],
            },
        ],
        vec![0, 1, 2, 0, 2, 3],
    )
}

fn canonical_builtin_mesh(name: &str, vertices: Vec<Vertex>, indices: Vec<u32>) -> MeshAsset {
    MeshAsset {
        name: name.into(),
        vertices,
        primitives: vec![Primitive {
            name: name.into(),
            material: None,
            indices,
        }],
        source: zerofps_assets::SourceInfo {
            format: "ZeroFPS built-in".into(),
            up_axis: zerofps_assets::AxisConvention::ZUp,
            handedness: zerofps_assets::Handedness::Right,
            unit_scale_meters: Some(1.0),
            ..Default::default()
        },
        ..Default::default()
    }
}

fn infer_grid_spacing(asset: &MeshAsset) -> Option<f32> {
    let mut longest_squared = 0.0f32;
    for primitive in &asset.primitives {
        for triangle in primitive.indices.chunks_exact(3) {
            for (left, right) in [(0, 1), (1, 2), (2, 0)] {
                let a = asset.vertices.get(triangle[left] as usize)?.position;
                let b = asset.vertices.get(triangle[right] as usize)?.position;
                let squared = (0..3)
                    .map(|axis| {
                        let delta = b[axis] - a[axis];
                        delta * delta
                    })
                    .sum::<f32>();
                if squared.is_finite() {
                    longest_squared = longest_squared.max(squared);
                }
            }
        }
    }
    let longest = longest_squared.sqrt();
    (longest > f32::EPSILON).then(|| {
        10.0f32
            .powf(longest.log10().round())
            .clamp(0.001, 1_000_000.0)
    })
}

fn mesh_bounds(asset: &MeshAsset) -> ([f32; 3], [f32; 3]) {
    asset.vertices.iter().fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(mut minimum, mut maximum), vertex| {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(vertex.position[axis]);
                maximum[axis] = maximum[axis].max(vertex.position[axis]);
            }
            (minimum, maximum)
        },
    )
}

fn fitted_collider_from_bounds(bounds: ([f32; 3], [f32; 3]), shape: ColliderShape) -> Component {
    let center = CoreVec3::new(
        (bounds.0[0] + bounds.1[0]) * 0.5,
        (bounds.0[1] + bounds.1[1]) * 0.5,
        (bounds.0[2] + bounds.1[2]) * 0.5,
    );
    let extents = CoreVec3::new(
        ((bounds.1[0] - bounds.0[0]) * 0.5).max(0.001),
        ((bounds.1[1] - bounds.0[1]) * 0.5).max(0.001),
        ((bounds.1[2] - bounds.0[2]) * 0.5).max(0.001),
    );
    let (half_extents, radius, height) = match shape {
        ColliderShape::Sphere => {
            let radius = extents.x.max(extents.y).max(extents.z);
            (CoreVec3::new(radius, radius, radius), radius, radius * 2.0)
        }
        ColliderShape::Cylinder => {
            let radius = extents.x.max(extents.y);
            (
                CoreVec3::new(radius, radius, extents.z),
                radius,
                extents.z * 2.0,
            )
        }
        ColliderShape::Box => (
            extents,
            extents.x.max(extents.y).max(extents.z),
            extents.z * 2.0,
        ),
        ColliderShape::Flat => (
            CoreVec3::new(extents.x, extents.y, 0.005),
            extents.x.max(extents.y),
            0.01,
        ),
    };
    let density = 1_000.0;
    let mass = collider_volume(shape, half_extents, radius, height) * density;
    let friction = CoreVec3::new(100.0, 100.0, 100.0);
    Component::Collider {
        shape: shape.storage_name().into(),
        center,
        half_extents,
        radius,
        height,
        coupling_stiffness: 0.5,
        coupling_damping: 1.0,
        elasticity_stiffness: 0.5,
        elasticity_damping: 1.0,
        restitution: 0.0,
        force_cutoff: 0.01,
        collision_force_cutoff: 0.01,
        density,
        mass,
        automatic_mass: true,
        friction,
        friction_margin_percent: 5.0,
        joint: CylinderJoint::None.storage_name().into(),
    }
}

fn collider_volume(shape: ColliderShape, half_extents: CoreVec3, radius: f32, height: f32) -> f32 {
    match shape {
        ColliderShape::Sphere => 4.0 / 3.0 * std::f32::consts::PI * radius.max(0.0).powi(3),
        ColliderShape::Cylinder => std::f32::consts::PI * radius.max(0.0).powi(2) * height.max(0.0),
        ColliderShape::Box | ColliderShape::Flat => {
            8.0 * half_extents.x.max(0.0) * half_extents.y.max(0.0) * half_extents.z.max(0.0)
        }
    }
}

fn collider_vertical_extent(
    shape: ColliderShape,
    half_extents: CoreVec3,
    scale: CoreVec3,
    rotation: Quat,
) -> f32 {
    let scaled = scale.component_mul(half_extents);
    match shape {
        ColliderShape::Sphere => scaled.x.abs().max(scaled.y.abs()).max(scaled.z.abs()),
        ColliderShape::Cylinder => {
            let axis = rotation.rotate(CoreVec3::new(0.0, 0.0, 1.0)).normalized();
            let radial_extent = scaled.x.abs().max(scaled.y.abs());
            let radial_vertical = (1.0 - axis.z * axis.z).max(0.0).sqrt();
            radial_extent * radial_vertical + scaled.z.abs() * axis.z.abs()
        }
        ColliderShape::Box | ColliderShape::Flat => [
            CoreVec3::new(scaled.x, 0.0, 0.0),
            CoreVec3::new(0.0, scaled.y, 0.0),
            CoreVec3::new(0.0, 0.0, scaled.z),
        ]
        .into_iter()
        .map(|axis| rotation.rotate(axis).z.abs())
        .sum(),
    }
}

#[derive(Clone, Copy, Debug)]
struct WorldCollider {
    center: CoreVec3,
    half_extents: CoreVec3,
    restitution: f32,
    friction: f32,
    friction_margin_percent: f32,
    shape: ColliderShape,
    joint: CylinderJoint,
}

#[derive(Clone, Copy, Debug)]
struct ColliderContact {
    normal: CoreVec3,
    penetration: f32,
    restitution: f32,
    friction: f32,
}

fn world_collider(transform: Transform, component: &Component) -> Option<WorldCollider> {
    let Component::Collider {
        shape,
        center,
        half_extents,
        radius,
        height,
        restitution,
        friction,
        friction_margin_percent,
        joint,
        ..
    } = component
    else {
        return None;
    };
    let shape = ColliderShape::from_storage(shape);
    let center = transform.translation
        + transform
            .rotation
            .rotate(transform.scale.component_mul(*center));
    let scale = CoreVec3::new(
        transform.scale.x.abs(),
        transform.scale.y.abs(),
        transform.scale.z.abs(),
    );
    let half_extents = match shape {
        ColliderShape::Sphere => {
            let radius = radius.max(0.001) * scale.x.max(scale.y).max(scale.z);
            CoreVec3::new(radius, radius, radius)
        }
        ColliderShape::Cylinder => {
            let axis = transform
                .rotation
                .rotate(CoreVec3::new(0.0, 0.0, 1.0))
                .normalized();
            let radial = radius.max(0.001) * scale.x.max(scale.y);
            let axial = (height.max(0.002) * 0.5) * scale.z;
            CoreVec3::new(
                axial * axis.x.abs() + radial * (1.0 - axis.x * axis.x).max(0.0).sqrt(),
                axial * axis.y.abs() + radial * (1.0 - axis.y * axis.y).max(0.0).sqrt(),
                axial * axis.z.abs() + radial * (1.0 - axis.z * axis.z).max(0.0).sqrt(),
            )
        }
        ColliderShape::Box | ColliderShape::Flat => {
            let scaled = scale.component_mul(*half_extents);
            let x = transform.rotation.rotate(CoreVec3::new(scaled.x, 0.0, 0.0));
            let y = transform.rotation.rotate(CoreVec3::new(0.0, scaled.y, 0.0));
            let z = transform.rotation.rotate(CoreVec3::new(0.0, 0.0, scaled.z));
            CoreVec3::new(
                x.x.abs() + y.x.abs() + z.x.abs(),
                x.y.abs() + y.y.abs() + z.y.abs(),
                x.z.abs() + y.z.abs() + z.z.abs(),
            )
        }
    };
    Some(WorldCollider {
        center,
        half_extents,
        restitution: restitution.clamp(0.0, 1.0),
        friction: ((friction.x.max(0.0) + friction.y.max(0.0) + friction.z.max(0.0)) / 3.0)
            .max(0.0),
        friction_margin_percent: friction_margin_percent.max(0.0),
        shape,
        joint: CylinderJoint::from_storage(joint),
    })
}

fn collider_pair_contact(a: WorldCollider, b: WorldCollider) -> Option<ColliderContact> {
    if a.shape == ColliderShape::Sphere && b.shape == ColliderShape::Sphere {
        let delta = a.center - b.center;
        let distance = delta.length();
        let combined_radius = a.half_extents.x + b.half_extents.x;
        if distance >= combined_radius {
            return None;
        }
        return Some(ColliderContact {
            normal: if distance > 1.0e-6 {
                delta * (1.0 / distance)
            } else {
                CoreVec3::Z
            },
            penetration: combined_radius - distance,
            restitution: a.restitution.max(b.restitution),
            friction: (a.friction * b.friction).sqrt(),
        });
    }
    if a.shape == ColliderShape::Sphere && b.shape != ColliderShape::Sphere {
        return sphere_bounds_contact(a, b);
    }
    if b.shape == ColliderShape::Sphere && a.shape != ColliderShape::Sphere {
        return sphere_bounds_contact(b, a).map(|contact| ColliderContact {
            normal: contact.normal * -1.0,
            ..contact
        });
    }
    let delta = a.center - b.center;
    let overlap = CoreVec3::new(
        a.half_extents.x + b.half_extents.x - delta.x.abs(),
        a.half_extents.y + b.half_extents.y - delta.y.abs(),
        a.half_extents.z + b.half_extents.z - delta.z.abs(),
    );
    if overlap.x <= 0.0 || overlap.y <= 0.0 || overlap.z <= 0.0 {
        return None;
    }
    let (normal, penetration) = if overlap.x <= overlap.y && overlap.x <= overlap.z {
        (
            CoreVec3::new(if delta.x < 0.0 { -1.0 } else { 1.0 }, 0.0, 0.0),
            overlap.x,
        )
    } else if overlap.y <= overlap.z {
        (
            CoreVec3::new(0.0, if delta.y < 0.0 { -1.0 } else { 1.0 }, 0.0),
            overlap.y,
        )
    } else {
        (
            CoreVec3::new(0.0, 0.0, if delta.z < 0.0 { -1.0 } else { 1.0 }),
            overlap.z,
        )
    };
    Some(ColliderContact {
        normal,
        penetration,
        restitution: a.restitution.max(b.restitution),
        friction: (a.friction * b.friction).sqrt(),
    })
}

fn friction_contact_collider(collider: WorldCollider) -> WorldCollider {
    let factor = 1.0 + collider.friction_margin_percent.max(0.0) * 0.01;
    WorldCollider {
        half_extents: collider.half_extents * factor,
        // The expanded proxy is used only to discover a nearby friction pair.
        // It must never add restitution or positional correction of its own.
        restitution: 0.0,
        ..collider
    }
}

fn friction_contact_tolerance(extent_along_normal: f32, margin_percent: f32) -> f32 {
    extent_along_normal.abs() * margin_percent.max(0.0) * 0.01
}

fn sphere_bounds_contact(sphere: WorldCollider, bounds: WorldCollider) -> Option<ColliderContact> {
    let minimum = bounds.center - bounds.half_extents;
    let maximum = bounds.center + bounds.half_extents;
    let closest = CoreVec3::new(
        sphere.center.x.clamp(minimum.x, maximum.x),
        sphere.center.y.clamp(minimum.y, maximum.y),
        sphere.center.z.clamp(minimum.z, maximum.z),
    );
    let delta = sphere.center - closest;
    let distance = delta.length();
    let radius = sphere.half_extents.x;
    if distance >= radius {
        return None;
    }
    if distance > 1.0e-6 {
        return Some(ColliderContact {
            normal: delta * (1.0 / distance),
            penetration: radius - distance,
            restitution: sphere.restitution.max(bounds.restitution),
            friction: (sphere.friction * bounds.friction).sqrt(),
        });
    }
    // The center is inside the bounds. The AABB minimum-translation axis
    // provides a stable escape direction instead of an undefined zero normal.
    let distances = [
        (sphere.center.x - minimum.x, CoreVec3::new(-1.0, 0.0, 0.0)),
        (maximum.x - sphere.center.x, CoreVec3::X),
        (sphere.center.y - minimum.y, CoreVec3::new(0.0, -1.0, 0.0)),
        (maximum.y - sphere.center.y, CoreVec3::Y),
        (sphere.center.z - minimum.z, CoreVec3::new(0.0, 0.0, -1.0)),
        (maximum.z - sphere.center.z, CoreVec3::Z),
    ];
    let (surface_distance, normal) = distances
        .into_iter()
        .min_by(|left, right| left.0.total_cmp(&right.0))
        .unwrap();
    Some(ColliderContact {
        normal,
        penetration: radius + surface_distance,
        restitution: sphere.restitution.max(bounds.restitution),
        friction: (sphere.friction * bounds.friction).sqrt(),
    })
}

fn collider_support_surface(
    position: CoreVec3,
    colliders: &[WorldCollider],
) -> Option<(f32, f32, f32)> {
    colliders
        .iter()
        .filter(|collider| {
            matches!(collider.shape, ColliderShape::Box | ColliderShape::Flat)
                && position.x >= collider.center.x - collider.half_extents.x - f32::EPSILON
                && position.x <= collider.center.x + collider.half_extents.x + f32::EPSILON
                && position.y >= collider.center.y - collider.half_extents.y - f32::EPSILON
                && position.y <= collider.center.y + collider.half_extents.y + f32::EPSILON
        })
        .map(|collider| {
            (
                collider.center.z + collider.half_extents.z,
                collider.friction,
                friction_contact_tolerance(
                    collider.half_extents.z,
                    collider.friction_margin_percent,
                ),
            )
        })
        .max_by(|left, right| left.0.total_cmp(&right.0))
}

fn resolve_general_collider_contacts(
    position: &mut CoreVec3,
    velocity: &mut CoreVec3,
    scene_body_position: CoreVec3,
    body_colliders: &[WorldCollider],
    external_colliders: &[WorldCollider],
    dt: f32,
) -> usize {
    let mut resolved = 0;
    for body in body_colliders {
        for external in external_colliders {
            let shifted_body = WorldCollider {
                center: body.center + (*position - scene_body_position),
                ..*body
            };
            let collision_contact = collider_pair_contact(shifted_body, *external);
            let friction_contact = collision_contact.or_else(|| {
                collider_pair_contact(
                    friction_contact_collider(shifted_body),
                    friction_contact_collider(*external),
                )
            });
            let Some(contact) = friction_contact else {
                continue;
            };
            let inward_speed = velocity.dot(contact.normal);
            if let Some(collision) = collision_contact {
                *position = *position + collision.normal * collision.penetration;
                if inward_speed < 0.0 {
                    *velocity = *velocity
                        - collision.normal * ((1.0 + collision.restitution) * inward_speed);
                }
            }
            // Jointed cylinders have a rotating contact surface. Their
            // tangential impulse and reaction torque are resolved by the
            // wheel-aware support solver; treating them as static shapes here
            // would immediately cancel engine traction.
            if body.shape != ColliderShape::Cylinder || body.joint == CylinderJoint::None {
                let normal_speed = inward_speed.abs().max(9.81 * dt);
                let tangent = *velocity - contact.normal * velocity.dot(contact.normal);
                let tangent_speed = tangent.length();
                if tangent_speed > 1.0e-6 {
                    let removed_speed = (contact.friction * normal_speed).min(tangent_speed);
                    *velocity = *velocity - tangent * (removed_speed / tangent_speed);
                }
            }
            resolved += 1;
        }
    }
    resolved
}

fn force_cut_scalar(value: f32, cutoff: f32) -> f32 {
    if value.abs() <= cutoff.max(0.0) {
        0.0
    } else {
        value
    }
}

fn smooth_rpm_governor_torque(
    throttle: f32,
    angular_velocity: f32,
    maximum_torque: f32,
    max_rpm: f32,
    direction: f32,
) -> f32 {
    let throttle = throttle.clamp(-1.0, 1.0);
    let max_rpm = max_rpm.max(0.0);
    let target_rpm = throttle * direction.signum() * max_rpm;
    let current_rpm = angular_velocity * 60.0 / std::f32::consts::TAU;
    let transition_rpm = (max_rpm * 0.05).max(50.0);
    let normalized_error = ((target_rpm - current_rpm) / transition_rpm).clamp(-1.0, 1.0);
    let magnitude = normalized_error.abs();
    let smooth_magnitude = magnitude * magnitude * (3.0 - 2.0 * magnitude);
    normalized_error.signum() * maximum_torque.max(0.0) * smooth_magnitude
}

fn force_cut_vector(value: CoreVec3, cutoff: f32) -> CoreVec3 {
    if value.length() <= cutoff.max(0.0) {
        CoreVec3::ZERO
    } else {
        value
    }
}

fn collider_friction_coefficient(
    shape: ColliderShape,
    rotation: Quat,
    friction: CoreVec3,
    tangent_velocity: CoreVec3,
) -> f32 {
    let direction = tangent_velocity.normalized();
    match shape {
        ColliderShape::Sphere => friction.x.max(0.0),
        ColliderShape::Cylinder => {
            let normal = CoreVec3::new(0.0, 0.0, 1.0);
            let axis = rotation.rotate(CoreVec3::new(0.0, 0.0, 1.0));
            let rolling = axis.cross(normal).normalized();
            if rolling.length() <= 1.0e-6 {
                friction.x.max(0.0)
            } else {
                let rolling_weight = direction.dot(rolling).abs().clamp(0.0, 1.0);
                let non_rolling = (friction.y.max(0.0) + friction.z.max(0.0)) * 0.5;
                friction.x.max(0.0) * rolling_weight + non_rolling * (1.0 - rolling_weight)
            }
        }
        ColliderShape::Box | ColliderShape::Flat => {
            let local = rotation.inverse().rotate(direction);
            let weights = CoreVec3::new(local.x.abs(), local.y.abs(), local.z.abs());
            (weights.x * friction.x.max(0.0)
                + weights.y * friction.y.max(0.0)
                + weights.z * friction.z.max(0.0))
                / (weights.x + weights.y + weights.z).max(1.0e-6)
        }
    }
}

fn contact_friction_force(
    tangent_velocity: CoreVec3,
    coefficient: f32,
    normal_force: f32,
    effective_mass: f32,
    dt: f32,
) -> CoreVec3 {
    let speed = tangent_velocity.length();
    if speed <= 1.0e-6 || coefficient <= 0.0 || normal_force <= 0.0 {
        return CoreVec3::ZERO;
    }
    let coulomb_limit = coefficient.max(0.0) * normal_force.max(0.0);
    let stop_force = effective_mass.max(0.0) * speed / dt.max(1.0e-6);
    tangent_velocity.normalized() * -coulomb_limit.min(stop_force)
}

fn cylinder_box_traction_force(
    cylinder: WorldCollider,
    box_collider: WorldCollider,
    axle: CoreVec3,
    angular_velocity: f32,
    body_velocity: CoreVec3,
    normal_force: f32,
    effective_mass: f32,
    dt: f32,
) -> Option<CoreVec3> {
    if cylinder.shape != ColliderShape::Cylinder
        || !matches!(box_collider.shape, ColliderShape::Box | ColliderShape::Flat)
    {
        return None;
    }
    let contact = collider_pair_contact(cylinder, box_collider).or_else(|| {
        collider_pair_contact(
            friction_contact_collider(cylinder),
            friction_contact_collider(box_collider),
        )
    })?;
    let radius_arm = contact.normal * -cylinder.half_extents.z;
    let surface_velocity = (axle.normalized() * angular_velocity).cross(radius_arm);
    let relative_velocity = body_velocity + surface_velocity;
    let tangent = relative_velocity - contact.normal * relative_velocity.dot(contact.normal);
    Some(contact_friction_force(
        tangent,
        contact.friction,
        normal_force,
        effective_mass,
        dt,
    ))
}

fn resolve_contact_normal_velocity(inward_velocity: f32, restitution: f32) -> f32 {
    if inward_velocity < 0.0 {
        -inward_velocity * restitution.clamp(0.0, 1.0)
    } else {
        inward_velocity
    }
}

fn contact_count_for_solver(actual_contacts: usize) -> (f32, bool) {
    (actual_contacts.max(1) as f32, actual_contacts > 0)
}

fn coupled_contact_mass(
    body_mass: f32,
    joint_inertia: f32,
    radius: f32,
    rolling_weight: f32,
) -> f32 {
    let body_inverse = 1.0 / body_mass.max(1.0e-6);
    let joint_inverse =
        rolling_weight.clamp(0.0, 1.0) * radius.max(0.0).powi(2) / joint_inertia.max(1.0e-6);
    1.0 / (body_inverse + joint_inverse).max(1.0e-6)
}

fn collider_wireframe_segments(
    shape: ColliderShape,
    center: CoreVec3,
    half_extents: CoreVec3,
    radius: f32,
    height: f32,
) -> Vec<[[f32; 3]; 2]> {
    let offset = |point: [f32; 3]| {
        [
            point[0] + center.x,
            point[1] + center.y,
            point[2] + center.z,
        ]
    };
    if matches!(shape, ColliderShape::Box | ColliderShape::Flat) {
        let corners = [
            [-half_extents.x, -half_extents.y, -half_extents.z],
            [half_extents.x, -half_extents.y, -half_extents.z],
            [half_extents.x, half_extents.y, -half_extents.z],
            [-half_extents.x, half_extents.y, -half_extents.z],
            [-half_extents.x, -half_extents.y, half_extents.z],
            [half_extents.x, -half_extents.y, half_extents.z],
            [half_extents.x, half_extents.y, half_extents.z],
            [-half_extents.x, half_extents.y, half_extents.z],
        ];
        return [
            (0, 1),
            (1, 2),
            (2, 3),
            (3, 0),
            (4, 5),
            (5, 6),
            (6, 7),
            (7, 4),
            (0, 4),
            (1, 5),
            (2, 6),
            (3, 7),
        ]
        .into_iter()
        .map(|(a, b)| [offset(corners[a]), offset(corners[b])])
        .collect();
    }

    const SIDES: usize = 32;
    let mut segments = Vec::new();
    let circle_point = |plane: usize, angle: f32, axial: f32| {
        let (sin, cos) = angle.sin_cos();
        let radial = radius.max(0.0);
        match plane {
            0 => offset([radial * cos, radial * sin, axial]),
            1 => offset([radial * cos, axial, radial * sin]),
            _ => offset([axial, radial * cos, radial * sin]),
        }
    };
    match shape {
        ColliderShape::Sphere => {
            for plane in 0..3 {
                for side in 0..SIDES {
                    let a = side as f32 * std::f32::consts::TAU / SIDES as f32;
                    let b = (side + 1) as f32 * std::f32::consts::TAU / SIDES as f32;
                    segments.push([circle_point(plane, a, 0.0), circle_point(plane, b, 0.0)]);
                }
            }
        }
        ColliderShape::Cylinder => {
            let half_height = height.max(0.0) * 0.5;
            for axial in [-half_height, half_height] {
                for side in 0..SIDES {
                    let a = side as f32 * std::f32::consts::TAU / SIDES as f32;
                    let b = (side + 1) as f32 * std::f32::consts::TAU / SIDES as f32;
                    segments.push([circle_point(0, a, axial), circle_point(0, b, axial)]);
                }
            }
            for side in 0..8 {
                let angle = side as f32 * std::f32::consts::TAU / 8.0;
                segments.push([
                    circle_point(0, angle, -half_height),
                    circle_point(0, angle, half_height),
                ]);
            }
        }
        ColliderShape::Box | ColliderShape::Flat => unreachable!(),
    }
    segments
}

fn pan_camera_target(
    target: CoreVec3,
    pointer_delta: Vec2,
    pixels_per_unit: f32,
    yaw: f32,
    pitch: f32,
) -> CoreVec3 {
    if pixels_per_unit <= f32::EPSILON {
        return target;
    }
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    let view_right = CoreVec3::new(cos_yaw, -sin_yaw, 0.0);
    let view_up = CoreVec3::new(-sin_yaw * sin_pitch, -cos_yaw * sin_pitch, cos_pitch);
    target
        + view_right * (-pointer_delta.x / pixels_per_unit)
        + view_up * (pointer_delta.y / pixels_per_unit)
}

#[allow(clippy::too_many_arguments)]
fn project(
    point: [f32; 3],
    center: Pos2,
    scale: f32,
    yaw: f32,
    pitch: f32,
    roll: f32,
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
) -> Option<Pos2> {
    let (x, y, depth) = view_coordinates(point, yaw, pitch, roll, camera_target);
    let projection_scale = match projection_mode {
        ProjectionMode::Perspective => {
            let camera_distance = PERSPECTIVE_CAMERA_DISTANCE * perspective_unit;
            let camera_depth = camera_distance + depth;
            if camera_depth < PERSPECTIVE_NEAR * perspective_unit {
                return None;
            }
            camera_distance / camera_depth
        }
        ProjectionMode::Orthographic => 1.0,
    };
    Some(center + Vec2::new(x, -y) * scale * projection_scale)
}

fn view_coordinates(
    point: [f32; 3],
    yaw: f32,
    pitch: f32,
    roll: f32,
    camera_target: CoreVec3,
) -> (f32, f32, f32) {
    let offset = CoreVec3::new(
        point[0] - camera_target.x,
        point[1] - camera_target.y,
        point[2] - camera_target.z,
    );
    let (right, up, forward) = camera_basis(yaw, pitch, roll);
    (offset.dot(right), offset.dot(up), offset.dot(forward))
}

fn view_depth(point: [f32; 3], yaw: f32, pitch: f32, roll: f32, camera_target: CoreVec3) -> f32 {
    view_coordinates(point, yaw, pitch, roll, camera_target).2
}

fn near_distance(
    point: [f32; 3],
    yaw: f32,
    pitch: f32,
    roll: f32,
    camera_target: CoreVec3,
    perspective_unit: f32,
) -> f32 {
    PERSPECTIVE_CAMERA_DISTANCE * perspective_unit
        + view_depth(point, yaw, pitch, roll, camera_target)
}

fn interpolate_point(a: [f32; 3], b: [f32; 3], amount: f32) -> [f32; 3] {
    [
        a[0] + (b[0] - a[0]) * amount,
        a[1] + (b[1] - a[1]) * amount,
        a[2] + (b[2] - a[2]) * amount,
    ]
}

fn clip_segment_to_near(
    mut segment: [[f32; 3]; 2],
    yaw: f32,
    pitch: f32,
    roll: f32,
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
) -> Option<[[f32; 3]; 2]> {
    if projection_mode == ProjectionMode::Orthographic {
        return Some(segment);
    }
    let distances = segment
        .map(|point| near_distance(point, yaw, pitch, roll, camera_target, perspective_unit));
    let near = PERSPECTIVE_NEAR * perspective_unit;
    let inside = distances.map(|distance| distance >= near);
    match inside {
        [false, false] => None,
        [true, true] => Some(segment),
        _ => {
            let amount = (near - distances[0]) / (distances[1] - distances[0]);
            let intersection = interpolate_point(segment[0], segment[1], amount);
            if inside[0] {
                segment[1] = intersection;
            } else {
                segment[0] = intersection;
            }
            Some(segment)
        }
    }
}

#[allow(clippy::too_many_arguments)]
fn project_segment(
    segment: [[f32; 3]; 2],
    center: Pos2,
    scale: f32,
    yaw: f32,
    pitch: f32,
    roll: f32,
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
) -> Option<[Pos2; 2]> {
    let clipped = clip_segment_to_near(
        segment,
        yaw,
        pitch,
        roll,
        camera_target,
        projection_mode,
        perspective_unit,
    )?;
    Some([
        project(
            clipped[0],
            center,
            scale,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            perspective_unit,
        )?,
        project(
            clipped[1],
            center,
            scale,
            yaw,
            pitch,
            roll,
            camera_target,
            projection_mode,
            perspective_unit,
        )?,
    ])
}

fn clip_polygon_to_near(
    polygon: &[[f32; 3]],
    yaw: f32,
    pitch: f32,
    roll: f32,
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
) -> Vec<[f32; 3]> {
    let mut output = Vec::with_capacity(polygon.len() + 1);
    clip_polygon_to_near_into(
        polygon,
        yaw,
        pitch,
        roll,
        camera_target,
        projection_mode,
        perspective_unit,
        &mut output,
    );
    output
}

#[allow(clippy::too_many_arguments)]
fn clip_polygon_to_near_into(
    polygon: &[[f32; 3]],
    yaw: f32,
    pitch: f32,
    roll: f32,
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
    output: &mut Vec<[f32; 3]>,
) {
    output.clear();
    if projection_mode == ProjectionMode::Orthographic {
        output.extend_from_slice(polygon);
        return;
    }
    for index in 0..polygon.len() {
        let current = polygon[index];
        let previous = polygon[(index + polygon.len() - 1) % polygon.len()];
        let current_distance =
            near_distance(current, yaw, pitch, roll, camera_target, perspective_unit);
        let previous_distance =
            near_distance(previous, yaw, pitch, roll, camera_target, perspective_unit);
        let near = PERSPECTIVE_NEAR * perspective_unit;
        let current_inside = current_distance >= near;
        let previous_inside = previous_distance >= near;
        if current_inside != previous_inside {
            let amount = (near - previous_distance) / (current_distance - previous_distance);
            output.push(interpolate_point(previous, current, amount));
        }
        if current_inside {
            output.push(current);
        }
    }
}

fn interpolate_preview_vertex(a: PreviewVertex, b: PreviewVertex, amount: f32) -> PreviewVertex {
    PreviewVertex {
        source_index: if amount < 0.5 {
            a.source_index
        } else {
            b.source_index
        },
        local_position: interpolate_point(a.local_position, b.local_position, amount),
        local_normal: interpolate_point(a.local_normal, b.local_normal, amount),
        position: interpolate_point(a.position, b.position, amount),
        normal: interpolate_point(a.normal, b.normal, amount),
        uv: [
            a.uv[0] + (b.uv[0] - a.uv[0]) * amount,
            a.uv[1] + (b.uv[1] - a.uv[1]) * amount,
        ],
        color: [0, 1, 2, 3]
            .map(|channel| a.color[channel] + (b.color[channel] - a.color[channel]) * amount),
    }
}

#[allow(clippy::too_many_arguments)]
fn clip_preview_polygon_to_near_into(
    triangle: &PreviewTriangle,
    yaw: f32,
    pitch: f32,
    roll: f32,
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
    output: &mut Vec<PreviewVertex>,
) {
    output.clear();
    if projection_mode == ProjectionMode::Orthographic {
        output.extend_from_slice(&triangle.vertices);
        return;
    }
    let polygon = &triangle.vertices;
    for index in 0..polygon.len() {
        let current = polygon[index];
        let previous = polygon[(index + polygon.len() - 1) % polygon.len()];
        let current_distance = near_distance(
            current.position,
            yaw,
            pitch,
            roll,
            camera_target,
            perspective_unit,
        );
        let previous_distance = near_distance(
            previous.position,
            yaw,
            pitch,
            roll,
            camera_target,
            perspective_unit,
        );
        let near = PERSPECTIVE_NEAR * perspective_unit;
        let current_inside = current_distance >= near;
        let previous_inside = previous_distance >= near;
        if current_inside != previous_inside {
            let amount = (near - previous_distance) / (current_distance - previous_distance);
            output.push(interpolate_preview_vertex(previous, current, amount));
        }
        if current_inside {
            output.push(current);
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn scene_ids_are_stable_and_unique() {
        let mut scene = EditorScene::default();
        let a = scene.add("A", ObjectKind::Empty, None);
        let b = scene.add("B", ObjectKind::Empty, None);
        assert_ne!(a, b);
        assert_eq!(scene.tree.node(a).unwrap().name, "A");
    }

    #[test]
    fn reparented_import_and_camera_hierarchy_survives_project_serialization() {
        let mut tree = GeometryTree::new();
        let imported = tree.create("Imported GLB", None).unwrap();
        tree.add_component(
            imported,
            Component::Model {
                asset: "assets/car.glb".into(),
                primitive: Some(0),
            },
        )
        .unwrap();
        let camera = tree.create("Camera", None).unwrap();
        tree.set_global_transform(
            camera,
            Transform {
                translation: CoreVec3::new(3.0, 4.0, 5.0),
                ..Transform::IDENTITY
            },
        )
        .unwrap();
        let camera_world = tree.node(camera).unwrap().global_transform();
        tree.reparent(camera, Some(imported), ReparentMode::KeepGlobal)
            .unwrap();

        let encoded = ProjectFile::new("project", "scene", tree)
            .to_json()
            .unwrap();
        let decoded = ProjectFile::from_json(&encoded).unwrap();
        assert_eq!(
            decoded.scene.geometry.node(camera).unwrap().parent(),
            Some(imported)
        );
        assert_eq!(
            decoded
                .scene
                .geometry
                .node(camera)
                .unwrap()
                .global_transform(),
            camera_world
        );
    }

    #[test]
    fn bundle_paths_are_safe_and_rewrite_only_referenced_models() {
        assert_eq!(safe_bundle_filename("my model (1).glb"), "my_model__1_.glb");
        let mut tree = GeometryTree::new();
        let model = tree.create("model", None).unwrap();
        tree.add_component(
            model,
            Component::Model {
                asset: "/source/model.glb".into(),
                primitive: None,
            },
        )
        .unwrap();
        let mut project = ProjectFile::new("project", "scene", tree);
        rewrite_asset_paths(
            &mut project,
            &BTreeMap::<String, String>::from([(
                "/source/model.glb".into(),
                "assets/0000/model.glb".into(),
            )]),
        );
        let node = project.scene.geometry.node(model).unwrap();
        assert!(matches!(
            &node.components[0],
            Component::Model { asset, .. } if asset == "assets/0000/model.glb"
        ));

        project.project.properties.insert(
            "compositor.image.0000".into(),
            "assets/compositor/0000-reference.png".into(),
        );
        rewrite_compositor_image_paths(
            &mut project,
            &BTreeMap::<String, PathBuf>::from([(
                "assets/compositor/0000-reference.png".into(),
                PathBuf::from("target/zfp-cache/test/reference.png"),
            )]),
        );
        assert_eq!(
            project.project.properties["compositor.image.0000"],
            "target/zfp-cache/test/reference.png"
        );
    }

    #[test]
    fn deleting_parent_removes_direct_children() {
        let mut scene = EditorScene {
            tree: GeometryTree::new(),
            selected: None,
        };
        let parent = scene.add("parent", ObjectKind::Empty, None);
        scene.add("child", ObjectKind::Model, Some(parent));
        scene.selected = Some(parent);
        scene.remove_selected();
        assert_eq!(scene.tree.iter().count(), 0);
    }

    #[test]
    fn projection_is_finite() {
        let position = project(
            [1.0, 2.0, 3.0],
            Pos2::ZERO,
            100.0,
            0.3,
            0.2,
            0.0,
            CoreVec3::ZERO,
            ProjectionMode::Perspective,
            1.0,
        )
        .expect("point is in front of near plane");
        assert!(position.x.is_finite() && position.y.is_finite());
    }

    #[test]
    fn orthographic_projection_is_depth_independent() {
        let near = project(
            [2.0, 1.0, 3.0],
            Pos2::ZERO,
            100.0,
            0.0,
            0.0,
            0.0,
            CoreVec3::ZERO,
            ProjectionMode::Orthographic,
            1.0,
        );
        let far = project(
            [2.0, 100.0, 3.0],
            Pos2::ZERO,
            100.0,
            0.0,
            0.0,
            0.0,
            CoreVec3::ZERO,
            ProjectionMode::Orthographic,
            1.0,
        );
        assert_eq!(near, far);
    }

    #[test]
    fn perspective_rejects_and_clips_camera_plane_geometry() {
        let behind = [0.0, -PERSPECTIVE_CAMERA_DISTANCE - 1.0, 0.0];
        assert!(
            project(
                behind,
                Pos2::ZERO,
                100.0,
                0.0,
                0.0,
                0.0,
                CoreVec3::ZERO,
                ProjectionMode::Perspective,
                1.0,
            )
            .is_none()
        );
        let crossing = clip_segment_to_near(
            [behind, [0.0, 1.0, 0.0]],
            0.0,
            0.0,
            0.0,
            CoreVec3::ZERO,
            ProjectionMode::Perspective,
            1.0,
        )
        .expect("crossing segment survives");
        assert!(
            (near_distance(crossing[0], 0.0, 0.0, 0.0, CoreVec3::ZERO, 1.0) - PERSPECTIVE_NEAR)
                .abs()
                < 1.0e-4
        );
    }

    #[test]
    fn near_clipped_triangle_produces_a_finite_quad() {
        let polygon = clip_polygon_to_near(
            &[
                [0.0, -PERSPECTIVE_CAMERA_DISTANCE - 1.0, 0.0],
                [-1.0, 1.0, 0.0],
                [1.0, 1.0, 0.0],
            ],
            0.0,
            0.0,
            0.0,
            CoreVec3::ZERO,
            ProjectionMode::Perspective,
            1.0,
        );
        assert_eq!(polygon.len(), 4);
        assert!(polygon.iter().all(|point| {
            near_distance(*point, 0.0, 0.0, 0.0, CoreVec3::ZERO, 1.0) >= PERSPECTIVE_NEAR - 1.0e-4
        }));
    }

    #[test]
    fn middle_mouse_pan_moves_target_in_view_plane() {
        let target = pan_camera_target(CoreVec3::ZERO, Vec2::new(20.0, -10.0), 10.0, 0.0, 0.0);
        assert!(target.approx_eq(CoreVec3::new(-2.0, 0.0, -1.0), 1.0e-5));
    }

    #[test]
    fn projection_does_not_change_source_aspect_ratio() {
        for mode in [ProjectionMode::Orthographic, ProjectionMode::Perspective] {
            let origin = project(
                [0.0, 0.0, 0.0],
                Pos2::ZERO,
                100.0,
                0.0,
                0.0,
                0.0,
                CoreVec3::ZERO,
                mode,
                1.0,
            )
            .unwrap();
            let x = project(
                [1.0, 0.0, 0.0],
                Pos2::ZERO,
                100.0,
                0.0,
                0.0,
                0.0,
                CoreVec3::ZERO,
                mode,
                1.0,
            )
            .unwrap();
            let z = project(
                [0.0, 0.0, 1.0],
                Pos2::ZERO,
                100.0,
                0.0,
                0.0,
                0.0,
                CoreVec3::ZERO,
                mode,
                1.0,
            )
            .unwrap();
            assert!(((x - origin).length() - (z - origin).length()).abs() < 1.0e-5);
        }
    }

    #[test]
    fn grid_spacing_uses_longest_edge_order_of_magnitude() {
        let asset = MeshAsset {
            vertices: vec![
                zerofps_assets::Vertex {
                    position: [0.0, 0.0, 0.0],
                    ..Default::default()
                },
                zerofps_assets::Vertex {
                    position: [63.0, 0.0, 0.0],
                    ..Default::default()
                },
                zerofps_assets::Vertex {
                    position: [0.0, 1.0, 0.0],
                    ..Default::default()
                },
            ],
            primitives: vec![zerofps_assets::Primitive {
                indices: vec![0, 1, 2],
                ..Default::default()
            }],
            ..Default::default()
        };
        assert_eq!(infer_grid_spacing(&asset), Some(100.0));
        assert_eq!(infer_grid_spacing(&MeshAsset::default()), None);
    }

    #[test]
    fn builtin_primitives_are_valid_non_degenerate_z_up_meshes() {
        for (kind, triangles) in [
            (BuiltinPrimitive::Cube, 12),
            (BuiltinPrimitive::Sphere, 720),
            (BuiltinPrimitive::Floor, 2),
        ] {
            let mesh = builtin_primitive_mesh(kind);
            mesh.validate().expect("built-in mesh must be canonical");
            assert_eq!(mesh.triangle_count(), triangles);
            assert_eq!(mesh.source.up_axis, zerofps_assets::AxisConvention::ZUp);
            for primitive in &mesh.primitives {
                for triangle in primitive.indices.chunks_exact(3) {
                    let points = [
                        mesh.vertices[triangle[0] as usize].position,
                        mesh.vertices[triangle[1] as usize].position,
                        mesh.vertices[triangle[2] as usize].position,
                    ];
                    let ab = CoreVec3::new(
                        points[1][0] - points[0][0],
                        points[1][1] - points[0][1],
                        points[1][2] - points[0][2],
                    );
                    let ac = CoreVec3::new(
                        points[2][0] - points[0][0],
                        points[2][1] - points[0][1],
                        points[2][2] - points[0][2],
                    );
                    assert!(
                        ab.cross(ac).length() > 1.0e-6,
                        "{} contains a degenerate triangle",
                        kind.label()
                    );
                }
            }
        }
    }

    #[test]
    fn builtin_assets_have_stable_non_file_paths() {
        let assets = builtin_imported_assets();
        assert_eq!(assets.len(), 3);
        assert_eq!(
            assets
                .iter()
                .map(|asset| asset.path.as_str())
                .collect::<Vec<_>>(),
            ["builtin:cube", "builtin:sphere", "builtin:floor"]
        );
    }

    #[test]
    fn collider_shapes_best_fit_bounds_with_editable_parameters() {
        let bounds = ([-2.0, -1.0, -0.5], [2.0, 3.0, 1.5]);
        let collider = |shape| fitted_collider_from_bounds(bounds, shape);
        let Component::Collider {
            center,
            half_extents,
            ..
        } = collider(ColliderShape::Box)
        else {
            panic!("box collider");
        };
        assert_eq!(center, CoreVec3::new(0.0, 1.0, 0.5));
        assert_eq!(half_extents, CoreVec3::new(2.0, 2.0, 1.0));

        let Component::Collider { radius, .. } = collider(ColliderShape::Sphere) else {
            panic!("sphere collider");
        };
        assert_eq!(radius, 2.0);

        let Component::Collider { radius, height, .. } = collider(ColliderShape::Cylinder) else {
            panic!("cylinder collider");
        };
        assert_eq!((radius, height), (2.0, 2.0));

        let Component::Collider {
            half_extents,
            height,
            ..
        } = collider(ColliderShape::Flat)
        else {
            panic!("flat collider");
        };
        assert_eq!(half_extents, CoreVec3::new(2.0, 2.0, 0.005));
        assert_eq!(height, 0.01);
    }

    #[test]
    fn arbitrary_collider_pairs_report_separating_contact() {
        let collider = |center, half_extents, shape| WorldCollider {
            center,
            half_extents,
            restitution: 0.0,
            friction: 0.8,
            friction_margin_percent: 5.0,
            shape,
            joint: CylinderJoint::None,
        };
        let car = collider(
            CoreVec3::new(0.0, 0.0, 1.0),
            CoreVec3::new(1.0, 0.5, 0.5),
            ColliderShape::Box,
        );
        let circuit = collider(
            CoreVec3::ZERO,
            CoreVec3::new(45.0, 32.5, 0.6),
            ColliderShape::Box,
        );
        let contact = collider_pair_contact(car, circuit).expect("car intersects circuit surface");
        assert_eq!(contact.normal, CoreVec3::Z);
        assert!((contact.penetration - 0.1).abs() < 1.0e-6);
        let separated = WorldCollider {
            center: CoreVec3::new(0.0, 0.0, 2.0),
            ..car
        };
        assert!(collider_pair_contact(separated, circuit).is_none());
    }

    #[test]
    fn general_contact_stops_inward_motion_and_corrects_penetration() {
        let body = WorldCollider {
            center: CoreVec3::new(0.0, 0.0, 0.4),
            half_extents: CoreVec3::new(0.5, 0.5, 0.5),
            restitution: 0.0,
            friction: 0.8,
            friction_margin_percent: 5.0,
            shape: ColliderShape::Sphere,
            joint: CylinderJoint::None,
        };
        let floor = WorldCollider {
            center: CoreVec3::ZERO,
            half_extents: CoreVec3::new(10.0, 10.0, 0.05),
            restitution: 0.0,
            friction: 0.8,
            friction_margin_percent: 5.0,
            shape: ColliderShape::Flat,
            joint: CylinderJoint::None,
        };
        let mut position = CoreVec3::ZERO;
        let mut velocity = CoreVec3::new(2.0, 0.0, -3.0);
        assert_eq!(
            resolve_general_collider_contacts(
                &mut position,
                &mut velocity,
                CoreVec3::ZERO,
                &[body],
                &[floor],
                1.0 / 60.0,
            ),
            1
        );
        assert!(position.z > 0.0);
        assert!(velocity.z >= 0.0);
        assert!(velocity.x < 2.0);
    }

    #[test]
    fn friction_margin_bridges_small_gaps_without_collision_correction() {
        let body = WorldCollider {
            center: CoreVec3::new(0.0, 0.0, 0.56),
            half_extents: CoreVec3::new(0.5, 0.5, 0.5),
            restitution: 0.0,
            friction: 1.0,
            friction_margin_percent: 5.0,
            shape: ColliderShape::Sphere,
            joint: CylinderJoint::None,
        };
        let floor = WorldCollider {
            center: CoreVec3::ZERO,
            half_extents: CoreVec3::new(10.0, 10.0, 0.05),
            restitution: 0.0,
            friction: 1.0,
            friction_margin_percent: 5.0,
            shape: ColliderShape::Flat,
            joint: CylinderJoint::None,
        };
        assert!(collider_pair_contact(body, floor).is_none());
        let mut position = CoreVec3::ZERO;
        let mut velocity = CoreVec3::new(2.0, 0.0, 0.0);
        assert_eq!(
            resolve_general_collider_contacts(
                &mut position,
                &mut velocity,
                CoreVec3::ZERO,
                &[body],
                &[floor],
                1.0 / 60.0,
            ),
            1
        );
        assert_eq!(position, CoreVec3::ZERO);
        assert!(velocity.x < 2.0);
    }

    #[test]
    fn wheel_contact_margin_scales_with_vertical_extent() {
        let tolerance = friction_contact_tolerance(0.43, 5.0);
        assert!((tolerance - 0.0215).abs() < 1.0e-6);
        assert!(0.02 <= tolerance);
        assert!(0.03 > tolerance);
    }

    #[test]
    fn two_driven_cylinders_move_four_wheel_vehicle_over_box() {
        let floor = WorldCollider {
            center: CoreVec3::ZERO,
            half_extents: CoreVec3::new(10.0, 10.0, 0.05),
            restitution: 0.0,
            friction: 1.0,
            friction_margin_percent: 5.0,
            shape: ColliderShape::Box,
            joint: CylinderJoint::None,
        };
        let wheel = |x: f32, y: f32| WorldCollider {
            // A 0.01-unit gap deliberately exercises the friction margin.
            center: CoreVec3::new(x, y, 0.56),
            half_extents: CoreVec3::new(0.2, 0.5, 0.5),
            restitution: 0.0,
            friction: 1.0,
            friction_margin_percent: 5.0,
            shape: ColliderShape::Cylinder,
            joint: CylinderJoint::Engine,
        };
        let wheels = [
            wheel(-0.8, 1.0),
            wheel(0.8, 1.0),
            wheel(-0.8, -1.0),
            wheel(0.8, -1.0),
        ];
        let axles = [CoreVec3::X, CoreVec3::new(-1.0, 0.0, 0.0)];
        // Mirrored axles require mirrored signed angular velocity to produce
        // the same forward tread velocity on both driven wheels.
        let driven_angular_velocities = [20.0, -20.0];
        let mut total_body_force = CoreVec3::ZERO;
        for index in 0..2 {
            total_body_force = total_body_force
                + cylinder_box_traction_force(
                    wheels[index],
                    floor,
                    axles[index],
                    driven_angular_velocities[index],
                    CoreVec3::ZERO,
                    100.0,
                    25.0,
                    1.0 / 60.0,
                )
                .expect("driven wheel should retain tolerated box contact");
        }
        for wheel in &wheels[2..] {
            let passive_force = cylinder_box_traction_force(
                *wheel,
                floor,
                CoreVec3::X,
                0.0,
                CoreVec3::ZERO,
                100.0,
                25.0,
                1.0 / 60.0,
            )
            .expect("passive wheel should retain tolerated box contact");
            assert_eq!(passive_force, CoreVec3::ZERO);
        }
        assert!(total_body_force.y < 0.0);
        assert!(total_body_force.x.abs() < 1.0e-6);

        let mut position = CoreVec3::ZERO;
        let mut velocity_after_traction = total_body_force * (1.0 / 60.0 / 100.0);
        let expected_velocity = velocity_after_traction;
        resolve_general_collider_contacts(
            &mut position,
            &mut velocity_after_traction,
            CoreVec3::ZERO,
            &wheels[..2],
            &[floor],
            1.0 / 60.0,
        );
        assert!(
            velocity_after_traction.approx_eq(expected_velocity, 1.0e-6),
            "generic friction must not cancel joint-aware wheel traction"
        );
    }

    #[test]
    fn fast_driver_zfp_driven_wheels_generate_box_traction() {
        let fixture = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../appdata/models/Fast-Driver/game_v0.zfp");
        assert!(
            fixture.is_file(),
            "missing test fixture: {}",
            fixture.display()
        );
        let cache = PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("../../target/test-zfp-cache/fast-driver-traction");
        let bundle = load_zfp(&fixture, &cache).expect("load Fast-Driver project");
        let tree = &bundle.project.scene.geometry;
        let find = |name: &str| {
            tree.iter()
                .find(|(_, node)| node.name == name)
                .map(|(id, node)| (id, node))
                .unwrap_or_else(|| panic!("missing `{name}` in Fast-Driver fixture"))
        };
        let (_, circuit_node) = find("Circuit");
        let circuit = circuit_node
            .components
            .iter()
            .find_map(|component| world_collider(circuit_node.global_transform(), component))
            .expect("Circuit box collider");
        assert_eq!(circuit.shape, ColliderShape::Box);

        let properties = &bundle.project.project.properties;
        let engine_objects = properties
            .iter()
            .filter_map(|(key, value)| {
                (key.ends_with(".kind") && value == "29").then(|| {
                    key.trim_start_matches("compositor.node.")
                        .trim_end_matches(".kind")
                        .to_owned()
                })
            })
            .filter_map(|node_id| {
                properties
                    .get(&format!("compositor.node.{node_id}.settings_object_name"))
                    .cloned()
            })
            .collect::<Vec<_>>();
        assert_eq!(engine_objects.len(), 2);

        let mut total_traction = CoreVec3::ZERO;
        let mut saved_contact_gaps = Vec::new();
        for wheel_name in &engine_objects {
            let (_, wheel_node) = find(wheel_name);
            let transform = wheel_node.global_transform();
            let mut wheel = wheel_node
                .components
                .iter()
                .find_map(|component| world_collider(transform, component))
                .expect("engine wheel cylinder collider");
            assert_eq!(wheel.shape, ColliderShape::Cylinder);
            let tolerance =
                friction_contact_tolerance(wheel.half_extents.z, wheel.friction_margin_percent);
            let axle = transform.rotation.rotate(CoreVec3::Z).normalized();
            let circuit_top = circuit.center.z + circuit.half_extents.z;
            let saved_gap = wheel.center.z - wheel.half_extents.z - circuit_top;
            saved_contact_gaps.push((saved_gap, tolerance));
            assert!(
                cylinder_box_traction_force(
                    wheel,
                    circuit,
                    axle,
                    20.0,
                    CoreVec3::ZERO,
                    100.0,
                    25.0,
                    1.0 / 60.0,
                )
                .is_none(),
                "saved wheel unexpectedly contacts the saved Circuit"
            );
            wheel.center.z =
                circuit.center.z + circuit.half_extents.z + wheel.half_extents.z + tolerance * 0.5;
            total_traction = total_traction
                + cylinder_box_traction_force(
                    wheel,
                    circuit,
                    axle,
                    20.0,
                    CoreVec3::ZERO,
                    100.0,
                    25.0,
                    1.0 / 60.0,
                )
                .expect("engine wheel should contact Circuit through tolerance");
        }
        assert!(
            total_traction.length() > 1.0,
            "the two fixture engines must not cancel each other's traction"
        );
        assert!(
            saved_contact_gaps
                .iter()
                .all(|(gap, tolerance)| gap > tolerance)
        );
    }

    #[test]
    fn box_collider_exposes_bounded_surface_for_wheel_friction() {
        let circuit = WorldCollider {
            center: CoreVec3::new(0.0, 0.0, -0.2),
            half_extents: CoreVec3::new(45.0, 32.5, 0.2),
            restitution: 0.0,
            friction: 0.8,
            friction_margin_percent: 5.0,
            shape: ColliderShape::Box,
            joint: CylinderJoint::None,
        };
        let (height, friction, tolerance) =
            collider_support_surface(CoreVec3::new(34.0, 0.0, 2.0), &[circuit]).unwrap();
        assert_eq!((height, friction), (0.0, 0.8));
        assert!((tolerance - 0.01).abs() < 1.0e-6);
        assert!(collider_support_surface(CoreVec3::new(50.0, 0.0, 2.0), &[circuit]).is_none());
    }

    #[test]
    fn collider_wireframes_cover_every_supported_shape() {
        let center = CoreVec3::new(3.0, -2.0, 1.0);
        let extents = CoreVec3::new(2.0, 1.0, 0.5);
        let box_lines = collider_wireframe_segments(ColliderShape::Box, center, extents, 2.0, 1.0);
        let flat_lines =
            collider_wireframe_segments(ColliderShape::Flat, center, extents, 2.0, 0.01);
        let sphere_lines =
            collider_wireframe_segments(ColliderShape::Sphere, center, extents, 2.0, 1.0);
        let cylinder_lines =
            collider_wireframe_segments(ColliderShape::Cylinder, center, extents, 2.0, 1.0);

        assert_eq!(box_lines.len(), 12);
        assert_eq!(flat_lines.len(), 12);
        assert_eq!(sphere_lines.len(), 96);
        assert_eq!(cylinder_lines.len(), 72);
        assert!(
            box_lines
                .into_iter()
                .flatten()
                .flatten()
                .all(f32::is_finite)
        );
        assert!(
            sphere_lines
                .into_iter()
                .flatten()
                .flatten()
                .all(f32::is_finite)
        );
        assert!(
            cylinder_lines
                .into_iter()
                .flatten()
                .flatten()
                .all(f32::is_finite)
        );
    }

    #[test]
    fn collider_volume_supports_automatic_component_mass() {
        let extents = CoreVec3::new(1.0, 2.0, 3.0);
        assert_eq!(collider_volume(ColliderShape::Box, extents, 0.0, 0.0), 48.0);
        assert_eq!(
            collider_volume(ColliderShape::Flat, extents, 0.0, 0.0),
            48.0
        );
        assert!(
            (collider_volume(ColliderShape::Sphere, extents, 1.0, 0.0)
                - 4.0 * std::f32::consts::PI / 3.0)
                .abs()
                < 1.0e-5
        );
        assert!(
            (collider_volume(ColliderShape::Cylinder, extents, 2.0, 3.0)
                - 12.0 * std::f32::consts::PI)
                .abs()
                < 1.0e-5
        );
    }

    #[test]
    fn force_cut_filter_removes_small_vibration_signals() {
        assert_eq!(force_cut_scalar(0.01, 0.01), 0.0);
        assert_eq!(force_cut_scalar(-0.009, 0.01), 0.0);
        assert_eq!(force_cut_scalar(0.02, 0.01), 0.02);
        assert_eq!(
            force_cut_vector(CoreVec3::new(0.003, 0.004, 0.0), 0.01),
            CoreVec3::ZERO
        );
        assert_eq!(
            force_cut_vector(CoreVec3::new(0.02, 0.0, 0.0), 0.01),
            CoreVec3::new(0.02, 0.0, 0.0)
        );
    }

    #[test]
    fn collider_friction_defaults_distinguish_rolling_from_sliding() {
        let low = CoreVec3::new(0.05, 0.8, 0.8);
        // Cylinder axis is world Y; rolling is therefore world X.
        let rotation = Quat::from_euler_xyz(CoreVec3::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0));
        let rolling = collider_friction_coefficient(
            ColliderShape::Cylinder,
            rotation,
            low,
            CoreVec3::new(1.0, 0.0, 0.0),
        );
        let axial = collider_friction_coefficient(
            ColliderShape::Cylinder,
            rotation,
            low,
            CoreVec3::new(0.0, 1.0, 0.0),
        );
        assert!(rolling < axial);
        assert_eq!(
            collider_friction_coefficient(
                ColliderShape::Sphere,
                Quat::IDENTITY,
                CoreVec3::new(0.05, 0.05, 0.05),
                CoreVec3::new(1.0, 1.0, 0.0),
            ),
            0.05
        );
    }

    #[test]
    fn contact_friction_opposes_slip_and_respects_coulomb_limit() {
        let force =
            contact_friction_force(CoreVec3::new(4.0, 0.0, 0.0), 0.5, 100.0, 10.0, 1.0 / 60.0);
        assert!(force.x < 0.0);
        assert!(force.y.abs() <= f32::EPSILON);
        assert!((force.length() - 50.0).abs() < 1.0e-4);
    }

    #[test]
    fn contact_friction_stops_small_slip_without_reversing_it() {
        let dt = 0.1;
        let mass = 2.0;
        let velocity = CoreVec3::new(0.2, -0.1, 0.0);
        let force = contact_friction_force(velocity, 10.0, 100.0, mass, dt);
        let next = velocity + force * (dt / mass);
        assert!(next.length() < 1.0e-5);
    }

    #[test]
    fn impulse_contact_is_non_bouncing_by_default() {
        assert_eq!(resolve_contact_normal_velocity(-12.0, 0.0), 0.0);
        assert_eq!(resolve_contact_normal_velocity(-12.0, 0.25), 3.0);
        assert_eq!(resolve_contact_normal_velocity(2.0, 0.0), 2.0);
        assert_eq!(contact_count_for_solver(0), (1.0, false));
        assert_eq!(contact_count_for_solver(4), (4.0, true));
    }

    #[test]
    fn wheel_contact_uses_coupled_body_and_rotational_mass() {
        let wheel_mass = 196.0;
        let radius = 0.43;
        let inertia = 0.5 * wheel_mass * radius * radius;
        let rolling = coupled_contact_mass(wheel_mass, inertia, radius, 1.0);
        let lateral = coupled_contact_mass(wheel_mass, inertia, radius, 0.0);
        assert!((rolling - wheel_mass / 3.0).abs() < 1.0e-4);
        assert!((lateral - wheel_mass).abs() < 1.0e-4);

        // An uncapped rolling impulse computed from this mass removes exactly
        // the relative slip shared by body translation and wheel rotation.
        let slip = 2.0;
        let impulse = rolling * slip;
        let removed_slip = impulse * (1.0 / wheel_mass + radius * radius / inertia);
        assert!((removed_slip - slip).abs() < 1.0e-4);
    }

    #[test]
    fn cylinder_ground_extent_is_invariant_under_axle_rotation() {
        let half_extents = CoreVec3::new(0.43, 0.43, 0.17);
        let wheel_orientation =
            Quat::from_euler_xyz(CoreVec3::new(-std::f32::consts::FRAC_PI_2, 0.0, 0.0));
        let expected = collider_vertical_extent(
            ColliderShape::Cylinder,
            half_extents,
            CoreVec3::new(1.0, 1.0, 1.0),
            wheel_orientation,
        );
        assert!((expected - 0.43).abs() < 1.0e-5);
        for step in 0..32 {
            let spin = step as f32 * std::f32::consts::TAU / 32.0;
            let rotation = wheel_orientation * Quat::from_euler_xyz(CoreVec3::new(0.0, 0.0, spin));
            let extent = collider_vertical_extent(
                ColliderShape::Cylinder,
                half_extents,
                CoreVec3::new(1.0, 1.0, 1.0),
                rotation,
            );
            assert!((extent - expected).abs() < 1.0e-5);
        }
    }

    #[test]
    fn orthographic_grid_segment_matches_direct_projection() {
        let segment = [[-2.0, 1.0, 0.0], [4.0, 1.0, 0.0]];
        let projected = project_segment(
            segment,
            Pos2::ZERO,
            100.0,
            -0.4,
            0.5,
            0.0,
            CoreVec3::ZERO,
            ProjectionMode::Orthographic,
            1.0,
        )
        .expect("line is visible");
        for index in 0..2 {
            let direct = project(
                segment[index],
                Pos2::ZERO,
                100.0,
                -0.4,
                0.5,
                0.0,
                CoreVec3::ZERO,
                ProjectionMode::Orthographic,
                1.0,
            )
            .expect("endpoint is visible");
            assert!(projected[index].distance(direct) < 1.0e-5);
        }
    }

    #[test]
    fn analytical_vertical_line_clips_to_viewport_edges() {
        let rect = Rect::from_min_max(Pos2::new(0.0, 0.0), Pos2::new(100.0, 80.0));
        let clipped =
            clip_infinite_line_to_rect([1.0, 0.0, -40.0], rect).expect("line crosses viewport");
        assert_eq!(clipped, [Pos2::new(40.0, 0.0), Pos2::new(40.0, 80.0)]);
    }

    #[test]
    fn perspective_grid_line_definition_is_independent_of_patch_radius() {
        let small = canonical_ground_line([[10.0, -20.0, 0.0], [10.0, 20.0, 0.0]]);
        let huge =
            canonical_ground_line([[10.0, -2_000_000_000.0, 0.0], [10.0, 2_000_000_000.0, 0.0]]);
        assert_eq!(small, huge);
        assert_eq!(small, [[10.0, 0.0, 0.0], [10.0, 1.0, 0.0]]);
    }

    #[test]
    fn projective_grid_is_an_analytical_homogeneous_line() {
        let segment = [[-4.0, 2.0, 0.0], [5.0, 2.0, 0.0]];
        let center = Pos2::new(400.0, 325.0);
        let first = homogeneous_projection(
            segment[0],
            center,
            100.0,
            -0.4,
            0.5,
            CoreVec3::ZERO,
            ProjectionMode::Perspective,
            1.0,
        );
        let second = homogeneous_projection(
            segment[1],
            center,
            100.0,
            -0.4,
            0.5,
            CoreVec3::ZERO,
            ProjectionMode::Perspective,
            1.0,
        );
        let line = cross_homogeneous(first, second);
        for amount in [0.0, 0.25, 0.5, 0.75, 1.0] {
            let point = homogeneous_projection(
                interpolate_point(segment[0], segment[1], amount),
                center,
                100.0,
                -0.4,
                0.5,
                CoreVec3::ZERO,
                ProjectionMode::Perspective,
                1.0,
            );
            let incidence = line[0] * point[0] + line[1] * point[1] + line[2] * point[2];
            let scale = line.iter().map(|value| value.abs()).sum::<f32>()
                * point.iter().map(|value| value.abs()).sum::<f32>();
            assert!(incidence.abs() / scale < 1.0e-6);
        }
    }

    #[test]
    fn perspective_projection_is_not_affine() {
        let center = Pos2::ZERO;
        let a = [1.0, 0.0, 0.0];
        let b = [1.0, 10.0, 0.0];
        let midpoint = interpolate_point(a, b, 0.5);
        let project_point = |point| {
            project(
                point,
                center,
                100.0,
                0.0,
                0.5,
                0.0,
                CoreVec3::ZERO,
                ProjectionMode::Perspective,
                1.0,
            )
            .expect("test point is visible")
        };
        let projected_affine_midpoint = project_point(a).lerp(project_point(b), 0.5);
        let projected_true_midpoint = project_point(midpoint);
        assert!(projected_affine_midpoint.distance(projected_true_midpoint) > 0.1);
    }

    #[test]
    fn perspective_foreshortens_model_geometry_with_depth() {
        let projected_width = |depth| {
            let left = project(
                [-1.0, depth, 0.0],
                Pos2::ZERO,
                100.0,
                0.0,
                0.0,
                0.0,
                CoreVec3::ZERO,
                ProjectionMode::Perspective,
                1.0,
            )
            .unwrap();
            let right = project(
                [1.0, depth, 0.0],
                Pos2::ZERO,
                100.0,
                0.0,
                0.0,
                0.0,
                CoreVec3::ZERO,
                ProjectionMode::Perspective,
                1.0,
            )
            .unwrap();
            left.distance(right)
        };
        assert!(projected_width(10.0) < projected_width(0.0));
    }

    #[test]
    fn perspective_camera_scales_with_grid_size() {
        let unit = project(
            [2.0, 3.0, 1.0],
            Pos2::ZERO,
            perspective_view_scale(100.0, ProjectionMode::Perspective, 1.0),
            -0.4,
            0.5,
            0.0,
            CoreVec3::ZERO,
            ProjectionMode::Perspective,
            1.0,
        )
        .expect("unit-scale point is visible");
        let hundred = project(
            [200.0, 300.0, 100.0],
            Pos2::ZERO,
            perspective_view_scale(100.0, ProjectionMode::Perspective, 100.0),
            -0.4,
            0.5,
            0.0,
            CoreVec3::ZERO,
            ProjectionMode::Perspective,
            100.0,
        )
        .expect("scaled point is visible");
        assert!(unit.distance(hundred) < 1.0e-4);
        assert_eq!(
            perspective_view_scale(100.0, ProjectionMode::Orthographic, 100.0),
            100.0
        );
    }

    #[test]
    fn ray_triangle_returns_nearest_linear_distance() {
        let triangle = [
            CoreVec3::new(-1.0, 5.0, -1.0),
            CoreVec3::new(1.0, 5.0, -1.0),
            CoreVec3::new(0.0, 5.0, 1.0),
        ];
        let distance = ray_triangle_distance(CoreVec3::ZERO, CoreVec3::Y, triangle)
            .expect("center ray intersects triangle");
        assert!((distance - 5.0).abs() < 1.0e-5);
    }

    #[test]
    fn unambiguous_face_uses_polygon_coverage_and_plane_depth() {
        let polygon = [
            Pos2::new(0.0, 0.0),
            Pos2::new(10.0, 0.0),
            Pos2::new(5.0, 10.0),
        ];
        assert!(point_in_convex_polygon(Pos2::new(5.0, 5.0), &polygon));
        assert!(!point_in_convex_polygon(Pos2::new(9.0, 9.0), &polygon));
        let triangle = [
            CoreVec3::new(-1.0, 5.0, -1.0),
            CoreVec3::new(1.0, 5.0, -1.0),
            CoreVec3::new(0.0, 5.0, 1.0),
        ];
        let plane = ray_plane_distance(CoreVec3::ZERO, CoreVec3::Y, triangle)
            .expect("ray crosses face plane");
        let exact = ray_triangle_distance(CoreVec3::ZERO, CoreVec3::Y, triangle)
            .expect("ray crosses triangle");
        assert!((plane - exact).abs() < 1.0e-6);
    }

    #[test]
    fn rasterized_frame_produces_color_and_reusable_depth() {
        let triangle = |positions: [[f32; 3]; 3]| PreviewTriangle {
            object_id: NodeId {
                slot: 0,
                generation: 0,
            },
            object_transform: Transform::IDENTITY,
            vertices: positions.map(|position| PreviewVertex {
                source_index: 0,
                local_position: position,
                local_normal: [0.0, 0.0, 1.0],
                position,
                normal: [0.0, 0.0, 1.0],
                uv: [0.0; 2],
                color: [1.0; 4],
            }),
            base_color: [0.42, 0.64, 0.78, 1.0],
            source_base_color: [0.42, 0.64, 0.78, 1.0],
            texture: None,
            texture_cache_key: 0,
            gpu_texture: None,
            source_texture: None,
            shader: ShaderMode::Toon,
            smooth_normals: false,
            transmission: 0.0,
            ior: 1.5,
            casts_shadows: true,
        };
        let triangles = [
            triangle([[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [0.0, 0.0, 2.0]]),
            triangle([[-2.0, 5.0, -2.0], [2.0, 5.0, -2.0], [0.0, 5.0, 2.0]]),
        ];
        let original_batches = build_vulkan_batches(&triangles);
        let mut deformed = triangles.clone();
        deformed[0].vertices[0].local_position[0] += 0.25;
        deformed[0].object_transform.translation.x += 2.0;
        let deformed_batches = build_vulkan_batches(&deformed);
        assert_eq!(original_batches.len(), deformed_batches.len());
        assert_eq!(
            original_batches[0].cache_key, deformed_batches[0].cache_key,
            "deformation and transforms must update a stable GPU buffer"
        );
        assert_ne!(
            original_batches[0].vertices[0].position,
            deformed_batches[0].vertices[0].position
        );
        let camera = (
            0.0,
            0.0,
            0.0,
            1.0,
            CoreVec3::ZERO,
            1.0,
            ProjectionMode::Perspective,
        );
        let mut workspace = RasterWorkspace::default();
        let lighting = ViewportLighting {
            global_enabled: true,
            global_shadow_resolution: 0,
            shadow_filter_radius: 1,
            shadow_blur_radius: 0,
            points: Vec::new(),
            directional_shadow: None,
            point_shadows: None,
        };
        let frame = rasterize_depth_frame(
            Vec2::new(64.0, 64.0),
            &triangles,
            camera,
            &lighting,
            Vec::new(),
            &mut workspace,
        );
        let reversed = rasterize_depth_frame(
            Vec2::new(64.0, 64.0),
            &[triangles[1].clone(), triangles[0].clone()],
            camera,
            &lighting,
            Vec::new(),
            &mut workspace,
        );
        assert_eq!(frame.linear_depth.len(), 64 * 64);
        assert_eq!(frame.linear_depth, reversed.linear_depth);
        let (FrameColor::Cpu(frame_color), FrameColor::Cpu(reversed_color)) =
            (&frame.color, &reversed.color)
        else {
            panic!("CPU rasterizer returned a non-CPU frame");
        };
        assert_eq!(frame_color, reversed_color);
        let center = 57 * 64 + 32;
        assert!(frame.linear_depth[center].is_finite());
        assert!((frame.linear_depth[center] - PERSPECTIVE_CAMERA_DISTANCE).abs() < 0.1);
        assert_ne!(frame_color.pixels[center], Color32::TRANSPARENT);
    }

    #[test]
    fn nearest_texture_sampling_repeats_uvs() {
        let texture = TextureAsset {
            name: "fixture".into(),
            width: 2,
            height: 2,
            pixels: vec![
                255, 0, 0, 255, 0, 255, 0, 255, 0, 0, 255, 255, 255, 255, 255, 255,
            ],
            cached_mips: Vec::new(),
        };
        assert_eq!(
            sample_texture_nearest(&texture, [0.1, 0.1]),
            [1.0, 0.0, 0.0, 1.0]
        );
        assert_eq!(
            sample_texture_nearest(&texture, [1.6, 0.1]),
            [0.0, 1.0, 0.0, 1.0]
        );
    }

    #[test]
    fn cpu_texture_lod_tracks_projected_texel_density() {
        let vertex = |position: Pos2, uv: [f32; 2]| RasterVertex {
            position,
            camera_depth: 1.0,
            world_position: [0.0; 3],
            normal: [0.0, 0.0, 1.0],
            uv,
            color: [1.0; 4],
        };
        let texture = TextureAsset {
            name: "large".into(),
            width: 1024,
            height: 1024,
            pixels: vec![255; 1024 * 1024 * 4],
            cached_mips: Vec::new(),
        };
        let close = [
            vertex(Pos2::new(0.0, 0.0), [0.0, 0.0]),
            vertex(Pos2::new(1024.0, 0.0), [1.0, 0.0]),
            vertex(Pos2::new(0.0, 1024.0), [0.0, 1.0]),
        ];
        let distant = [
            vertex(Pos2::new(0.0, 0.0), [0.0, 0.0]),
            vertex(Pos2::new(64.0, 0.0), [1.0, 0.0]),
            vertex(Pos2::new(0.0, 64.0), [0.0, 1.0]),
        ];
        assert_eq!(triangle_texture_lod(close, &texture, 3), 0);
        assert_eq!(triangle_texture_lod(distant, &texture, 3), 2);
    }

    #[test]
    fn shader_modes_quantize_only_toon_lighting() {
        let diffuse = shader_light_factor(0.42, ShaderMode::Diffuse);
        let toon = shader_light_factor(0.42, ShaderMode::Toon);
        assert!((diffuse - 0.565).abs() < 1.0e-6);
        assert_eq!(toon, 2.0 / 3.0);
        assert_ne!(toon, diffuse);
    }

    #[test]
    fn material_shader_is_an_inherited_object_attribute() {
        let mut tree = GeometryTree::new();
        tree.set_default(
            material_shader_key(),
            Attribute::Text(ShaderMode::Toon.attribute_value().into()),
        );
        let parent = tree.create("parent", None).unwrap();
        let child = tree.create("child", Some(parent)).unwrap();
        tree.set_attribute(
            parent,
            material_shader_key(),
            AttributeDeclaration::Value(Attribute::Text(
                ShaderMode::Diffuse.attribute_value().into(),
            )),
        )
        .unwrap();

        assert_eq!(
            tree.resolve_attribute(child, &material_shader_key())
                .unwrap()
                .value,
            Attribute::Text(ShaderMode::Diffuse.attribute_value().into())
        );
    }

    #[test]
    fn material_defaults_to_diffuse_with_smooth_normals() {
        let mut tree = GeometryTree::new();
        let node = tree.create("model", None).unwrap();
        assert_eq!(
            tree.resolve_attribute(node, &material_shader_key())
                .unwrap()
                .value,
            Attribute::Text(ShaderMode::Diffuse.attribute_value().into())
        );
        assert_eq!(
            tree.resolve_attribute(node, &material_smooth_normals_key())
                .unwrap()
                .value,
            Attribute::Bool(true)
        );
    }

    #[test]
    fn transmission_preview_blends_surface_over_existing_background() {
        let background = Color32::from_rgb(20, 80, 140);
        let surface = Color32::from_rgb(180, 220, 255);
        assert_eq!(blend_preview_surface(background, surface, 0.0), surface);
        assert_eq!(blend_preview_surface(background, surface, 1.0), background);
        assert_eq!(
            blend_preview_surface(background, surface, 0.5),
            Color32::from_rgb(100, 150, 198)
        );
    }

    #[test]
    fn schlick_fresnel_strengthens_toward_grazing_angles() {
        let normal = schlick_fresnel(1.0, 1.31);
        let diagonal = schlick_fresnel(0.5, 1.31);
        let grazing = schlick_fresnel(0.0, 1.31);
        assert!((normal - 0.018).abs() < 0.001);
        assert!(normal < diagonal && diagonal < grazing);
        assert_eq!(grazing, 1.0);
    }

    #[test]
    fn inverse_transpose_normal_stays_perpendicular_after_non_uniform_scale() {
        let scale = CoreVec3::new(2.0, 1.0, 0.5);
        let tangent = scale.component_mul(CoreVec3::new(1.0, 1.0, 0.0));
        let normal = transform_normal(
            CoreVec3::new(1.0, -1.0, 0.0).normalized(),
            scale,
            Quat::IDENTITY,
        );
        assert!(tangent.dot(normal).abs() < 1.0e-6);
        assert!((normal.length() - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn directional_light_is_global_and_object_rotation_changes_world_normal() {
        let light = global_light_direction();
        let local_normal = light;
        let identity_world =
            transform_normal(local_normal, CoreVec3::new(1.0, 1.0, 1.0), Quat::IDENTITY);
        let turned_world = transform_normal(
            local_normal,
            CoreVec3::new(1.0, 1.0, 1.0),
            Quat::from_axis_angle(CoreVec3::Z, std::f32::consts::PI),
        );

        assert!(identity_world.dot(light) > 0.99);
        assert!(
            turned_world.dot(light) < identity_world.dot(light) - 0.5,
            "the normal must rotate into world space while the light stays fixed"
        );
    }

    #[test]
    fn point_light_is_omnidirectional_and_global_light_is_optional() {
        assert_eq!(light_intensity_from_exponent(0.0), 1.0);
        assert_eq!(light_intensity_from_exponent(3.0), 1_000.0);
        assert!((intensity_to_exponent(1_000.0) - 3.0).abs() < 1.0e-6);
        assert!(light_intensity_from_exponent(12.0) > 1_000_000.0);

        let unlit = ViewportLighting {
            global_enabled: false,
            global_shadow_resolution: 0,
            shadow_filter_radius: 1,
            shadow_blur_radius: 0,
            points: Vec::new(),
            directional_shadow: None,
            point_shadows: None,
        };
        let ambient = viewport_light_factor(
            CoreVec3::Z,
            CoreVec3::ZERO,
            ShaderMode::Diffuse,
            &unlit,
            None,
        );
        let point = ViewportLight {
            position: CoreVec3::new(0.0, 0.0, 1.0),
            color: [1.0; 3],
            intensity: 4.0,
            radius: 0.0,
            shadow_resolution: 256,
        };
        let lit = ViewportLighting {
            global_enabled: false,
            global_shadow_resolution: 0,
            shadow_filter_radius: 1,
            shadow_blur_radius: 0,
            points: vec![point],
            directional_shadow: None,
            point_shadows: None,
        };
        let from_origin =
            viewport_light_factor(CoreVec3::Z, CoreVec3::ZERO, ShaderMode::Diffuse, &lit, None);
        let translated = ViewportLighting {
            global_enabled: false,
            global_shadow_resolution: 0,
            shadow_filter_radius: 1,
            shadow_blur_radius: 0,
            points: vec![ViewportLight {
                position: CoreVec3::new(5.0, -3.0, 2.0),
                ..point
            }],
            directional_shadow: None,
            point_shadows: None,
        };
        let from_translated_surface = viewport_light_factor(
            CoreVec3::Z,
            CoreVec3::new(5.0, -3.0, 1.0),
            ShaderMode::Diffuse,
            &translated,
            None,
        );
        assert!(from_origin > ambient);
        assert!((from_origin - from_translated_surface).abs() < 1.0e-6);

        let distant_surface = CoreVec3::new(0.0, 0.0, -99.0);
        let distant_weak = viewport_light_factor(
            CoreVec3::Z,
            distant_surface,
            ShaderMode::Diffuse,
            &lit,
            None,
        );
        let distant_strong = viewport_light_factor(
            CoreVec3::Z,
            distant_surface,
            ShaderMode::Diffuse,
            &ViewportLighting {
                global_enabled: false,
                global_shadow_resolution: 0,
                shadow_filter_radius: 1,
                shadow_blur_radius: 0,
                points: vec![ViewportLight {
                    intensity: 1_000_000.0,
                    ..point
                }],
                directional_shadow: None,
                point_shadows: None,
            },
            None,
        );
        assert!(distant_strong > distant_weak + 0.5);
    }

    #[test]
    fn spherical_light_radius_softens_the_diffuse_terminator() {
        let point_dark = spherical_light_lambert(-0.1, 10.0, 0.0);
        let small = spherical_light_lambert(-0.1, 10.0, 1.0);
        let large = spherical_light_lambert(-0.1, 10.0, 5.0);
        assert_eq!(point_dark, 0.0);
        assert!(small > point_dark);
        assert!(large > small);
        assert!((spherical_light_lambert(0.7, 10.0, 0.0) - 0.7).abs() < 1.0e-6);
        assert!((spherical_light_lambert(1.0, 10.0, 5.0) - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn directional_shadow_depth_comparison_and_pcf_are_stable() {
        let mut depth = vec![f32::INFINITY; 9];
        depth[4] = 0.0;
        let map = DirectionalShadowMap {
            resolution: 3,
            depth,
            origin: CoreVec3::ZERO,
            right: CoreVec3::X,
            up: CoreVec3::Y,
            forward: CoreVec3::Z,
            extent: 1.0,
            bias: 0.01,
            filter_radius: 1,
        };
        let behind = directional_shadow_visibility(CoreVec3::new(0.0, 0.0, 1.0), CoreVec3::Z, &map);
        let before =
            directional_shadow_visibility(CoreVec3::new(0.0, 0.0, -1.0), CoreVec3::Z, &map);
        assert!(behind < 1.0);
        assert_eq!(before, 1.0);
    }

    #[test]
    fn shadow_quality_downsamples_requested_maps_predictably() {
        assert_eq!(effective_shadow_resolution(0, 0), 0);
        assert_eq!(effective_shadow_resolution(1024, 0), 256);
        assert_eq!(effective_shadow_resolution(1024, 1), 512);
        assert_eq!(effective_shadow_resolution(1024, 2), 768);
        assert_eq!(effective_shadow_resolution(1024, 3), 1024);
        assert_eq!(effective_shadow_resolution(1024, 4), 1024);
        assert!(shadow_filter_radius(0) > shadow_filter_radius(3));
    }

    #[test]
    fn viewport_frame_deadline_keeps_a_fixed_phase_and_skips_late_frames() {
        let start = Instant::now();
        let period = target_frame_period(60);
        let first = advance_frame_deadline(start, start, 60);
        assert_eq!(first, start + period);
        let late = start + period * 3 + period / 2;
        assert_eq!(advance_frame_deadline(first, late, 60), start + period * 4);
        assert_eq!(sanitize_target_fps(0), 15);
        assert_eq!(sanitize_target_fps(1_000), 360);
    }

    #[test]
    fn fast_shadow_blur_is_separable_and_stays_inside_its_region() {
        let mut depth = vec![f32::INFINITY; 15];
        depth[6] = 0.0;
        depth[7] = 3.0;
        depth[8] = 6.0;
        depth[5] = -10.0;
        depth[9] = 10.0;
        fast_blur_shadow_region(&mut depth, 5, 3, 1, 1, 3, 1, 1);
        assert!((depth[6] - 1.5).abs() < 1.0e-6);
        assert!((depth[7] - 3.0).abs() < 1.0e-6);
        assert!((depth[8] - 4.5).abs() < 1.0e-6);
        assert_eq!(depth[5], -10.0);
        assert_eq!(depth[9], 10.0);
        assert!(depth[0].is_infinite());
    }

    #[test]
    fn point_shadow_atlas_uses_cube_depth_and_resolution_toggle() {
        let mut atlas = PointShadowAtlas {
            width: 18,
            height: 3,
            depth: vec![f32::INFINITY; 54],
            regions: [PointShadowRegion::default(); MAX_VIEWPORT_LIGHTS],
        };
        atlas.regions[0] = PointShadowRegion {
            row: 0,
            resolution: 3,
            bias: 0.001,
            filter_radius: 1,
        };
        atlas.depth[atlas.width + 1] = 0.5;
        let light = ViewportLight {
            position: CoreVec3::ZERO,
            color: [1.0; 3],
            intensity: 1.0,
            radius: 0.0,
            shadow_resolution: 3,
        };
        assert!(
            point_shadow_visibility(CoreVec3::X, CoreVec3::new(-1.0, 0.0, 0.0), light, 0, &atlas)
                < 1.0
        );
        atlas.regions[0].resolution = 0;
        assert_eq!(
            point_shadow_visibility(CoreVec3::X, CoreVec3::new(-1.0, 0.0, 0.0), light, 0, &atlas),
            1.0
        );
    }

    #[test]
    fn cube_shadow_coordinates_cross_faces_without_leaving_valid_uvs() {
        let first = cube_shadow_coordinate(CoreVec3::new(1.0, 0.99, 0.0).normalized());
        let second = cube_shadow_coordinate(CoreVec3::new(0.99, 1.0, 0.0).normalized());
        assert_ne!(first.0, second.0);
        for (_, u, v) in [first, second] {
            assert!((0.0..=1.0).contains(&u));
            assert!((0.0..=1.0).contains(&v));
        }
    }

    #[test]
    fn grid_alpha_decreases_with_camera_distance() {
        let near = grid_distance_alpha(1.0, 1.0);
        let medium = grid_distance_alpha(0.25, 1.0);
        let far = grid_distance_alpha(0.01, 1.0);
        assert!(near > medium && medium > far);
        assert!((0.0..=1.0).contains(&far));
    }

    #[test]
    fn grid_alpha_distance_scales_with_grid_spacing() {
        let unit_grid = grid_distance_alpha(0.1, 1.0);
        let ten_unit_grid = grid_distance_alpha(0.1, 10.0);
        let equivalent_distance = grid_distance_alpha(1.0, 1.0);
        assert!(ten_unit_grid > unit_grid);
        assert!((ten_unit_grid - equivalent_distance).abs() < 1.0e-6);
    }

    #[test]
    fn compositor_link_curve_preserves_socket_endpoints() {
        let start = Pos2::new(12.0, 45.0);
        let end = Pos2::new(240.0, 130.0);
        let curve = compositor_link_curve(start, end);
        assert_eq!(curve.first(), Some(&start));
        assert_eq!(curve.last(), Some(&end));
        assert_eq!(curve.len(), 25);
    }

    #[test]
    fn compositor_editor_properties_parse_positions_and_links() {
        assert_eq!(
            parse_compositor_position("-12.5,300"),
            Some(Vec2::new(-12.5, 300.0))
        );
        // parse_compositor_link: 2-number (old) -> (from, 0, to, 0)
        assert_eq!(parse_compositor_link("3,7"), Some((3, 0, 7, 0)));
        // 3-number (old) -> (from, 0, to, input)
        assert_eq!(parse_compositor_link("3,9,2"), Some((3, 0, 9, 2)));
        // 4-number (new) -> (from_id, from_out, to_id, to_input)
        assert_eq!(parse_compositor_link("3,1,9,2"), Some((3, 1, 9, 2)));
        assert_eq!(parse_compositor_position("invalid"), None);
        assert_eq!(parse_compositor_link("3,invalid"), None);
        assert_eq!(parse_compositor_color("0.1,0.5,1"), Some([0.1, 0.5, 1.0]));
        assert_eq!(
            parse_compositor_bezier("0,0;0.2,0.1;0.8,0.9;1,1"),
            Some(vec![[0.0, 0.0], [0.2, 0.1], [0.8, 0.9], [1.0, 1.0]])
        );
        assert_eq!(parse_compositor_color("0,1"), None);
        assert_eq!(parse_compositor_bezier("0,0;1,1"), None);
        assert_eq!(compositor_input_count(4, 0), 1);
        assert_eq!(compositor_input_count(0, 0), 0);
        assert_eq!(compositor_input_count(1, 0), 0);
        assert_eq!(compositor_input_count(2, 0), 0);
        assert_eq!(compositor_input_count(15, 0), 0);
        assert_eq!(compositor_input_count(9, 0), 2);
        assert_eq!(compositor_input_count(9, 1), 3);
        assert_eq!(compositor_input_count(13, 0), 4);
        assert_eq!(compositor_input_count(25, 0), 3);
        assert_eq!(compositor_input_count(26, 0), 3);
        assert_eq!(compositor_input_count(27, 0), 2);
        assert_eq!(compositor_output_count(11), 4);
        assert_eq!(compositor_output_count(25), 1);
        assert_eq!(compositor_output_count(26), 1);
        assert_eq!(compositor_output_count(27), 2);
        assert_eq!(compositor_output_count(0), 1);
        assert_eq!(compositor_input_count(28, 0), 3);
        assert_eq!(compositor_output_count(28), 0);
        assert_eq!(compositor_input_count(30, 0), 3);
        assert_eq!(compositor_output_count(30), 1);
        for kind in 0..31 {
            assert!(!compositor_node_description(kind).is_empty());
            for input in 0..compositor_input_count(kind, 0) {
                assert!(!compositor_input_label(kind, input).is_empty());
            }
            for output in 0..compositor_output_count(kind) {
                assert!(!compositor_output_label(kind, output).is_empty());
            }
            let socket_bottom = (0..compositor_input_count(kind, 0))
                .map(|input| compositor_input_socket_y(kind, input))
                .chain(
                    (0..compositor_output_count(kind))
                        .map(|output| compositor_output_socket_y(kind, output)),
                )
                .fold(0.0_f32, f32::max);
            assert!(compositor_controls_top(kind, 0) >= socket_bottom + 20.0);
        }
        assert!(
            compositor_controls_top(9, 1) > compositor_controls_top(9, 0),
            "the optional Alpha socket must reserve its own row"
        );
        assert!(matches!(
            NodeSettings::default_for_kind(25),
            Some(NodeSettings::Position { values }) if values == [0.0; 3]
        ));
        assert!(matches!(
            NodeSettings::default_for_kind(26),
            Some(NodeSettings::Rotation { degrees }) if degrees == [0.0; 3]
        ));
        let parent = egui::LayerId::new(egui::Order::Background, Id::new("fixture"));
        assert_eq!(compositor_control_layer(parent, 3).order, parent.order);
        let canvas = Rect::from_min_max(Pos2::new(100.0, 50.0), Pos2::new(900.0, 650.0));
        let origin = Pos2::new(180.0, 90.0);
        let scale = 2.0;
        let size = Vec2::new(200.0, 100.0);
        let position = compositor_centered_position(canvas, origin, scale, size);
        let screen_center = origin + (position + size * 0.5) * scale;
        assert_eq!(screen_center, canvas.center());
    }

    #[test]
    fn object_handles_copy_live_values_and_reject_cycles() {
        let handle = |id, value, source_handle| CompositorNode {
            id,
            object_index: 0,
            object_name: "fixture".to_owned(),
            settings_object_name: Some("fixture".to_owned()),
            settings: NodeSettings::ObjectHandle {
                object_index: 0,
                label: format!("Handle {id}"),
                control: 0,
                value,
                minimum: -10.0,
                maximum: 10.0,
                source_handle,
            },
            position: Vec2::ZERO,
        };
        let mut nodes = vec![
            handle(1, 3.5, None),
            handle(2, 0.0, Some(1)),
            handle(3, 0.0, Some(2)),
        ];
        assert_eq!(resolve_object_handle_value(&nodes, 3), Some(3.5));
        if let NodeSettings::ObjectHandle { value, .. } = &mut nodes[0].settings {
            *value = -2.25;
        }
        assert_eq!(resolve_object_handle_value(&nodes, 2), Some(-2.25));
        if let NodeSettings::ObjectHandle { source_handle, .. } = &mut nodes[0].settings {
            *source_handle = Some(3);
        }
        assert_eq!(resolve_object_handle_value(&nodes, 1), None);
        assert_eq!(resolve_object_handle_value(&nodes, 3), None);
    }

    #[test]
    fn compositor_object_names_survive_index_shifts() {
        let before = vec!["Deleted".to_owned(), "Flag".to_owned(), "Pole".to_owned()];
        assert_eq!(object_index_by_name(&before, "Flag"), Some(1));
        assert_eq!(object_index_by_name(&before, "Pole"), Some(2));

        let after = vec!["Flag".to_owned(), "Pole".to_owned()];
        assert_eq!(object_index_by_name(&after, "Flag"), Some(0));
        assert_eq!(object_index_by_name(&after, "Pole"), Some(1));
        assert_eq!(object_index_by_name(&after, "Deleted"), None);
    }

    #[test]
    fn compositor_mix_uses_optional_alpha_formula() {
        let texture = |name: &str, red: u8| TextureAsset {
            name: name.into(),
            width: 1,
            height: 1,
            pixels: vec![red, red, red, 255],
            cached_mips: Vec::new(),
        };
        let a = texture("a", 255);
        let b = texture("b", 0);
        let alpha = texture("alpha", 64);
        let fallback = combine_compositor_textures(&a, &b, None, 1, 0, 0.5);
        let driven = combine_compositor_textures(&a, &b, Some(&alpha), 1, 0, 0.5);
        assert_eq!(fallback.pixels[0], 128);
        assert_eq!(driven.pixels[0], 64);
    }

    #[test]
    #[ignore = "requires a Vulkan device"]
    fn vulkan_worker_mixes_and_returns_asynchronously() {
        let mut worker = vulkan_compositor::VulkanCompositorWorker::new()
            .expect("Vulkan worker should initialize");
        let texture = |name: &str, pixels| {
            Arc::new(TextureAsset {
                name: name.into(),
                width: 1,
                height: 1,
                pixels,
                cached_mips: Vec::new(),
            })
        };
        let generation = worker.submit_latest(
            9,
            u32::MAX,
            texture("red", vec![255, 0, 0, 255]),
            texture("blue", vec![0, 0, 255, 255]),
            1,
            0,
            0.5,
        );
        let deadline = Instant::now() + Duration::from_secs(2);
        let result = loop {
            if let Some(result) = worker.try_result() {
                break result;
            }
            assert!(Instant::now() < deadline, "Vulkan worker timed out");
            std::thread::sleep(Duration::from_millis(1));
        };
        assert_eq!(result.generation, generation);
        let output = result.texture.expect("Vulkan mix should succeed");
        assert_eq!((output.width, output.height), (1, 1));
    }

    #[test]
    fn compositor_math_multiply_zero_blacks_rgb_and_preserves_alpha() {
        let texture = TextureAsset {
            name: "earth".into(),
            width: 1,
            height: 1,
            pixels: vec![210, 120, 45, 173],
            cached_mips: Vec::new(),
        };
        let result = apply_compositor_math(texture, 2, 0.0);
        assert_eq!(result.pixels, vec![0, 0, 0, 173]);
    }

    #[test]
    fn compositor_lod_is_quantized_and_hysteretic() {
        assert_eq!(select_compositor_lod(100.0, u32::MAX), 256);
        assert_eq!(select_compositor_lod(350.0, u32::MAX), 1024);
        assert_eq!(select_compositor_lod(350.0, 1024), 1024);
        assert_eq!(select_compositor_lod(100.0, 1024), 256);
        assert_eq!(
            select_compositor_lod_for_backend(1.0, 128, RenderDevice::Vulkan),
            u32::MAX
        );
    }

    #[test]
    fn compositor_time_scales_and_only_wraps_for_positive_modulus() {
        assert_eq!(scaled_modulated_time(3.0, 2.0, 0.0), 6.0);
        assert_eq!(scaled_modulated_time(3.0, 2.0, -4.0), 6.0);
        assert!((scaled_modulated_time(3.0, 2.0, 2.5) - 1.0).abs() < 1.0e-6);
        assert!((scaled_modulated_time(0.75, -2.0, 1.0) - 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn timing_metric_tracks_latest_ema_maximum_and_samples() {
        let mut metric = TimingMetric::default();
        metric.record(Duration::from_millis(2));
        metric.record(Duration::from_millis(4));
        assert_eq!(metric.samples, 2);
        assert_eq!(metric.latest_ms, 4.0);
        assert_eq!(metric.maximum_ms, 4.0);
        assert!((metric.average_ms - 2.2).abs() < 1.0e-9);
    }

    #[test]
    fn copied_telemetry_report_contains_latency_columns_and_all_stages() {
        let mut performance = EditorPerformanceTelemetry::default();
        performance
            .control_to_present
            .record(Duration::from_millis(17));
        let report = format_telemetry_report(&performance);
        assert!(report.starts_with("Stage\tLatest\tEMA\tMaximum\tSamples\n"));
        assert!(report.contains("Control → presented frame\t17.000 ms"));
        assert!(report.contains("Vulkan graph encode + submission"));
        assert!(report.contains("GPU resource loading / upload"));
        assert!(report.contains("GPU mesh-buffer upload"));
        assert!(report.contains("GPU texture + mip upload"));
        assert!(report.contains("Viewport target allocation"));
        assert!(report.contains("Shadow target allocation"));
        assert!(report.contains("Vulkan renderer initialization"));
        assert_eq!(report.lines().count(), 19);
    }

    #[test]
    fn loading_indicator_finishes_only_after_its_scene_revision_is_presented() {
        let pending = BTreeMap::from([
            ("first.glb".to_owned(), 4_u64),
            ("second.glb".to_owned(), 7_u64),
        ]);
        assert!(completed_asset_loads(&pending, 3).is_empty());
        assert_eq!(completed_asset_loads(&pending, 4), vec!["first.glb"]);
        assert_eq!(
            completed_asset_loads(&pending, 7),
            vec!["first.glb", "second.glb"]
        );
    }

    #[test]
    fn compositor_lod_resize_preserves_aspect_ratio() {
        let texture = Arc::new(TextureAsset {
            name: "wide".into(),
            width: 8,
            height: 4,
            pixels: vec![255; 8 * 4 * 4],
            cached_mips: Vec::new(),
        });
        let resized = resize_texture_for_lod(&texture, 4);
        assert_eq!((resized.width, resized.height), (4, 2));
    }

    #[test]
    fn compositor_channels_extract_join_and_grayscale() {
        let source = TextureAsset {
            name: "source".into(),
            width: 1,
            height: 1,
            pixels: vec![30, 90, 210, 128],
            cached_mips: Vec::new(),
        };
        let red = extract_compositor_channel(source.clone(), 0);
        let green = extract_compositor_channel(source.clone(), 1);
        let blue = extract_compositor_channel(source.clone(), 2);
        let alpha = extract_compositor_channel(source.clone(), 3);
        let joined = join_compositor_channels([&red, &green, &blue], Some(&alpha));
        assert_eq!(joined.pixels, source.pixels);
        let gray = grayscale_compositor_texture(source, 1);
        assert_eq!(gray.pixels, vec![110, 110, 110, 128]);
    }

    #[test]
    fn compositor_srgb_linear_round_trip_is_close() {
        let source = TextureAsset {
            name: "source".into(),
            width: 1,
            height: 1,
            pixels: vec![32, 128, 230, 77],
            cached_mips: Vec::new(),
        };
        let linear = convert_compositor_color_space(source.clone(), 0, 1);
        let round_trip = convert_compositor_color_space(linear, 1, 0);
        for (actual, expected) in round_trip.pixels.iter().zip(source.pixels) {
            assert!((*actual as i16 - expected as i16).abs() <= 4);
        }
    }

    #[test]
    fn painted_mask_produces_grayscale_texture_values() {
        let mask = PaintedMask {
            width: 2,
            height: 1,
            pixels: vec![0, 255],
            revision: 1,
        };
        assert_eq!(
            mask.texture(false).pixels,
            vec![0, 0, 0, 255, 255, 255, 255, 255]
        );
    }

    #[test]
    fn painted_mask_hex_persistence_is_lossless() {
        let bytes = vec![0, 1, 15, 16, 127, 128, 254, 255];
        assert_eq!(decode_hex(&encode_hex(&bytes)), Some(bytes));
        assert!(decode_hex("abc").is_none());
        assert!(decode_hex("zz").is_none());
    }

    #[test]
    fn density_texture_produces_area_weighted_positive_particle_masses() {
        let mesh = MeshAsset {
            vertices: vec![
                zerofps_assets::Vertex {
                    position: [0.0, 0.0, 0.0],
                    uv: [0.0, 0.0],
                    ..Default::default()
                },
                zerofps_assets::Vertex {
                    position: [1.0, 0.0, 0.0],
                    uv: [1.0, 0.0],
                    ..Default::default()
                },
                zerofps_assets::Vertex {
                    position: [0.0, 1.0, 0.0],
                    uv: [0.0, 1.0],
                    ..Default::default()
                },
            ],
            primitives: vec![zerofps_assets::Primitive {
                indices: vec![0, 1, 2],
                ..Default::default()
            }],
            ..Default::default()
        };
        let mask = PaintedMask {
            width: 1,
            height: 1,
            pixels: vec![255],
            revision: 1,
        };
        let masses = area_weighted_particle_masses(
            &mesh,
            &mask,
            CoreVec3::ONE,
            3.0,
            2.0,
            1.0e-6,
            false,
            1.0,
        );
        for mass in masses {
            assert!((mass - 1.0).abs() < 1.0e-6);
        }
    }

    #[test]
    fn force_formula_supports_cartesian_time_functions_and_precedence() {
        let value = evaluate_force_formula("x + 2*y - z + sin(pi*t)", 1.0, 2.0, 0.5, 0.5)
            .expect("valid force formula");
        assert!((value - 5.5).abs() < 1.0e-5);
        assert!(evaluate_force_formula("sqrt(-1)", 0.0, 0.0, 0.0, 0.0).is_err());
        assert!(evaluate_force_formula("unknown(x)", 0.0, 0.0, 0.0, 0.0).is_err());
    }

    #[test]
    fn texture_paint_barycentrics_identify_inside_and_outside_points() {
        let triangle = [
            Pos2::new(0.0, 0.0),
            Pos2::new(10.0, 0.0),
            Pos2::new(0.0, 10.0),
        ];
        let weights = barycentric_2d(Pos2::new(2.0, 3.0), triangle).unwrap();
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        assert!(barycentric_2d(Pos2::new(9.0, 9.0), triangle).is_none());
    }

    #[test]
    fn texture_paint_raycast_returns_nearest_hit_weights() {
        let triangle = [
            CoreVec3::new(-1.0, 5.0, -1.0),
            CoreVec3::new(1.0, 5.0, -1.0),
            CoreVec3::new(0.0, 5.0, 1.0),
        ];
        let (distance, weights) = ray_triangle_hit(CoreVec3::ZERO, CoreVec3::Y, triangle).unwrap();
        assert!((distance - 5.0).abs() < 1.0e-6);
        assert!((weights.iter().sum::<f32>() - 1.0).abs() < 1.0e-6);
        assert!(weights.iter().all(|weight| *weight >= 0.0));
    }

    #[test]
    fn scene_camera_preview_uses_camera_position_and_forward_axis() {
        let camera = SceneCameraPreview {
            transform: Transform::IDENTITY,
            perspective: true,
            field_of_view_degrees: 90.0,
            aspect_ratio: 2.0,
            orthographic_size: 10.0,
        };
        let view = scene_camera_render_view(&camera, Vec2::new(800.0, 400.0), 1.0);
        assert!(view.0.abs() < 1.0e-6);
        assert!(view.1.abs() < 1.0e-6);
        assert_eq!(view.6, ProjectionMode::Perspective);
        assert!((view.4.y - PERSPECTIVE_CAMERA_DISTANCE).abs() < 1.0e-6);
    }

    #[test]
    fn scene_camera_preview_respects_perspective_aspect_and_orthographic_size() {
        let mut camera = SceneCameraPreview {
            transform: Transform::IDENTITY,
            perspective: true,
            field_of_view_degrees: 90.0,
            aspect_ratio: 2.0,
            orthographic_size: 10.0,
        };
        let fitted = fit_aspect_rect(
            Rect::from_min_size(Pos2::ZERO, Vec2::new(1000.0, 800.0)),
            camera.aspect_ratio,
        );
        assert!((fitted.width() / fitted.height() - 2.0).abs() < 1.0e-6);
        let perspective = scene_camera_render_view(&camera, fitted.size(), 1.0);
        assert_eq!(perspective.6, ProjectionMode::Perspective);

        camera.perspective = false;
        let orthographic = scene_camera_render_view(&camera, fitted.size(), 1.0);
        assert_eq!(orthographic.6, ProjectionMode::Orthographic);
        assert!(orthographic.3 > perspective.3);
    }

    #[test]
    fn scene_camera_preview_preserves_world_quaternion_roll() {
        // Camera rotation from appdata/models/Fast-Driver/game_v0.zfp.
        let rotation = Quat {
            x: -0.6830127,
            y: 0.6830127,
            z: 0.18301277,
            w: -0.18301277,
        }
        .normalized();
        let camera = SceneCameraPreview {
            transform: Transform {
                rotation,
                ..Transform::IDENTITY
            },
            perspective: true,
            field_of_view_degrees: 130.0,
            aspect_ratio: 16.0 / 9.0,
            orthographic_size: 10.0,
        };
        let view = scene_camera_render_view(&camera, Vec2::new(1280.0, 720.0), 10.0);
        let (right, up, forward) = camera_basis(view.0, view.1, view.2);
        assert!(right.approx_eq(rotation.rotate(CoreVec3::X), 1.0e-5));
        assert!(up.approx_eq(rotation.rotate(CoreVec3::Z), 1.0e-5));
        assert!(forward.approx_eq(rotation.rotate(CoreVec3::Y), 1.0e-5));
        assert!(
            view.2.abs() > 3.0,
            "fixture includes an approximately 180° roll"
        );
    }

    #[test]
    fn auto_braking_smoothly_opposes_rotation_at_low_throttle() {
        let positive = smooth_rpm_governor_torque(0.0, 100.0, 250.0, 6_000.0, 1.0);
        let negative = smooth_rpm_governor_torque(0.0, -100.0, 250.0, 6_000.0, 1.0);
        assert!(positive < 0.0);
        assert!(negative > 0.0);
        assert!((positive.abs() - negative.abs()).abs() < 1.0e-5);
        assert_eq!(
            smooth_rpm_governor_torque(0.0, 0.0, 250.0, 6_000.0, 1.0),
            0.0
        );
    }

    #[test]
    fn auto_braking_drives_toward_and_holds_throttle_rpm_target() {
        let below_target = 2_000.0 * std::f32::consts::TAU / 60.0;
        assert!(smooth_rpm_governor_torque(0.5, below_target, 250.0, 6_000.0, 1.0) > 0.0);
        let above_target = 4_000.0 * std::f32::consts::TAU / 60.0;
        assert!(smooth_rpm_governor_torque(0.5, above_target, 250.0, 6_000.0, 1.0) < 0.0);
        assert!(smooth_rpm_governor_torque(0.5, 0.0, 250.0, 6_000.0, -1.0) < 0.0);
    }

    #[test]
    fn cameras_are_weightless_and_non_collidable() {
        let mut scene = EditorScene::default();
        let camera = scene.add("Camera", ObjectKind::Camera, None);
        assert!(!scene.participates_in_physics(camera));
        assert!(
            scene
                .tree
                .node(camera)
                .unwrap()
                .components
                .iter()
                .all(|component| { !matches!(component, Component::Collider { .. }) })
        );

        let model = scene.add("Model", ObjectKind::Model, None);
        assert!(scene.participates_in_physics(model));
    }
}
