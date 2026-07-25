//! ZeroFPS scene editor prototype.
//!
//! This executable intentionally keeps an editor-local authoring model.  The
//! model is a thin adapter which can later be replaced by `zerofps-core`
//! handles without coupling the UI to runtime ownership.

use std::{
    collections::{BTreeMap, BTreeSet},
    path::PathBuf,
    sync::{Arc, Condvar, Mutex, mpsc},
    time::{Duration, Instant},
};

use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, FontId, Id, Key, Layout, Pos2, Rect, RichText, Sense,
    Stroke, TextureHandle, TextureId, TextureOptions, Vec2,
};
use zerofps_assets::{MeshAsset, MeshAutofixReport, TextureAsset, autofix_mesh, import_file};
use zerofps_core::{
    Attribute, AttributeDeclaration, AttributeKey, Component, GeometryTree, NodeId, Quat,
    Vec3 as CoreVec3,
};
use zerofps_formats::{BundleAsset, ProjectFile, load_zfp, save_zfp};

fn main() -> eframe::Result<()> {
    let options = eframe::NativeOptions {
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
            }),
            ObjectKind::Light => Some(Component::Light {
                intensity: 1.0,
                color: [1.0; 3],
            }),
            ObjectKind::Camera => Some(Component::Camera {
                field_of_view_degrees: 60.0,
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
enum WorkspaceTab {
    Scene,
    Compositing,
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
}

struct CompositorImageAsset {
    path: String,
}

#[derive(Clone, Copy)]
struct PreviewVertex {
    position: [f32; 3],
    normal: [f32; 3],
    uv: [f32; 2],
    color: [f32; 4],
}

#[derive(Clone)]
struct PreviewTriangle {
    vertices: [PreviewVertex; 3],
    base_color: [f32; 4],
    texture: Option<Arc<TextureAsset>>,
    shader: ShaderMode,
    smooth_normals: bool,
    transmission: f32,
    ior: f32,
}

struct DepthFrame {
    color: ColorImage,
    linear_depth: Vec<f32>,
}

struct RenderJob {
    key: DepthCacheKey,
    viewport_size: Vec2,
    triangles: Arc<Vec<PreviewTriangle>>,
    camera: (f32, f32, f32, CoreVec3, f32, ProjectionMode),
    show_grid: bool,
    mode: ViewportMode,
    tool: Tool,
    reusable_depth: Vec<f32>,
}

struct RenderResult {
    key: DepthCacheKey,
    frame: DepthFrame,
    camera: (f32, f32, f32, CoreVec3, f32, ProjectionMode),
    triangles: Arc<Vec<PreviewTriangle>>,
    show_grid: bool,
    mode: ViewportMode,
    tool: Tool,
}

struct DisplayWorker {
    pending: Arc<(Mutex<Option<RenderJob>>, Condvar)>,
    results: mpsc::Receiver<RenderResult>,
}

struct PresentedView {
    camera: (f32, f32, f32, CoreVec3, f32, ProjectionMode),
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
struct ProcessedInput {
    orbit: Vec2,
    pan: Vec2,
    zoom_log: f32,
    viewport_extent: f32,
}

struct InputWorker {
    samples: mpsc::Sender<InputSample>,
    actions: mpsc::Receiver<ProcessedInput>,
}

struct ImportRequest {
    path: String,
    add_to_scene: bool,
}

struct ImportResult {
    path: String,
    add_to_scene: bool,
    asset: Result<MeshAsset, zerofps_assets::ImportError>,
}

struct AssetImportWorker {
    requests: mpsc::Sender<ImportRequest>,
    results: mpsc::Receiver<ImportResult>,
}

impl AssetImportWorker {
    fn new(ctx: egui::Context) -> Self {
        let (request_sender, request_receiver) = mpsc::channel::<ImportRequest>();
        let (result_sender, results) = mpsc::channel();
        std::thread::Builder::new()
            .name("zerofps-asset-import".into())
            .spawn(move || {
                while let Ok(request) = request_receiver.recv() {
                    let result = ImportResult {
                        asset: import_file(&request.path),
                        path: request.path,
                        add_to_scene: request.add_to_scene,
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
        }
    }

    fn submit(&self, path: String, add_to_scene: bool) {
        let _ = self.requests.send(ImportRequest { path, add_to_scene });
    }
}

impl InputWorker {
    fn new(ctx: egui::Context) -> Self {
        let (sample_sender, sample_receiver) = mpsc::channel::<InputSample>();
        let (action_sender, actions) = mpsc::channel();
        std::thread::Builder::new()
            .name("zerofps-input".into())
            .spawn(move || {
                while let Ok(first) = sample_receiver.recv() {
                    let mut processed = ProcessedInput {
                        orbit: first.orbit,
                        pan: first.pan,
                        zoom_log: first.zoom_log,
                        viewport_extent: first.viewport_extent,
                    };
                    for sample in sample_receiver.try_iter() {
                        processed.orbit += sample.orbit;
                        processed.pan += sample.pan;
                        processed.zoom_log += sample.zoom_log;
                        processed.viewport_extent = sample.viewport_extent;
                    }
                    if action_sender.send(processed).is_err() {
                        break;
                    }
                    ctx.request_repaint();
                }
            })
            .expect("input worker thread should start");
        Self {
            samples: sample_sender,
            actions,
        }
    }

    fn submit(&self, sample: InputSample) {
        let _ = self.samples.send(sample);
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
                loop {
                    let job = {
                        let (lock, ready) = &*worker_pending;
                        let guard = lock.lock().unwrap_or_else(|poisoned| poisoned.into_inner());
                        let mut guard = ready
                            .wait_while(guard, |job| job.is_none())
                            .unwrap_or_else(|poisoned| poisoned.into_inner());
                        guard.take().expect("display job became available")
                    };
                    let frame = if job.mode == ViewportMode::Wireframe {
                        let width = job.viewport_size.x.round().max(1.0) as usize;
                        let height = job.viewport_size.y.round().max(1.0) as usize;
                        let mut linear_depth = job.reusable_depth;
                        linear_depth.resize(width * height, f32::INFINITY);
                        linear_depth.fill(f32::INFINITY);
                        DepthFrame {
                            color: ColorImage::new([width, height], Color32::TRANSPARENT),
                            linear_depth,
                        }
                    } else {
                        rasterize_depth_frame(
                            job.viewport_size,
                            &job.triangles,
                            job.camera,
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

#[derive(Default)]
struct RasterWorkspace {
    prepared: Vec<PreparedRasterTriangle>,
    bands: Vec<Vec<PreparedRasterTriangle>>,
    clipped: Vec<PreviewVertex>,
    projected: Vec<RasterVertex>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
struct DepthCacheKey {
    size: [usize; 2],
    yaw: f32,
    pitch: f32,
    zoom: f32,
    target: CoreVec3,
    grid_spacing: f32,
    projection: ProjectionMode,
    scene_revision: u64,
    show_grid: bool,
    mode: ViewportMode,
    tool: Tool,
}

struct EditorApp {
    scene: EditorScene,
    workspace_tab: WorkspaceTab,
    active_tool: Tool,
    play_state: PlayState,
    build_started: Option<Instant>,
    bottom_tab: BottomTab,
    material_tab: MaterialTab,
    compositor_pan: Vec2,
    compositor_zoom: f32,
    compositor_selected_node: usize,
    compositor_node_active: Vec<bool>,
    compositor_node_positions: Vec<Vec2>,
    compositor_links: Vec<(usize, usize, usize)>,
    compositor_dragging_node: Option<(usize, Vec2)>,
    compositor_pending_output: Option<usize>,
    compositor_pending_spawn: Option<usize>,
    compositor_object_index: usize,
    compositor_texture_channel: usize,
    compositor_constant: f32,
    compositor_color: [f32; 3],
    compositor_bezier_points: Vec<[f32; 2]>,
    compositor_math_operation: usize,
    compositor_threshold: f32,
    compositor_threshold_width: f32,
    compositor_filter: usize,
    compositor_filter_radius: f32,
    compositor_output_width: u32,
    compositor_output_height: u32,
    compositor_output_object_index: usize,
    compositor_output_texture_channel: usize,
    compositor_combine_mode: usize,
    compositor_combine_operation: usize,
    compositor_combine_alpha: f32,
    compositor_color_space_from: usize,
    compositor_color_space_to: usize,
    compositor_extract_channel: usize,
    compositor_grayscale_mode: usize,
    compositor_texture_overrides: Vec<(NodeId, TextureAsset)>,
    advanced: bool,
    show_grid: bool,
    grid_spacing: f32,
    snap: bool,
    viewport_focused: bool,
    camera_yaw: f32,
    camera_pitch: f32,
    camera_zoom: f32,
    camera_target: CoreVec3,
    hierarchy_filter: String,
    logs: Vec<LogEntry>,
    project_path: PathBuf,
    project_has_destination: bool,
    project_dirty: bool,
    undo_stack: Vec<GeometryTree>,
    redo_stack: Vec<GeometryTree>,
    asset_import_path: String,
    imported_assets: Vec<ImportedAsset>,
    compositor_images: Vec<CompositorImageAsset>,
    viewport_mode: ViewportMode,
    projection_mode: ProjectionMode,
    viewport_color: Option<TextureHandle>,
    viewport_depth: Vec<f32>,
    viewport_depth_key: Option<DepthCacheKey>,
    scene_revision: u64,
    cached_preview_revision: u64,
    cached_preview: Arc<Vec<PreviewTriangle>>,
    display_worker: DisplayWorker,
    input_worker: InputWorker,
    asset_import_worker: AssetImportWorker,
    viewport_requested_key: Option<DepthCacheKey>,
    presented_view: Option<PresentedView>,
    dialog_result: Option<mpsc::Receiver<Option<PathBuf>>>,
    save_dialog_result: Option<mpsc::Receiver<Option<PathBuf>>>,
    load_dialog_result: Option<mpsc::Receiver<Option<PathBuf>>>,
    compositor_image_dialog_result: Option<mpsc::Receiver<Option<PathBuf>>>,
}

impl EditorApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        Self {
            scene: EditorScene::default(),
            workspace_tab: WorkspaceTab::Scene,
            active_tool: Tool::Move,
            play_state: PlayState::Editing,
            build_started: None,
            bottom_tab: BottomTab::Assets,
            material_tab: MaterialTab::Shader,
            compositor_pan: Vec2::ZERO,
            compositor_zoom: 1.0,
            compositor_selected_node: 8,
            compositor_node_active: vec![
                false, false, false, false, false, false, false, false, true, false, false, false,
                false, false,
            ],
            compositor_node_positions: vec![
                Vec2::new(0.0, 0.0),
                Vec2::new(0.0, 240.0),
                Vec2::new(0.0, 440.0),
                Vec2::new(280.0, 0.0),
                Vec2::new(280.0, 340.0),
                Vec2::new(560.0, 0.0),
                Vec2::new(560.0, 190.0),
                Vec2::new(560.0, 420.0),
                Vec2::new(300.0, 100.0),
                Vec2::new(280.0, 560.0),
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::ZERO,
                Vec2::ZERO,
            ],
            compositor_links: Vec::new(),
            compositor_dragging_node: None,
            compositor_pending_output: None,
            compositor_pending_spawn: None,
            compositor_object_index: 0,
            compositor_texture_channel: 0,
            compositor_constant: 0.5,
            compositor_color: [0.5, 0.5, 0.5],
            compositor_bezier_points: vec![[0.0, 0.0], [0.33, 0.33], [0.67, 0.67], [1.0, 1.0]],
            compositor_math_operation: 2,
            compositor_threshold: 0.5,
            compositor_threshold_width: 0.1,
            compositor_filter: 0,
            compositor_filter_radius: 3.0,
            compositor_output_width: 1920,
            compositor_output_height: 1080,
            compositor_output_object_index: 0,
            compositor_output_texture_channel: 0,
            compositor_combine_mode: 0,
            compositor_combine_operation: 0,
            compositor_combine_alpha: 0.5,
            compositor_color_space_from: 0,
            compositor_color_space_to: 1,
            compositor_extract_channel: 0,
            compositor_grayscale_mode: 0,
            compositor_texture_overrides: Vec::new(),
            advanced: false,
            show_grid: true,
            grid_spacing: 1.0,
            snap: false,
            viewport_focused: false,
            camera_yaw: -0.55,
            camera_pitch: 0.42,
            camera_zoom: 1.0,
            camera_target: CoreVec3::ZERO,
            hierarchy_filter: String::new(),
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
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            asset_import_path: String::new(),
            imported_assets: Vec::new(),
            compositor_images: Vec::new(),
            viewport_mode: ViewportMode::Shaded,
            projection_mode: ProjectionMode::Orthographic,
            viewport_color: None,
            viewport_depth: Vec::new(),
            viewport_depth_key: None,
            scene_revision: 0,
            cached_preview_revision: u64::MAX,
            cached_preview: Arc::new(Vec::new()),
            display_worker: DisplayWorker::new(cc.egui_ctx.clone()),
            input_worker: InputWorker::new(cc.egui_ctx.clone()),
            asset_import_worker: AssetImportWorker::new(cc.egui_ctx.clone()),
            viewport_requested_key: None,
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
        self.asset_import_worker.submit(path, add_to_scene);
        self.asset_import_path.clear();
    }

    fn poll_asset_imports(&mut self) {
        while let Ok(result) = self.asset_import_worker.results.try_recv() {
            let path = result.path;
            match result.asset {
                Ok(asset) => {
                    let triangle_count = asset.triangle_count();
                    let (autofixed_mesh, autofix_report) = autofix_mesh(&asset);
                    let inferred_grid_spacing = self
                        .imported_assets
                        .is_empty()
                        .then(|| infer_grid_spacing(&asset))
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
                        index
                    } else {
                        self.imported_assets.push(ImportedAsset {
                            path,
                            mesh: asset,
                            autofixed_mesh,
                            autofix_report,
                        });
                        self.imported_assets.len() - 1
                    };
                    self.scene_revision = self.scene_revision.wrapping_add(1);
                    if result.add_to_scene {
                        self.add_asset_to_scene(asset_index);
                    }
                }
                Err(error) => self.logs.push(LogEntry {
                    level: "ERROR",
                    color: Color32::from_rgb(235, 91, 91),
                    message: format!("Asset import failed: {error}"),
                }),
            }
        }
    }

    fn add_asset_to_scene(&mut self, asset_index: usize) {
        let Some(asset) = self.imported_assets.get(asset_index) else {
            return;
        };
        let previous = self.scene.tree.clone();
        let object_name = std::path::Path::new(&asset.path)
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or(&asset.mesh.name)
            .to_owned();
        let id = self.scene.add(&object_name, ObjectKind::Empty, None);
        if self
            .scene
            .tree
            .add_component(
                id,
                Component::Model {
                    asset: asset.path.clone(),
                },
            )
            .is_ok()
        {
            let _ = self.scene.tree.set_attribute(
                id,
                mesh_autofix_key(),
                AttributeDeclaration::Value(Attribute::Bool(true)),
            );
            self.scene.selected = Some(id);
            self.record_undo(previous);
            self.logs.push(LogEntry {
                level: "SCENE",
                color: Color32::from_rgb(103, 191, 255),
                message: format!("Added `{object_name}` to the scene"),
            });
        }
    }

    fn build_preview_triangles(&self) -> Vec<PreviewTriangle> {
        let total_scene_triangles = self
            .scene
            .tree
            .iter()
            .filter_map(|(_, node)| {
                let path = node
                    .components
                    .iter()
                    .find_map(|component| match component {
                        Component::Model { asset } => Some(asset.as_str()),
                        _ => None,
                    })?;
                self.imported_assets
                    .iter()
                    .find(|asset| asset.path == path)
                    .map(|asset| asset.mesh.triangle_count())
            })
            .sum::<usize>();
        let mut output = Vec::with_capacity(total_scene_triangles);
        for (id, node) in self.scene.tree.iter() {
            let Some(path) = node
                .components
                .iter()
                .find_map(|component| match component {
                    Component::Model { asset } => Some(asset.as_str()),
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
            let compositor_override = self
                .compositor_texture_overrides
                .iter()
                .find(|(target, _)| *target == id)
                .map(|(_, texture)| Arc::new(texture.clone()));
            for primitive in &asset.primitives {
                let material = primitive
                    .material
                    .as_ref()
                    .and_then(|name| asset.materials.get(name));
                let base_color = if compositor_override.is_some() {
                    [1.0; 4]
                } else {
                    material
                        .map(|material| material.base_color)
                        .unwrap_or([0.42, 0.64, 0.78, 1.0])
                };
                let texture = compositor_override.clone().or_else(|| {
                    material
                        .and_then(|material| material.base_color_texture.as_ref())
                        .and_then(|name| asset.textures.get(name))
                        .cloned()
                        .map(Arc::new)
                });
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
                        position: [0.0; 3],
                        normal: [0.0, 0.0, 1.0],
                        uv: [0.0; 2],
                        color: [1.0; 4],
                    }; 3];
                    for (destination, index) in vertices.iter_mut().zip(triangle) {
                        let source_vertex = asset.vertices[*index as usize];
                        let source = source_vertex.position;
                        let local = CoreVec3::new(source[0], source[1], source[2]);
                        let world = transform
                            .rotation
                            .rotate(transform.scale.component_mul(local))
                            + transform.translation;
                        let local_normal = CoreVec3::new(
                            source_vertex.normal[0],
                            source_vertex.normal[1],
                            source_vertex.normal[2],
                        );
                        let world_normal =
                            transform_normal(local_normal, transform.scale, transform.rotation);
                        *destination = PreviewVertex {
                            position: [world.x, world.y, world.z],
                            normal: [world_normal.x, world_normal.y, world_normal.z],
                            uv: source_vertex.uv,
                            color: source_vertex.color,
                        };
                    }
                    output.push(PreviewTriangle {
                        vertices,
                        base_color,
                        texture: texture.clone(),
                        shader,
                        smooth_normals,
                        transmission,
                        ior,
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
            Err(error) => self.logs.push(LogEntry {
                level: "ERROR",
                color: Color32::from_rgb(235, 91, 91),
                message: format!("Could not save project: {error}"),
            }),
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
                    let path = path.to_string_lossy().into_owned();
                    if !self
                        .compositor_images
                        .iter()
                        .any(|image| image.path == path)
                    {
                        self.compositor_images
                            .push(CompositorImageAsset { path: path.clone() });
                        self.project_dirty = true;
                        self.logs.push(LogEntry {
                            level: "IMPORT",
                            color: Color32::from_rgb(112, 210, 156),
                            message: format!("Registered compositor image `{path}`"),
                        });
                    }
                }
            }
            Err(mpsc::TryRecvError::Disconnected) => self.compositor_image_dialog_result = None,
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
                let referenced_assets: Vec<String> = self
                    .scene
                    .tree
                    .iter()
                    .flat_map(|(_, node)| node.components.iter())
                    .filter_map(|component| match component {
                        Component::Model { asset } => Some(asset.clone()),
                        _ => None,
                    })
                    .collect();
                for path in referenced_assets {
                    if !self.imported_assets.iter().any(|asset| asset.path == path) {
                        self.asset_import_path = path;
                        self.import_asset(false);
                    }
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
            Err(error) => self.logs.push(LogEntry {
                level: "ERROR",
                color: Color32::from_rgb(235, 91, 91),
                message: format!("Could not load project: {error}"),
            }),
        }
    }

    fn project_bundle(&self) -> (Result<ProjectFile, String>, Vec<BundleAsset>) {
        let mut project = ProjectFile::new("ZeroFPS Project", "Scene 01", self.scene.tree.clone());
        for (key, value) in [
            ("editor.grid_spacing", self.grid_spacing.to_string()),
            ("editor.show_grid", self.show_grid.to_string()),
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
                    WorkspaceTab::Compositing => "compositing",
                }
                .into(),
            ),
        ] {
            project.project.properties.insert(key.into(), value);
        }
        for (key, value) in [
            (
                "compositor.setting.object_index",
                self.compositor_object_index.to_string(),
            ),
            (
                "compositor.setting.texture_channel",
                self.compositor_texture_channel.to_string(),
            ),
            (
                "compositor.setting.constant",
                self.compositor_constant.to_string(),
            ),
            (
                "compositor.setting.color",
                format!(
                    "{},{},{}",
                    self.compositor_color[0], self.compositor_color[1], self.compositor_color[2]
                ),
            ),
            (
                "compositor.setting.bezier",
                self.compositor_bezier_points
                    .iter()
                    .map(|point| format!("{},{}", point[0], point[1]))
                    .collect::<Vec<_>>()
                    .join(";"),
            ),
            (
                "compositor.setting.math_operation",
                self.compositor_math_operation.to_string(),
            ),
            (
                "compositor.setting.threshold",
                self.compositor_threshold.to_string(),
            ),
            (
                "compositor.setting.threshold_width",
                self.compositor_threshold_width.to_string(),
            ),
            (
                "compositor.setting.filter",
                self.compositor_filter.to_string(),
            ),
            (
                "compositor.setting.filter_radius",
                self.compositor_filter_radius.to_string(),
            ),
            (
                "compositor.setting.output_width",
                self.compositor_output_width.to_string(),
            ),
            (
                "compositor.setting.output_height",
                self.compositor_output_height.to_string(),
            ),
            (
                "compositor.setting.output_object_index",
                self.compositor_output_object_index.to_string(),
            ),
            (
                "compositor.setting.output_texture_channel",
                self.compositor_output_texture_channel.to_string(),
            ),
            (
                "compositor.setting.combine_mode",
                self.compositor_combine_mode.to_string(),
            ),
            (
                "compositor.setting.combine_operation",
                self.compositor_combine_operation.to_string(),
            ),
            (
                "compositor.setting.combine_alpha",
                self.compositor_combine_alpha.to_string(),
            ),
            (
                "compositor.setting.color_space_from",
                self.compositor_color_space_from.to_string(),
            ),
            (
                "compositor.setting.color_space_to",
                self.compositor_color_space_to.to_string(),
            ),
            (
                "compositor.setting.extract_channel",
                self.compositor_extract_channel.to_string(),
            ),
            (
                "compositor.setting.grayscale_mode",
                self.compositor_grayscale_mode.to_string(),
            ),
        ] {
            project.project.properties.insert(key.into(), value);
        }
        let referenced: BTreeSet<String> = project
            .scene
            .geometry
            .iter()
            .flat_map(|(_, node)| node.components.iter())
            .filter_map(|component| match component {
                Component::Model { asset } if !asset.starts_with("builtin:") => Some(asset.clone()),
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
                source: source_path,
                archive_path,
            });
        }
        rewrite_asset_paths(&mut project, &mapping);
        for (index, image) in self.compositor_images.iter().enumerate() {
            let source = PathBuf::from(&image.path);
            if !source.is_file() {
                return (
                    Err(format!(
                        "referenced compositor image does not exist: {}",
                        image.path
                    )),
                    Vec::new(),
                );
            }
            let filename = source
                .file_name()
                .and_then(|name| name.to_str())
                .map(safe_bundle_filename)
                .unwrap_or_else(|| "image.png".into());
            let archive_path = format!("assets/compositor/{index:04}-{filename}");
            project
                .project
                .properties
                .insert(format!("compositor.image.{index:04}"), archive_path.clone());
            files.push(BundleAsset {
                source,
                archive_path,
            });
        }
        for (index, position) in self.compositor_node_positions.iter().enumerate() {
            project.project.properties.insert(
                format!("compositor.node_position.{index:04}"),
                format!("{},{}", position.x, position.y),
            );
        }
        for (index, active) in self.compositor_node_active.iter().enumerate() {
            project.project.properties.insert(
                format!("compositor.node_active.{index:04}"),
                active.to_string(),
            );
        }
        for (index, (from, to, input)) in self.compositor_links.iter().enumerate() {
            project.project.properties.insert(
                format!("compositor.link.{index:04}"),
                format!("{from},{to},{input}"),
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
            Some("compositing") => WorkspaceTab::Compositing,
            _ => self.workspace_tab,
        };
        let unsigned = |key: &str| {
            properties
                .get(key)
                .and_then(|value| value.parse::<usize>().ok())
        };
        self.compositor_object_index =
            unsigned("compositor.setting.object_index").unwrap_or(self.compositor_object_index);
        self.compositor_texture_channel = unsigned("compositor.setting.texture_channel")
            .filter(|value| *value < 6)
            .unwrap_or(self.compositor_texture_channel);
        self.compositor_constant =
            number("compositor.setting.constant").unwrap_or(self.compositor_constant);
        if let Some(color) = properties
            .get("compositor.setting.color")
            .and_then(|value| parse_compositor_color(value))
        {
            self.compositor_color = color;
        }
        if let Some(points) = properties
            .get("compositor.setting.bezier")
            .and_then(|value| parse_compositor_bezier(value))
        {
            self.compositor_bezier_points = points;
        }
        self.compositor_math_operation = unsigned("compositor.setting.math_operation")
            .filter(|value| *value < 8)
            .unwrap_or(self.compositor_math_operation);
        self.compositor_threshold =
            number("compositor.setting.threshold").unwrap_or(self.compositor_threshold);
        self.compositor_threshold_width =
            number("compositor.setting.threshold_width").unwrap_or(self.compositor_threshold_width);
        self.compositor_filter = unsigned("compositor.setting.filter")
            .filter(|value| *value < 7)
            .unwrap_or(self.compositor_filter);
        self.compositor_filter_radius =
            number("compositor.setting.filter_radius").unwrap_or(self.compositor_filter_radius);
        self.compositor_output_width = properties
            .get("compositor.setting.output_width")
            .and_then(|value| value.parse().ok())
            .unwrap_or(self.compositor_output_width);
        self.compositor_output_height = properties
            .get("compositor.setting.output_height")
            .and_then(|value| value.parse().ok())
            .unwrap_or(self.compositor_output_height);
        self.compositor_output_object_index = unsigned("compositor.setting.output_object_index")
            .unwrap_or(self.compositor_output_object_index);
        self.compositor_output_texture_channel =
            unsigned("compositor.setting.output_texture_channel")
                .filter(|value| *value < 6)
                .unwrap_or(self.compositor_output_texture_channel);
        self.compositor_combine_mode = unsigned("compositor.setting.combine_mode")
            .filter(|value| *value < 2)
            .unwrap_or(self.compositor_combine_mode);
        self.compositor_combine_operation = unsigned("compositor.setting.combine_operation")
            .filter(|value| *value < 8)
            .unwrap_or(self.compositor_combine_operation);
        self.compositor_combine_alpha =
            number("compositor.setting.combine_alpha").unwrap_or(self.compositor_combine_alpha);
        self.compositor_color_space_from = unsigned("compositor.setting.color_space_from")
            .filter(|value| *value < 2)
            .unwrap_or(self.compositor_color_space_from);
        self.compositor_color_space_to = unsigned("compositor.setting.color_space_to")
            .filter(|value| *value < 2)
            .unwrap_or(self.compositor_color_space_to);
        self.compositor_extract_channel = unsigned("compositor.setting.extract_channel")
            .filter(|value| *value < 4)
            .unwrap_or(self.compositor_extract_channel);
        self.compositor_grayscale_mode = unsigned("compositor.setting.grayscale_mode")
            .filter(|value| *value < 3)
            .unwrap_or(self.compositor_grayscale_mode);
        self.compositor_images = properties
            .iter()
            .filter(|(key, _)| key.starts_with("compositor.image."))
            .map(|(_, path)| CompositorImageAsset { path: path.clone() })
            .collect();
        for (key, value) in properties {
            if let Some(index) = key
                .strip_prefix("compositor.node_position.")
                .and_then(|value| value.parse::<usize>().ok())
                && let Some(position) = parse_compositor_position(value)
                && let Some(stored) = self.compositor_node_positions.get_mut(index)
            {
                *stored = position;
            }
            if let Some(index) = key
                .strip_prefix("compositor.node_active.")
                .and_then(|value| value.parse::<usize>().ok())
                && let Ok(active) = value.parse::<bool>()
                && let Some(stored) = self.compositor_node_active.get_mut(index)
            {
                *stored = active;
            }
        }
        self.compositor_node_active[8] = true;
        if !self.compositor_node_active[self.compositor_selected_node] {
            self.compositor_selected_node = 8;
        }
        let stored_links: Vec<_> = properties
            .iter()
            .filter_map(|(key, value)| {
                key.starts_with("compositor.link.")
                    .then(|| parse_compositor_link(value))
                    .flatten()
            })
            .filter(|(from, to, input)| {
                from != to
                    && *from < self.compositor_node_positions.len()
                    && *to < self.compositor_node_positions.len()
                    && *input < compositor_input_count(*to, self.compositor_combine_mode)
            })
            .collect();
        if properties
            .keys()
            .any(|key| key.starts_with("compositor.link."))
        {
            self.compositor_links = stored_links;
        }
    }

    fn compositor_node_controls_ui(&mut self, node: usize, ui: &mut egui::Ui) {
        let mut changed = false;
        match node {
            0 => {
                let objects: Vec<String> = self
                    .scene
                    .tree
                    .iter()
                    .map(|(_, node)| node.name.clone())
                    .collect();
                self.compositor_object_index = self
                    .compositor_object_index
                    .min(objects.len().saturating_sub(1));
                ui.label("Object");
                egui::ComboBox::from_id_salt("compositor_object")
                    .selected_text(
                        objects
                            .get(self.compositor_object_index)
                            .map(String::as_str)
                            .unwrap_or("No scene objects"),
                    )
                    .show_ui(ui, |ui| {
                        for (index, name) in objects.iter().enumerate() {
                            changed |= ui
                                .selectable_value(
                                    &mut self.compositor_object_index,
                                    index,
                                    name.as_str(),
                                )
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
                egui::ComboBox::from_id_salt("compositor_texture_channel")
                    .selected_text(channels[self.compositor_texture_channel])
                    .show_ui(ui, |ui| {
                        for (index, channel) in channels.iter().enumerate() {
                            changed |= ui
                                .selectable_value(
                                    &mut self.compositor_texture_channel,
                                    index,
                                    *channel,
                                )
                                .changed();
                        }
                    });
            }
            1 => {
                ui.label("External texture");
                ui.horizontal(|ui| {
                    let mut path = self
                        .compositor_images
                        .first()
                        .map(|image| image.path.clone())
                        .unwrap_or_default();
                    ui.add_enabled(
                        false,
                        egui::TextEdit::singleline(&mut path)
                            .hint_text("Choose PNG or JPEG")
                            .desired_width(125.0),
                    );
                    if ui
                        .add_enabled(
                            self.compositor_image_dialog_result.is_none(),
                            egui::Button::new("Browse…"),
                        )
                        .clicked()
                    {
                        self.start_compositor_image_import(ui.ctx());
                    }
                });
                ui.label("Color space");
                ui.label("Auto (from image metadata)");
            }
            2 => {
                ui.label("Scalar");
                changed |= ui
                    .add(egui::Slider::new(&mut self.compositor_constant, 0.0..=1.0))
                    .changed();
                ui.label("RGB color");
                changed |= ui
                    .color_edit_button_rgb(&mut self.compositor_color)
                    .changed();
                ui.horizontal(|ui| {
                    for channel in &mut self.compositor_color {
                        changed |= ui
                            .add(egui::DragValue::new(channel).range(0.0..=1.0).speed(0.01))
                            .changed();
                    }
                });
            }
            3 => {
                ui.label("RGB transfer curve");
                changed |= bezier_editor(ui, &mut self.compositor_bezier_points);
                ui.small("Drag the two inner points; endpoints remain at (0,0) and (1,1).");
            }
            4 => {
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
                egui::ComboBox::from_id_salt("compositor_math_operation")
                    .selected_text(operations[self.compositor_math_operation])
                    .show_ui(ui, |ui| {
                        for (index, operation) in operations.iter().enumerate() {
                            changed |= ui
                                .selectable_value(
                                    &mut self.compositor_math_operation,
                                    index,
                                    *operation,
                                )
                                .changed();
                        }
                    });
                ui.label("Fallback value");
                changed |= ui
                    .add(egui::DragValue::new(&mut self.compositor_constant).speed(0.01))
                    .changed();
            }
            5 => {
                ui.label("Threshold");
                changed |= ui
                    .add(egui::Slider::new(&mut self.compositor_threshold, 0.0..=1.0))
                    .changed();
                ui.small("Values below the threshold become 0; values above it become 1.");
            }
            6 => {
                ui.label("Threshold");
                changed |= ui
                    .add(egui::Slider::new(&mut self.compositor_threshold, 0.0..=1.0))
                    .changed();
                ui.label("Transition width");
                changed |= ui
                    .add(egui::Slider::new(
                        &mut self.compositor_threshold_width,
                        0.0..=1.0,
                    ))
                    .changed();
            }
            7 => {
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
                egui::ComboBox::from_id_salt("compositor_filter")
                    .selected_text(filters[self.compositor_filter])
                    .show_ui(ui, |ui| {
                        for (index, filter) in filters.iter().enumerate() {
                            changed |= ui
                                .selectable_value(&mut self.compositor_filter, index, *filter)
                                .changed();
                        }
                    });
                ui.label("Radius");
                changed |= ui
                    .add(
                        egui::DragValue::new(&mut self.compositor_filter_radius)
                            .range(0.0..=128.0)
                            .speed(0.25),
                    )
                    .changed();
            }
            8 => {
                let objects: Vec<String> = self
                    .scene
                    .tree
                    .iter()
                    .map(|(_, node)| node.name.clone())
                    .collect();
                self.compositor_output_object_index = self
                    .compositor_output_object_index
                    .min(objects.len().saturating_sub(1));
                ui.label("Target object");
                egui::ComboBox::from_id_salt("compositor_output_object")
                    .selected_text(
                        objects
                            .get(self.compositor_output_object_index)
                            .map(String::as_str)
                            .unwrap_or("No scene objects"),
                    )
                    .show_ui(ui, |ui| {
                        for (index, name) in objects.iter().enumerate() {
                            changed |= ui
                                .selectable_value(
                                    &mut self.compositor_output_object_index,
                                    index,
                                    name,
                                )
                                .changed();
                        }
                    });
                let channels = [
                    "Base Color",
                    "Normal",
                    "Roughness",
                    "Metallic",
                    "Emissive",
                    "Occlusion",
                ];
                ui.label("Overwrite texture");
                egui::ComboBox::from_id_salt("compositor_output_channel")
                    .selected_text(channels[self.compositor_output_texture_channel])
                    .show_ui(ui, |ui| {
                        for (index, channel) in channels.iter().enumerate() {
                            changed |= ui
                                .selectable_value(
                                    &mut self.compositor_output_texture_channel,
                                    index,
                                    *channel,
                                )
                                .changed();
                        }
                    });
                ui.small("The connected image replaces this texture when the graph is applied.");
            }
            9 => {
                ui.label("Combine mode");
                let modes = ["Algebra", "Mix"];
                egui::ComboBox::from_id_salt("compositor_combine_mode")
                    .selected_text(modes[self.compositor_combine_mode])
                    .show_ui(ui, |ui| {
                        for (index, mode) in modes.iter().enumerate() {
                            let mode_changed = ui
                                .selectable_value(&mut self.compositor_combine_mode, index, *mode)
                                .changed();
                            changed |= mode_changed;
                            if mode_changed && self.compositor_combine_mode == 0 {
                                self.compositor_links
                                    .retain(|(_, to, input)| *to != 9 || *input != 2);
                            }
                        }
                    });
                if self.compositor_combine_mode == 0 {
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
                    egui::ComboBox::from_id_salt("compositor_combine_operation")
                        .selected_text(operations[self.compositor_combine_operation])
                        .show_ui(ui, |ui| {
                            for (index, operation) in operations.iter().enumerate() {
                                changed |= ui
                                    .selectable_value(
                                        &mut self.compositor_combine_operation,
                                        index,
                                        *operation,
                                    )
                                    .changed();
                            }
                        });
                } else {
                    ui.label("Alpha");
                    let alpha_connected = self
                        .compositor_links
                        .iter()
                        .any(|(_, to, input)| *to == 9 && *input == 2);
                    changed |= ui
                        .add_enabled(
                            !alpha_connected,
                            egui::DragValue::new(&mut self.compositor_combine_alpha)
                                .range(0.0..=1.0)
                                .speed(0.01),
                        )
                        .changed();
                    if alpha_connected {
                        ui.small("Driven by the Alpha input socket.");
                    } else {
                        ui.small("Fallback used while Alpha is not connected.");
                    }
                    ui.small("Result = alpha × A + (1 − alpha) × B");
                }
            }
            10 => {
                let spaces = ["sRGB", "Linear RGB"];
                ui.label("From");
                egui::ComboBox::from_id_salt("compositor_color_space_from")
                    .selected_text(spaces[self.compositor_color_space_from])
                    .show_ui(ui, |ui| {
                        for (index, space) in spaces.iter().enumerate() {
                            changed |= ui
                                .selectable_value(
                                    &mut self.compositor_color_space_from,
                                    index,
                                    *space,
                                )
                                .changed();
                        }
                    });
                ui.label("To");
                egui::ComboBox::from_id_salt("compositor_color_space_to")
                    .selected_text(spaces[self.compositor_color_space_to])
                    .show_ui(ui, |ui| {
                        for (index, space) in spaces.iter().enumerate() {
                            changed |= ui
                                .selectable_value(
                                    &mut self.compositor_color_space_to,
                                    index,
                                    *space,
                                )
                                .changed();
                        }
                    });
            }
            11 => {
                let channels = ["Red", "Green", "Blue", "Alpha"];
                ui.label("Extract channel");
                egui::ComboBox::from_id_salt("compositor_extract_channel")
                    .selected_text(channels[self.compositor_extract_channel])
                    .show_ui(ui, |ui| {
                        for (index, channel) in channels.iter().enumerate() {
                            changed |= ui
                                .selectable_value(
                                    &mut self.compositor_extract_channel,
                                    index,
                                    *channel,
                                )
                                .changed();
                        }
                    });
            }
            12 => {
                let modes = ["Luminance (Rec. 709)", "Average", "Lightness"];
                ui.label("Conversion");
                egui::ComboBox::from_id_salt("compositor_grayscale_mode")
                    .selected_text(modes[self.compositor_grayscale_mode])
                    .show_ui(ui, |ui| {
                        for (index, mode) in modes.iter().enumerate() {
                            changed |= ui
                                .selectable_value(&mut self.compositor_grayscale_mode, index, *mode)
                                .changed();
                        }
                    });
            }
            _ => {
                ui.label("Inputs");
                ui.small("R, G, B and A are joined into one RGBA image.");
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
                    for title in ["Scene", "Build", "Window"] {
                        ui.menu_button(title, |ui| {
                            ui.label(format!("{title} commands"));
                            ui.small("Prototype surface");
                        });
                    }
                    ui.menu_button("Settings", |ui| {
                        ui.strong("Viewport");
                        ui.checkbox(&mut self.show_grid, "Show grid");
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
                    });
                });
                ui.separator();
                ui.horizontal(|ui| {
                    ui.selectable_value(&mut self.workspace_tab, WorkspaceTab::Scene, "Scene");
                    ui.selectable_value(
                        &mut self.workspace_tab,
                        WorkspaceTab::Compositing,
                        "Compositing",
                    );
                    ui.separator();
                    if self.workspace_tab == WorkspaceTab::Scene {
                        tool_button(ui, &mut self.active_tool, Tool::Select, "Q", "Select");
                        tool_button(ui, &mut self.active_tool, Tool::Move, "W", "Move");
                        tool_button(ui, &mut self.active_tool, Tool::Rotate, "E", "Rotate");
                        tool_button(ui, &mut self.active_tool, Tool::Scale, "R", "Scale");
                        ui.separator();
                        ui.toggle_value(&mut self.snap, "⌗ Snap");
                        ui.toggle_value(&mut self.show_grid, "Grid");
                    } else {
                        ui.label(RichText::new("Node Graph").weak().small());
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
                let roots = self.scene.tree.roots().to_vec();
                for root in roots {
                    self.object_tree(ui, root, 0);
                }
                ui.add_space(10.0);
                if ui.button("+ Add object").clicked() {
                    let previous = self.scene.tree.clone();
                    let id = self.scene.add("New Object", ObjectKind::Empty, None);
                    self.scene.selected = Some(id);
                    self.record_undo(previous);
                }
            });
    }

    fn object_tree(&mut self, ui: &mut egui::Ui, id: NodeId, depth: usize) {
        let Ok(object) = self.scene.tree.node(id) else {
            return;
        };
        let name = object.name.clone();
        let children = object.children().to_vec();
        let kind = self.scene.kind(id);
        ui.horizontal(|ui| {
            ui.add_space(depth as f32 * 13.0);
            if !children.is_empty() {
                ui.small("⌄");
            } else {
                ui.add_space(11.0);
            }
            let selected = self.scene.selected == Some(id);
            if ui
                .selectable_label(selected, format!("{}  {}", kind.icon(), name))
                .clicked()
            {
                self.scene.selected = Some(id);
            }
        });
        for child in children {
            self.object_tree(ui, child, depth + 1);
        }
    }

    fn inspector(&mut self, ctx: &egui::Context) {
        egui::SidePanel::right("inspector")
            .resizable(true)
            .default_width(300.0)
            .width_range(240.0..=430.0)
            .frame(panel_frame(Color32::from_rgb(28, 30, 37)))
            .show(ctx, |ui| {
                section_title(ui, "INSPECTOR", "⋮");
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
                            let mut transform = node.local_transform();
                            let previous = self.scene.tree.clone();
                            let mut changed = false;
                            let kind = self.scene.kind(id);
                            let model_asset =
                                node.components
                                    .iter()
                                    .find_map(|component| match component {
                                        Component::Model { asset } => Some(asset.clone()),
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
                            egui::CollapsingHeader::new(RichText::new("Mesh Renderer").strong())
                                .default_open(true)
                                .show(ui, |ui| {
                                    property(
                                        ui,
                                        "Mesh",
                                        model_asset.as_deref().unwrap_or("No mesh assigned"),
                                    );
                                    if model_asset.is_some() {
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
                                self.record_undo(previous);
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
                    BottomTab::Telemetry => telemetry_panel(ui, self.play_state),
                }
            });
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

    fn activate_compositor_node(&mut self, index: usize) {
        self.compositor_node_active[index] = true;
        self.compositor_selected_node = index;
        self.compositor_pending_spawn = Some(index);
        self.compositor_pending_output = None;
        self.project_dirty = true;
    }

    fn compositor_input_source(&self, node: usize, input: usize) -> Result<usize, String> {
        self.compositor_links
            .iter()
            .find_map(|(from, to, target_input)| {
                (*to == node && *target_input == input).then_some(*from)
            })
            .ok_or_else(|| format!("input {} on node {} is not connected", input + 1, node))
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
                        Component::Model { asset } => Some(asset.as_str()),
                        _ => None,
                    })
            })
    }

    fn object_node_id(&self, object_index: usize) -> Option<NodeId> {
        self.scene.tree.iter().nth(object_index).map(|(id, _)| id)
    }

    fn evaluate_compositor_node(
        &self,
        node: usize,
        visiting: &mut BTreeSet<usize>,
    ) -> Result<TextureAsset, String> {
        if !visiting.insert(node) {
            return Err("compositor graph contains a cycle".into());
        }
        let result = match node {
            0 => {
                let path = self
                    .object_asset_path(self.compositor_object_index)
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
                mesh.textures
                    .get(texture_name)
                    .cloned()
                    .ok_or_else(|| format!("texture `{texture_name}` is unavailable"))
            }
            1 => {
                let path = self
                    .compositor_images
                    .first()
                    .map(|image| image.path.as_str())
                    .ok_or("Image Asset has no imported image")?;
                let image = image::open(path)
                    .map_err(|error| format!("could not decode `{path}`: {error}"))?
                    .into_rgba8();
                Ok(TextureAsset {
                    name: PathBuf::from(path)
                        .file_name()
                        .and_then(|name| name.to_str())
                        .unwrap_or("compositor-image")
                        .into(),
                    width: image.width(),
                    height: image.height(),
                    pixels: image.into_raw(),
                })
            }
            2 => Ok(TextureAsset {
                name: "compositor-constant".into(),
                width: 1,
                height: 1,
                pixels: self
                    .compositor_color
                    .iter()
                    .map(|channel| (channel.clamp(0.0, 1.0) * 255.0).round() as u8)
                    .chain(std::iter::once(255))
                    .collect(),
            }),
            4 => {
                let source = self.compositor_input_source(node, 0)?;
                let texture = self.evaluate_compositor_node(source, visiting)?;
                Ok(apply_compositor_math(
                    texture,
                    self.compositor_math_operation,
                    self.compositor_constant,
                ))
            }
            3 | 5..=7 => {
                let source = self.compositor_input_source(node, 0)?;
                self.evaluate_compositor_node(source, visiting)
            }
            9 => {
                let a = self
                    .evaluate_compositor_node(self.compositor_input_source(node, 0)?, visiting)?;
                let b = self
                    .evaluate_compositor_node(self.compositor_input_source(node, 1)?, visiting)?;
                let alpha = if self.compositor_combine_mode == 1 {
                    self.compositor_links
                        .iter()
                        .find_map(|(from, to, input)| (*to == 9 && *input == 2).then_some(*from))
                        .map(|source| self.evaluate_compositor_node(source, visiting))
                        .transpose()?
                } else {
                    None
                };
                Ok(combine_compositor_textures(
                    &a,
                    &b,
                    alpha.as_ref(),
                    self.compositor_combine_mode,
                    self.compositor_combine_operation,
                    self.compositor_combine_alpha,
                ))
            }
            10 => {
                let source = self.compositor_input_source(node, 0)?;
                let texture = self.evaluate_compositor_node(source, visiting)?;
                Ok(convert_compositor_color_space(
                    texture,
                    self.compositor_color_space_from,
                    self.compositor_color_space_to,
                ))
            }
            11 => {
                let source = self.compositor_input_source(node, 0)?;
                let texture = self.evaluate_compositor_node(source, visiting)?;
                Ok(extract_compositor_channel(
                    texture,
                    self.compositor_extract_channel,
                ))
            }
            12 => {
                let source = self.compositor_input_source(node, 0)?;
                let texture = self.evaluate_compositor_node(source, visiting)?;
                Ok(grayscale_compositor_texture(
                    texture,
                    self.compositor_grayscale_mode,
                ))
            }
            13 => {
                let mut channels = Vec::with_capacity(4);
                for input in 0..3 {
                    channels.push(self.evaluate_compositor_node(
                        self.compositor_input_source(node, input)?,
                        visiting,
                    )?);
                }
                let alpha = self
                    .compositor_links
                    .iter()
                    .find_map(|(from, to, input)| (*to == 13 && *input == 3).then_some(*from))
                    .map(|source| self.evaluate_compositor_node(source, visiting))
                    .transpose()?;
                Ok(join_compositor_channels(
                    [&channels[0], &channels[1], &channels[2]],
                    alpha.as_ref(),
                ))
            }
            _ => Err(format!("node {node} cannot be evaluated as an image")),
        };
        visiting.remove(&node);
        result
    }

    fn apply_compositor(&mut self) {
        let result = (|| {
            if self.compositor_output_texture_channel != 0 {
                return Err("only Base Color output is implemented in this build".into());
            }
            let source = self.compositor_input_source(8, 0)?;
            let texture = self.evaluate_compositor_node(source, &mut BTreeSet::new())?;
            let target = self
                .object_node_id(self.compositor_output_object_index)
                .ok_or("Output has no scene object selected")?;
            if self
                .object_asset_path(self.compositor_output_object_index)
                .is_none()
            {
                return Err("Output target is not a model object".into());
            }
            if let Some((_, current)) = self
                .compositor_texture_overrides
                .iter_mut()
                .find(|(id, _)| *id == target)
            {
                *current = texture;
            } else {
                self.compositor_texture_overrides.push((target, texture));
            }
            let name = self
                .scene
                .tree
                .node(target)
                .map(|node| node.name.clone())
                .unwrap_or_else(|_| "model".into());
            Ok::<_, String>(name)
        })();
        match result {
            Ok(name) => {
                self.scene_revision = self.scene_revision.wrapping_add(1);
                self.logs.push(LogEntry {
                    level: "COMPOSITE",
                    color: Color32::from_rgb(112, 210, 156),
                    message: format!("Applied compositor output to `{name}`"),
                });
            }
            Err(message) => self.logs.push(LogEntry {
                level: "ERROR",
                color: Color32::from_rgb(235, 91, 91),
                message: format!("Could not apply compositor: {message}"),
            }),
        }
    }

    fn compositing_workspace(&mut self, ctx: &egui::Context) {
        egui::CentralPanel::default()
            .frame(panel_frame(Color32::from_rgb(18, 20, 26)))
            .show(ctx, |ui| {
                ui.horizontal(|ui| {
                    ui.strong("Compositing");
                    ui.separator();
                    ui.menu_button("Add", |ui| {
                        ui.menu_button("Input", |ui| {
                            for (index, label) in [
                                (0, "Object Texture"),
                                (1, "Image Asset"),
                                (2, "Constant Value"),
                            ] {
                                if compositor_add_button(
                                    ui,
                                    !self.compositor_node_active[index],
                                    label,
                                ) {
                                    self.activate_compositor_node(index);
                                }
                            }
                        });
                        ui.menu_button("Color", |ui| {
                            for (index, label) in [
                                (3, "Bézier Curves"),
                                (10, "Color Space Convert"),
                                (11, "Channel Extract"),
                                (13, "Channel Join"),
                                (12, "Grayscale"),
                            ] {
                                if compositor_add_button(
                                    ui,
                                    !self.compositor_node_active[index],
                                    label,
                                ) {
                                    self.activate_compositor_node(index);
                                }
                            }
                        });
                        ui.menu_button("Converter", |ui| {
                            for (index, label) in [
                                (4, "Texture Math"),
                                (9, "Texture Combine"),
                                (5, "Sharp Threshold"),
                                (6, "Smooth Threshold"),
                            ] {
                                if compositor_add_button(
                                    ui,
                                    !self.compositor_node_active[index],
                                    label,
                                ) {
                                    self.activate_compositor_node(index);
                                }
                            }
                        });
                        ui.menu_button("Filter", |ui| {
                            if compositor_add_button(
                                ui,
                                !self.compositor_node_active[7],
                                "Image Filter",
                            ) {
                                self.activate_compositor_node(7);
                            }
                        });
                        ui.separator();
                        ui.add_enabled(false, egui::Button::new("Output (always present)"));
                    });
                    ui.menu_button("Node", |ui| {
                        if ui
                            .add_enabled(
                                self.compositor_selected_node != 8,
                                egui::Button::new("Remove from graph"),
                            )
                            .clicked()
                        {
                            let removed = self.compositor_selected_node;
                            self.compositor_node_active[removed] = false;
                            self.compositor_links
                                .retain(|(from, to, _)| *from != removed && *to != removed);
                            self.compositor_pending_output = None;
                            self.compositor_selected_node = 8;
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
                    if ui.button("Apply").clicked() {
                        self.apply_compositor();
                    }
                    ui.separator();
                    if ui.button("Import Image…").clicked() {
                        self.start_compositor_image_import(ui.ctx());
                    }
                    ui.label(
                        RichText::new(format!("{} image assets", self.compositor_images.len()))
                            .weak()
                            .small(),
                    );
                    ui.separator();
                    ui.label(
                        RichText::new("Apply writes the graph result to the target Base Color")
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
                        self.compositor_zoom =
                            (self.compositor_zoom * (scroll * 0.0015).exp()).clamp(0.35, 2.5);
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
                        painter.line_segment(
                            [Pos2::new(x, canvas.top()), Pos2::new(x, canvas.bottom())],
                            Stroke::new(1.0, Color32::from_rgb(31, 34, 42)),
                        );
                        x += grid_spacing;
                    }
                    let mut y = canvas.top() + offset_y;
                    while y < canvas.bottom() {
                        painter.line_segment(
                            [Pos2::new(canvas.left(), y), Pos2::new(canvas.right(), y)],
                            Stroke::new(1.0, Color32::from_rgb(31, 34, 42)),
                        );
                        y += grid_spacing;
                    }
                }

                let origin = canvas.min + Vec2::new(70.0, 100.0) + self.compositor_pan;
                let scale = self.compositor_zoom;
                let node_specs = [
                    ("Object Texture", "Input", Color32::from_rgb(76, 122, 155)),
                    ("Image Asset", "Input", Color32::from_rgb(76, 122, 155)),
                    ("Constant Value", "Input", Color32::from_rgb(76, 122, 155)),
                    ("Bézier Curves", "Color", Color32::from_rgb(122, 88, 151)),
                    (
                        "Texture Math",
                        "Converter",
                        Color32::from_rgb(105, 112, 122),
                    ),
                    (
                        "Sharp Threshold",
                        "Converter",
                        Color32::from_rgb(105, 112, 122),
                    ),
                    (
                        "Smooth Threshold",
                        "Converter",
                        Color32::from_rgb(105, 112, 122),
                    ),
                    ("Image Filter", "Filter", Color32::from_rgb(92, 128, 92)),
                    ("Output", "Texture Writer", Color32::from_rgb(128, 113, 72)),
                    (
                        "Texture Combine",
                        "Converter",
                        Color32::from_rgb(105, 112, 122),
                    ),
                    (
                        "Color Space Convert",
                        "Color",
                        Color32::from_rgb(122, 88, 151),
                    ),
                    (
                        "Channel Extract",
                        "Color",
                        Color32::from_rgb(122, 88, 151),
                    ),
                    ("Grayscale", "Color", Color32::from_rgb(122, 88, 151)),
                    (
                        "Channel Join",
                        "Color",
                        Color32::from_rgb(122, 88, 151),
                    ),
                ];
                let node_heights = [
                    205.0, 165.0, 215.0, 300.0, 175.0, 150.0, 185.0, 175.0, 220.0, 220.0,
                    175.0, 140.0, 140.0, 165.0,
                ];
                let node_width = 230.0;
                if let Some(index) = self.compositor_pending_spawn.take() {
                    self.compositor_node_positions[index] = compositor_centered_position(
                        canvas,
                        origin,
                        scale,
                        Vec2::new(node_width, node_heights[index]),
                    );
                }
                let mut node_rects: Vec<_> = self
                    .compositor_node_positions
                    .iter()
                    .enumerate()
                    .map(|(index, position)| {
                        Rect::from_min_size(
                            origin + *position * scale,
                            Vec2::new(node_width, node_heights[index]) * scale,
                        )
                    })
                    .collect();

                let output_socket = |node: Rect| Pos2::new(node.right(), node.top() + 70.0 * scale);
                let combine_mode = self.compositor_combine_mode;
                let input_socket = |node: Rect, index: usize, input: usize| {
                    let y = if compositor_input_count(index, combine_mode) == 1 {
                        70.0
                    } else if index == 13 {
                        75.0 + input as f32 * 22.0
                    } else {
                        85.0 + input as f32 * 30.0
                    };
                    Pos2::new(node.left(), node.top() + y * scale)
                };
                let (pointer, primary_pressed, primary_down, primary_released) =
                    ui.input(|input| {
                        (
                            input.pointer.interact_pos(),
                            input.pointer.button_pressed(egui::PointerButton::Primary),
                            input.pointer.button_down(egui::PointerButton::Primary),
                            input.pointer.button_released(egui::PointerButton::Primary),
                        )
                    });
                if primary_pressed
                    && let Some(pointer) = pointer
                    && canvas.contains(pointer)
                {
                    let socket_radius = 12.0 * scale.max(0.7);
                    let mut handled = false;
                    for (index, node_rect) in node_rects.iter().enumerate() {
                        if !self.compositor_node_active[index] {
                            continue;
                        }
                        if pointer.distance(output_socket(*node_rect)) <= socket_radius {
                            self.compositor_pending_output =
                                (self.compositor_pending_output != Some(index)).then_some(index);
                            self.compositor_selected_node = index;
                            handled = true;
                            break;
                        }
                        for input in 0..compositor_input_count(index, combine_mode) {
                            if pointer.distance(input_socket(*node_rect, index, input))
                                <= socket_radius
                            {
                                if let Some(from) = self.compositor_pending_output.take() {
                                    if from != index {
                                        if let Some(link) =
                                            self.compositor_links.iter().position(|candidate| {
                                                *candidate == (from, index, input)
                                            })
                                        {
                                            self.compositor_links.remove(link);
                                        } else {
                                            self.compositor_links.retain(
                                                |(_, to, target_input)| {
                                                    *to != index || *target_input != input
                                                },
                                            );
                                            self.compositor_links.push((from, index, input));
                                        }
                                        self.project_dirty = true;
                                    }
                                } else {
                                    let old_len = self.compositor_links.len();
                                    self.compositor_links.retain(|(_, to, target_input)| {
                                        *to != index || *target_input != input
                                    });
                                    self.project_dirty |= old_len != self.compositor_links.len();
                                }
                                self.compositor_selected_node = index;
                                handled = true;
                                break;
                            }
                        }
                        if handled {
                            break;
                        }
                    }
                    if !handled {
                        for (index, node_rect) in node_rects.iter().enumerate().rev() {
                            if !self.compositor_node_active[index] {
                                continue;
                            }
                            let header_rect = Rect::from_min_size(
                                node_rect.min,
                                Vec2::new(node_rect.width(), 30.0 * scale),
                            );
                            if header_rect.contains(pointer) {
                                self.compositor_selected_node = index;
                                self.compositor_dragging_node =
                                    Some((index, pointer - node_rect.min));
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
                if primary_down
                    && let (Some(pointer), Some((index, grab_offset))) =
                        (pointer, self.compositor_dragging_node)
                {
                    self.compositor_node_positions[index] =
                        (pointer - grab_offset - origin) / scale;
                    node_rects[index] = Rect::from_min_size(
                        pointer - grab_offset,
                        Vec2::new(node_width, node_heights[index]) * scale,
                    );
                    ctx.request_repaint();
                }
                if primary_released && self.compositor_dragging_node.take().is_some() {
                    self.project_dirty = true;
                }

                for &(from, to, input) in &self.compositor_links {
                    if !self.compositor_node_active[from] || !self.compositor_node_active[to] {
                        continue;
                    }
                    let start = output_socket(node_rects[from]);
                    let end = input_socket(node_rects[to], to, input);
                    painter.add(egui::Shape::line(
                        compositor_link_curve(start, end),
                        Stroke::new(3.0, Color32::from_rgb(218, 190, 92)),
                    ));
                }
                if let (Some(from), Some(pointer)) = (self.compositor_pending_output, pointer) {
                    painter.add(egui::Shape::line(
                        compositor_link_curve(output_socket(node_rects[from]), pointer),
                        Stroke::new(2.0, Color32::from_rgb(108, 190, 255)),
                    ));
                }

                for (index, ((title, kind, header), node_rect)) in node_specs
                    .iter()
                    .zip(node_rects.iter().copied())
                    .enumerate()
                {
                    if !self.compositor_node_active[index] {
                        continue;
                    }
                    let selected = self.compositor_selected_node == index;
                    painter.rect_filled(
                        node_rect,
                        6.0,
                        if selected {
                            Color32::from_rgb(47, 50, 61)
                        } else {
                            Color32::from_rgb(36, 39, 48)
                        },
                    );
                    let header_rect = Rect::from_min_size(
                        node_rect.min,
                        Vec2::new(node_rect.width(), 30.0 * scale),
                    );
                    painter.rect_filled(header_rect, 6.0, *header);
                    let border = if selected {
                        Color32::from_rgb(108, 190, 255)
                    } else {
                        Color32::from_rgb(67, 71, 83)
                    };
                    for (a, b) in [
                        (node_rect.left_top(), node_rect.right_top()),
                        (node_rect.right_top(), node_rect.right_bottom()),
                        (node_rect.right_bottom(), node_rect.left_bottom()),
                        (node_rect.left_bottom(), node_rect.left_top()),
                    ] {
                        painter.line_segment([a, b], Stroke::new(2.0, border));
                    }
                    painter.text(
                        header_rect.left_center() + Vec2::new(9.0 * scale, 0.0),
                        Align2::LEFT_CENTER,
                        *title,
                        FontId::proportional(13.0 * scale),
                        Color32::WHITE,
                    );
                    painter.text(
                        node_rect.left_top() + Vec2::new(10.0, 45.0) * scale,
                        Align2::LEFT_TOP,
                        *kind,
                        FontId::proportional(11.0 * scale),
                        Color32::from_gray(166),
                    );
                    for input in 0..compositor_input_count(index, combine_mode) {
                        let position = input_socket(node_rect, index, input);
                        painter.circle_filled(
                            position,
                            6.0 * scale,
                            Color32::from_rgb(218, 190, 92),
                        );
                        if index == 9 {
                            painter.text(
                                position + Vec2::new(10.0 * scale, 0.0),
                                Align2::LEFT_CENTER,
                                ["A", "B", "Alpha"][input],
                                FontId::proportional(10.0 * scale),
                                Color32::from_gray(180),
                            );
                        } else if index == 13 {
                            painter.text(
                                position + Vec2::new(10.0 * scale, 0.0),
                                Align2::LEFT_CENTER,
                                ["R", "G", "B", "A"][input],
                                FontId::proportional(10.0 * scale),
                                Color32::from_gray(180),
                            );
                        }
                    }
                    painter.circle_filled(
                        output_socket(node_rect),
                        6.0 * scale,
                        Color32::from_rgb(218, 190, 92),
                    );
                    let layer_id = compositor_control_layer(ui.layer_id(), index);
                    ui.ctx().set_sublayer(ui.layer_id(), layer_id);
                    let controls_origin = node_rect.min + Vec2::new(10.0, 62.0) * scale;
                    let transform =
                        egui::emath::TSTransform::from_translation(controls_origin.to_vec2())
                            * egui::emath::TSTransform::from_scaling(scale);
                    ui.ctx().set_transform_layer(layer_id, transform);
                    let local_rect = Rect::from_min_size(
                        Pos2::ZERO,
                        Vec2::new(node_width - 20.0, node_heights[index] - 70.0),
                    );
                    let mut controls_ui = ui.new_child(
                        egui::UiBuilder::new()
                            .layer_id(layer_id)
                            .max_rect(local_rect)
                            .layout(Layout::top_down(Align::Min)),
                    );
                    controls_ui.set_clip_rect(local_rect);
                    controls_ui.style_mut().spacing.item_spacing.y = 3.0;
                    self.compositor_node_controls_ui(index, &mut controls_ui);
                }

                let palette = Rect::from_min_max(rect.min, Pos2::new(canvas.left(), rect.bottom()));
                painter.rect_filled(palette, 0.0, Color32::from_rgb(25, 27, 34));
                painter.text(
                    palette.left_top() + Vec2::new(12.0, 14.0),
                    Align2::LEFT_TOP,
                    "NODES",
                    FontId::proportional(12.0),
                    Color32::from_rgb(108, 190, 255),
                );
                for (index, label) in [
                    "Input / Object Texture",
                    "Input / Image Asset",
                    "Input / Constant Value",
                    "Color / Bézier Curves",
                    "Color / Color Space Convert",
                    "Color / Channel Extract",
                    "Color / Channel Join",
                    "Color / Grayscale",
                    "Converter / Texture Math",
                    "Converter / Texture Combine",
                    "Converter / Sharp Threshold",
                    "Converter / Smooth Threshold",
                    "Filter / Image Filter",
                    "Output / Texture Writer",
                ]
                .iter()
                .enumerate()
                {
                    painter.text(
                        palette.left_top() + Vec2::new(12.0, 46.0 + index as f32 * 27.0),
                        Align2::LEFT_TOP,
                        *label,
                        FontId::proportional(12.0),
                        Color32::from_gray(190),
                    );
                }

                let properties =
                    Rect::from_min_max(Pos2::new(canvas.right(), rect.top()), rect.max);
                painter.rect_filled(properties, 0.0, Color32::from_rgb(25, 27, 34));
                let mut info_ui = ui.new_child(
                    egui::UiBuilder::new()
                        .max_rect(properties.shrink(12.0))
                        .layout(Layout::top_down(Align::Min)),
                );
                info_ui.heading("Selection");
                info_ui.label(node_specs[self.compositor_selected_node].0);
                info_ui.small(format!(
                    "Position: {:.0}, {:.0}",
                    self.compositor_node_positions[self.compositor_selected_node].x,
                    self.compositor_node_positions[self.compositor_selected_node].y
                ));
                info_ui.separator();
                info_ui.small(
                    "Edit settings directly inside the node. Drag its colored header to move it; click sockets to connect or disconnect.",
                );
            });
    }

    fn viewport(&mut self, ctx: &egui::Context) {
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
                    });
                });
                let available = ui.available_size();
                let (response, painter) = ui.allocate_painter(available, Sense::click_and_drag());
                self.viewport_focused = response.hovered();
                while let Ok(input) = self.input_worker.actions.try_recv() {
                    self.camera_yaw += input.orbit.x * 0.002;
                    self.camera_pitch =
                        (self.camera_pitch + input.orbit.y * 0.002).clamp(-1.2, 1.2);
                    self.camera_zoom =
                        (self.camera_zoom * input.zoom_log.exp()).clamp(0.001, 100.0);
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
                let pointer_delta = ui.input(|input| input.pointer.delta());
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
                    self.input_worker.submit(InputSample {
                        orbit,
                        pan,
                        zoom_log,
                        viewport_extent: response.rect.width().min(response.rect.height()),
                    });
                }
                self.refresh_preview_cache();
                let preview = Arc::clone(&self.cached_preview);
                let viewport_texture = {
                    let key = DepthCacheKey {
                        size: [
                            response.rect.width().round().max(1.0) as usize,
                            response.rect.height().round().max(1.0) as usize,
                        ],
                        yaw: self.camera_yaw,
                        pitch: self.camera_pitch,
                        zoom: self.camera_zoom,
                        target: self.camera_target,
                        grid_spacing: self.grid_spacing,
                        projection: self.projection_mode,
                        scene_revision: self.scene_revision,
                        show_grid: self.show_grid,
                        mode: self.viewport_mode,
                        tool: self.active_tool,
                    };
                    let mut newest_completed = None;
                    while let Ok(result) = self.display_worker.results.try_recv() {
                        newest_completed = Some(result);
                    }
                    if let Some(result) = newest_completed {
                        self.viewport_depth = result.frame.linear_depth;
                        if let Some(texture) = &mut self.viewport_color {
                            texture.set(result.frame.color, TextureOptions::NEAREST);
                        } else {
                            self.viewport_color = Some(ctx.load_texture(
                                "viewport-depth-color",
                                result.frame.color,
                                TextureOptions::NEAREST,
                            ));
                        }
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
                    {
                        self.display_worker.submit_latest(RenderJob {
                            key,
                            viewport_size: response.rect.size(),
                            triangles: Arc::clone(&preview),
                            camera: (
                                self.camera_yaw,
                                self.camera_pitch,
                                self.camera_zoom,
                                self.camera_target,
                                self.grid_spacing,
                                self.projection_mode,
                            ),
                            show_grid: self.show_grid,
                            mode: self.viewport_mode,
                            tool: self.active_tool,
                            reusable_depth: std::mem::take(&mut self.viewport_depth),
                        });
                        self.viewport_requested_key = Some(key);
                    }
                    self.viewport_color.as_ref().map(TextureHandle::id)
                };
                if let Some(presented) = &self.presented_view {
                    draw_viewport(
                        &painter,
                        response.rect,
                        presented.show_grid,
                        presented.tool,
                        presented.camera,
                        &presented.triangles,
                        presented.mode,
                        viewport_texture,
                    );
                } else {
                    painter.rect_filled(response.rect, 0.0, Color32::from_rgb(21, 24, 31));
                }
            });
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
                                            Component::Model { asset } => self
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
                            ui.small(if self.viewport_focused {
                                "Viewport focused  •  RMB orbit  •  Wheel zoom"
                            } else {
                                "Ready"
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
        self.status_bar(ctx);
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

fn bezier_editor(ui: &mut egui::Ui, points: &mut [[f32; 2]]) -> bool {
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
    let curve: Vec<_> = (0..=48)
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
        .collect();
    painter.line(curve, Stroke::new(2.5, Color32::from_rgb(108, 190, 255)));
    for pair in points.windows(2) {
        painter.line_segment(
            [to_screen(pair[0]), to_screen(pair[1])],
            Stroke::new(1.0, Color32::from_gray(105)),
        );
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

fn compositor_input_count(node: usize, combine_mode: usize) -> usize {
    if node == 9 {
        if combine_mode == 1 { 3 } else { 2 }
    } else if node == 13 {
        4
    } else {
        1
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
    }
}

fn apply_compositor_math(
    mut texture: TextureAsset,
    operation: usize,
    fallback: f32,
) -> TextureAsset {
    for pixel in texture.pixels.chunks_exact_mut(4) {
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
    }
    texture.name = "compositor-math".into();
    texture
}

fn convert_compositor_color_space(
    mut texture: TextureAsset,
    from: usize,
    to: usize,
) -> TextureAsset {
    if from != to {
        for pixel in texture.pixels.chunks_exact_mut(4) {
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
        }
    }
    texture.name = "compositor-color-space".into();
    texture
}

fn extract_compositor_channel(mut texture: TextureAsset, channel: usize) -> TextureAsset {
    for pixel in texture.pixels.chunks_exact_mut(4) {
        let value = pixel[channel.min(3)];
        pixel[0] = value;
        pixel[1] = value;
        pixel[2] = value;
        pixel[3] = 255;
    }
    texture.name = "compositor-channel".into();
    texture
}

fn grayscale_compositor_texture(mut texture: TextureAsset, mode: usize) -> TextureAsset {
    for pixel in texture.pixels.chunks_exact_mut(4) {
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
    }
    texture.name = "compositor-grayscale".into();
    texture
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
    }
}

fn parse_compositor_link(value: &str) -> Option<(usize, usize, usize)> {
    let mut fields = value.split(',');
    let from = fields.next()?.parse().ok()?;
    let to = fields.next()?.parse().ok()?;
    let input = fields.next().map(str::parse).transpose().ok()?.unwrap_or(0);
    fields.next().is_none().then_some((from, to, input))
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
            if let Component::Model { asset } = component
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
        if key.starts_with("compositor.image.")
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

fn telemetry_panel(ui: &mut egui::Ui, state: PlayState) {
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
        ui.label("Frame  16.67 ms");
        ui.label("Tick  8.33 ms");
        ui.label("Objects  5");
        ui.label("Draws  2");
        ui.label("Socket  loopback");
    });
    ui.add_space(8.0);
    ui.small(
        "The compiled game process owns simulation. This panel receives bounded debug telemetry \
         while the editor remains active.",
    );
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
    vertices: [RasterVertex; 3],
    base_color: [f32; 4],
    light: f32,
    face_normal: CoreVec3,
    shader: ShaderMode,
    smooth_normals: bool,
    transmission: f32,
    ior: f32,
    texture: Option<Arc<TextureAsset>>,
    min_y: usize,
    max_y: usize,
}

fn rasterize_depth_frame(
    viewport_size: Vec2,
    triangles: &[PreviewTriangle],
    camera: (f32, f32, f32, CoreVec3, f32, ProjectionMode),
    mut linear_depth: Vec<f32>,
    workspace: &mut RasterWorkspace,
) -> DepthFrame {
    let width = viewport_size.x.round().max(1.0) as usize;
    let height = viewport_size.y.round().max(1.0) as usize;
    let size = [width, height];
    let mut color = ColorImage::new(size, Color32::TRANSPARENT);
    linear_depth.resize(width * height, f32::INFINITY);
    linear_depth.fill(f32::INFINITY);
    let (yaw, pitch, zoom, camera_target, grid_spacing, projection_mode) = camera;
    let center = Pos2::new(width as f32 * 0.5, height as f32 * 0.5 + 25.0);
    let scale = perspective_view_scale(
        viewport_size.x.min(viewport_size.y) * 0.18 * zoom,
        projection_mode,
        grid_spacing,
    );
    let light = CoreVec3::new(-0.35, 0.8, 0.45).normalized();
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
        let diffuse = normal.dot(light).abs();
        let band = shader_light_factor(diffuse, triangle.shader);
        clip_preview_polygon_to_near_into(
            triangle,
            yaw,
            pitch,
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
                    camera_target,
                    projection_mode,
                    grid_spacing,
                )?;
                let camera_depth = PERSPECTIVE_CAMERA_DISTANCE * grid_spacing
                    + view_depth(vertex.position, yaw, pitch, camera_target);
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
            workspace.prepared.push(PreparedRasterTriangle {
                vertices: raster_vertices,
                base_color: triangle.base_color,
                light: band,
                face_normal: normal,
                shader: triangle.shader,
                smooth_normals: triangle.smooth_normals,
                transmission: triangle.transmission,
                ior: triangle.ior,
                texture: triangle.texture.clone(),
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
                        light,
                        triangle.transmission,
                        triangle.ior,
                        camera_position,
                        triangle.texture.as_deref(),
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
        color,
        linear_depth,
    }
}

#[allow(clippy::too_many_arguments)]
fn rasterize_triangle_band(
    vertices: [RasterVertex; 3],
    base_color: [f32; 4],
    light: f32,
    face_normal: CoreVec3,
    shader: ShaderMode,
    smooth_normals: bool,
    light_direction: CoreVec3,
    transmission: f32,
    ior: f32,
    camera_position: CoreVec3,
    texture: Option<&TextureAsset>,
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
                    let pixel_light = if smooth_normals {
                        shader_light_factor(shading_normal.dot(light_direction).abs(), shader)
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
                        let world_position = CoreVec3::new(
                            interpolate(|vertex| vertex.world_position[0]),
                            interpolate(|vertex| vertex.world_position[1]),
                            interpolate(|vertex| vertex.world_position[2]),
                        );
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
    camera: (f32, f32, f32, CoreVec3, f32, ProjectionMode),
) -> DepthFrame {
    const TILE_SIZE: usize = 16;
    let width = viewport_size.x.round().max(1.0) as usize;
    let height = viewport_size.y.round().max(1.0) as usize;
    let size = [width, height];
    let mut color = ColorImage::new(size, Color32::TRANSPARENT);
    let mut linear_depth = vec![f32::INFINITY; width * height];
    if triangles.is_empty() {
        return DepthFrame {
            color,
            linear_depth,
        };
    }

    let (yaw, pitch, zoom, camera_target, grid_spacing, projection_mode) = camera;
    let center = Pos2::new(width as f32 * 0.5, height as f32 * 0.5 + 25.0);
    let scale = perspective_view_scale(
        viewport_size.x.min(viewport_size.y) * 0.18 * zoom,
        projection_mode,
        grid_spacing,
    );
    let light = CoreVec3::new(-0.35, 0.8, 0.45).normalized();
    let mut raster_triangles = Vec::with_capacity(triangles.len());
    for triangle in triangles {
        let points = triangle.map(|point| CoreVec3::new(point[0], point[1], point[2]));
        let normal = (points[1] - points[0])
            .cross(points[2] - points[0])
            .normalized();
        let diffuse = normal.dot(light).abs();
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
                .map(|point| view_depth(*point, yaw, pitch, camera_target))
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

    let (right, up, forward) = camera_basis(yaw, pitch);
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
        color,
        linear_depth,
    }
}

fn camera_basis(yaw: f32, pitch: f32) -> (CoreVec3, CoreVec3, CoreVec3) {
    let (sin_yaw, cos_yaw) = yaw.sin_cos();
    let (sin_pitch, cos_pitch) = pitch.sin_cos();
    (
        CoreVec3::new(cos_yaw, -sin_yaw, 0.0),
        CoreVec3::new(-sin_yaw * sin_pitch, -cos_yaw * sin_pitch, cos_pitch),
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

#[allow(clippy::too_many_arguments)]
fn draw_viewport(
    painter: &egui::Painter,
    rect: Rect,
    grid: bool,
    tool: Tool,
    camera: (f32, f32, f32, CoreVec3, f32, ProjectionMode),
    triangles: &[PreviewTriangle],
    mode: ViewportMode,
    viewport_texture: Option<TextureId>,
) {
    let (yaw, pitch, zoom, camera_target, grid_spacing, projection_mode) = camera;
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
    let (view_x, view_y, depth) = view_coordinates(point, yaw, pitch, camera_target);
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
        let (view_x, view_y, depth) = view_coordinates(direction, yaw, pitch, CoreVec3::ZERO);
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
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
) -> Option<Pos2> {
    let (x, y, depth) = view_coordinates(point, yaw, pitch, camera_target);
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
    camera_target: CoreVec3,
) -> (f32, f32, f32) {
    let point = [
        point[0] - camera_target.x,
        point[1] - camera_target.y,
        point[2] - camera_target.z,
    ];
    let (sy, cy) = yaw.sin_cos();
    let (sp, cp) = pitch.sin_cos();
    let x = point[0] * cy - point[1] * sy;
    let forward = point[0] * sy + point[1] * cy;
    let y = point[2] * cp - forward * sp;
    let depth = point[2] * sp + forward * cp;
    (x, y, depth)
}

fn view_depth(point: [f32; 3], yaw: f32, pitch: f32, camera_target: CoreVec3) -> f32 {
    view_coordinates(point, yaw, pitch, camera_target).2
}

fn near_distance(
    point: [f32; 3],
    yaw: f32,
    pitch: f32,
    camera_target: CoreVec3,
    perspective_unit: f32,
) -> f32 {
    PERSPECTIVE_CAMERA_DISTANCE * perspective_unit + view_depth(point, yaw, pitch, camera_target)
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
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
) -> Option<[[f32; 3]; 2]> {
    if projection_mode == ProjectionMode::Orthographic {
        return Some(segment);
    }
    let distances =
        segment.map(|point| near_distance(point, yaw, pitch, camera_target, perspective_unit));
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
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
) -> Option<[Pos2; 2]> {
    let clipped = clip_segment_to_near(
        segment,
        yaw,
        pitch,
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
    camera_target: CoreVec3,
    projection_mode: ProjectionMode,
    perspective_unit: f32,
) -> Vec<[f32; 3]> {
    let mut output = Vec::with_capacity(polygon.len() + 1);
    clip_polygon_to_near_into(
        polygon,
        yaw,
        pitch,
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
        let current_distance = near_distance(current, yaw, pitch, camera_target, perspective_unit);
        let previous_distance =
            near_distance(previous, yaw, pitch, camera_target, perspective_unit);
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
            camera_target,
            perspective_unit,
        );
        let previous_distance = near_distance(
            previous.position,
            yaw,
            pitch,
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
    fn bundle_paths_are_safe_and_rewrite_only_referenced_models() {
        assert_eq!(safe_bundle_filename("my model (1).glb"), "my_model__1_.glb");
        let mut tree = GeometryTree::new();
        let model = tree.create("model", None).unwrap();
        tree.add_component(
            model,
            Component::Model {
                asset: "/source/model.glb".into(),
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
            Component::Model { asset } if asset == "assets/0000/model.glb"
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
            CoreVec3::ZERO,
            ProjectionMode::Perspective,
            1.0,
        )
        .expect("crossing segment survives");
        assert!(
            (near_distance(crossing[0], 0.0, 0.0, CoreVec3::ZERO, 1.0) - PERSPECTIVE_NEAR).abs()
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
            CoreVec3::ZERO,
            ProjectionMode::Perspective,
            1.0,
        );
        assert_eq!(polygon.len(), 4);
        assert!(polygon.iter().all(|point| {
            near_distance(*point, 0.0, 0.0, CoreVec3::ZERO, 1.0) >= PERSPECTIVE_NEAR - 1.0e-4
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
    fn orthographic_grid_segment_matches_direct_projection() {
        let segment = [[-2.0, 1.0, 0.0], [4.0, 1.0, 0.0]];
        let projected = project_segment(
            segment,
            Pos2::ZERO,
            100.0,
            -0.4,
            0.5,
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
            vertices: positions.map(|position| PreviewVertex {
                position,
                normal: [0.0, 0.0, 1.0],
                uv: [0.0; 2],
                color: [1.0; 4],
            }),
            base_color: [0.42, 0.64, 0.78, 1.0],
            texture: None,
            shader: ShaderMode::Toon,
            smooth_normals: false,
            transmission: 0.0,
            ior: 1.5,
        };
        let triangles = [
            triangle([[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [0.0, 0.0, 2.0]]),
            triangle([[-2.0, 5.0, -2.0], [2.0, 5.0, -2.0], [0.0, 5.0, 2.0]]),
        ];
        let camera = (
            0.0,
            0.0,
            1.0,
            CoreVec3::ZERO,
            1.0,
            ProjectionMode::Perspective,
        );
        let mut workspace = RasterWorkspace::default();
        let frame = rasterize_depth_frame(
            Vec2::new(64.0, 64.0),
            &triangles,
            camera,
            Vec::new(),
            &mut workspace,
        );
        let reversed = rasterize_depth_frame(
            Vec2::new(64.0, 64.0),
            &[triangles[1].clone(), triangles[0].clone()],
            camera,
            Vec::new(),
            &mut workspace,
        );
        assert_eq!(frame.linear_depth.len(), 64 * 64);
        assert_eq!(frame.linear_depth, reversed.linear_depth);
        assert_eq!(frame.color, reversed.color);
        let center = 57 * 64 + 32;
        assert!(frame.linear_depth[center].is_finite());
        assert!((frame.linear_depth[center] - PERSPECTIVE_CAMERA_DISTANCE).abs() < 0.1);
        assert_ne!(frame.color.pixels[center], Color32::TRANSPARENT);
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
        assert_eq!(parse_compositor_link("3,7"), Some((3, 7, 0)));
        assert_eq!(parse_compositor_link("3,9,2"), Some((3, 9, 2)));
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
        assert_eq!(compositor_input_count(9, 0), 2);
        assert_eq!(compositor_input_count(9, 1), 3);
        assert_eq!(compositor_input_count(13, 0), 4);
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
    fn compositor_mix_uses_optional_alpha_formula() {
        let texture = |name: &str, red: u8| TextureAsset {
            name: name.into(),
            width: 1,
            height: 1,
            pixels: vec![red, red, red, 255],
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
    fn compositor_math_multiply_zero_blacks_rgb_and_preserves_alpha() {
        let texture = TextureAsset {
            name: "earth".into(),
            width: 1,
            height: 1,
            pixels: vec![210, 120, 45, 173],
        };
        let result = apply_compositor_math(texture, 2, 0.0);
        assert_eq!(result.pixels, vec![0, 0, 0, 173]);
    }

    #[test]
    fn compositor_channels_extract_join_and_grayscale() {
        let source = TextureAsset {
            name: "source".into(),
            width: 1,
            height: 1,
            pixels: vec![30, 90, 210, 128],
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
        };
        let linear = convert_compositor_color_space(source.clone(), 0, 1);
        let round_trip = convert_compositor_color_space(linear, 1, 0);
        for (actual, expected) in round_trip.pixels.iter().zip(source.pixels) {
            assert!((*actual as i16 - expected as i16).abs() <= 4);
        }
    }
}
