//! ZeroFPS scene editor prototype.
//!
//! This executable intentionally keeps an editor-local authoring model.  The
//! model is a thin adapter which can later be replaced by `zerofps-core`
//! handles without coupling the UI to runtime ownership.

use std::{
    path::PathBuf,
    sync::{Arc, Condvar, Mutex, mpsc},
    time::{Duration, Instant},
};

use eframe::egui::{
    self, Align, Align2, Color32, ColorImage, FontId, Id, Key, Layout, Pos2, Rect, RichText, Sense,
    Stroke, TextureHandle, TextureId, TextureOptions, Vec2,
};
use zerofps_assets::{MeshAsset, import_file};
use zerofps_core::{
    Attribute, AttributeDeclaration, AttributeKey, Component, GeometryTree, NodeId, Quat,
    Vec3 as CoreVec3,
};
use zerofps_formats::ProjectFile;

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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum ViewportMode {
    Shaded,
    ShadedEdges,
    Wireframe,
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
}

struct DepthFrame {
    color: ColorImage,
    linear_depth: Vec<f32>,
}

struct RenderJob {
    key: DepthCacheKey,
    viewport_size: Vec2,
    triangles: Arc<Vec<[[f32; 3]; 3]>>,
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
    triangles: Arc<Vec<[[f32; 3]; 3]>>,
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
    triangles: Arc<Vec<[[f32; 3]; 3]>>,
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
    clipped: Vec<[f32; 3]>,
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
    active_tool: Tool,
    play_state: PlayState,
    build_started: Option<Instant>,
    bottom_tab: BottomTab,
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
    project_dirty: bool,
    undo_stack: Vec<GeometryTree>,
    redo_stack: Vec<GeometryTree>,
    asset_import_path: String,
    imported_assets: Vec<ImportedAsset>,
    viewport_mode: ViewportMode,
    projection_mode: ProjectionMode,
    viewport_color: Option<TextureHandle>,
    viewport_depth: Vec<f32>,
    viewport_depth_key: Option<DepthCacheKey>,
    scene_revision: u64,
    cached_preview_revision: u64,
    cached_preview: Arc<Vec<[[f32; 3]; 3]>>,
    display_worker: DisplayWorker,
    input_worker: InputWorker,
    viewport_requested_key: Option<DepthCacheKey>,
    presented_view: Option<PresentedView>,
    dialog_result: Option<mpsc::Receiver<Option<PathBuf>>>,
}

impl EditorApp {
    fn new(cc: &eframe::CreationContext<'_>) -> Self {
        configure_style(&cc.egui_ctx);
        Self {
            scene: EditorScene::default(),
            active_tool: Tool::Move,
            play_state: PlayState::Editing,
            build_started: None,
            bottom_tab: BottomTab::Assets,
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
            project_path: PathBuf::from("zerofps.project.json"),
            project_dirty: false,
            undo_stack: Vec::new(),
            redo_stack: Vec::new(),
            asset_import_path: String::new(),
            imported_assets: Vec::new(),
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
            viewport_requested_key: None,
            presented_view: None,
            dialog_result: None,
        }
    }

    fn import_asset(&mut self, add_to_scene: bool) {
        let path = self.asset_import_path.trim().to_owned();
        if path.is_empty() {
            return;
        }
        match import_file(&path) {
            Ok(asset) => {
                let triangle_count = asset.triangle_count();
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
                    index
                } else {
                    self.imported_assets
                        .push(ImportedAsset { path, mesh: asset });
                    self.imported_assets.len() - 1
                };
                self.scene_revision = self.scene_revision.wrapping_add(1);
                self.asset_import_path.clear();
                if add_to_scene {
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
            self.scene.selected = Some(id);
            self.record_undo(previous);
            self.logs.push(LogEntry {
                level: "SCENE",
                color: Color32::from_rgb(103, 191, 255),
                message: format!("Added `{object_name}` to the scene"),
            });
        }
    }

    fn build_preview_triangles(&self) -> Vec<[[f32; 3]; 3]> {
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
        for (_, node) in self.scene.tree.iter() {
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
            let Some(asset) = self
                .imported_assets
                .iter()
                .find(|asset| asset.path == path)
                .map(|asset| &asset.mesh)
            else {
                continue;
            };
            let transform = node.global_transform();
            for primitive in &asset.primitives {
                for triangle in primitive.indices.chunks_exact(3) {
                    let mut points = [[0.0; 3]; 3];
                    for (destination, index) in points.iter_mut().zip(triangle) {
                        let source = asset.vertices[*index as usize].position;
                        let local = CoreVec3::new(source[0], source[1], source[2]);
                        let world = transform
                            .rotation
                            .rotate(transform.scale.component_mul(local))
                            + transform.translation;
                        *destination = [world.x, world.y, world.z];
                    }
                    output.push(points);
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
        let project = ProjectFile::new("ZeroFPS Project", "Scene 01", self.scene.tree.clone());
        match project.save(&self.project_path) {
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

    fn load_project(&mut self) {
        match ProjectFile::load(&self.project_path) {
            Ok(project) => {
                self.scene.tree = project.scene.geometry;
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
        if ctx.input_mut(|i| {
            i.consume_shortcut(&egui::KeyboardShortcut::new(egui::Modifiers::CTRL, Key::S))
        }) {
            self.save_project();
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
                            self.save_project();
                            ui.close_menu();
                        }
                        if ui.button("Load project").clicked() {
                            self.load_project();
                            ui.close_menu();
                        }
                        ui.separator();
                        ui.small(self.project_path.display().to_string());
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
                        ui.label(
                            RichText::new(if self.project_dirty {
                                "Scene 01  •  Modified"
                            } else {
                                "Scene 01  •  Saved"
                            })
                            .weak()
                            .small(),
                        );
                    });
                });
                ui.separator();
                ui.horizontal(|ui| {
                    tool_button(ui, &mut self.active_tool, Tool::Select, "Q", "Select");
                    tool_button(ui, &mut self.active_tool, Tool::Move, "W", "Move");
                    tool_button(ui, &mut self.active_tool, Tool::Rotate, "E", "Rotate");
                    tool_button(ui, &mut self.active_tool, Tool::Scale, "R", "Scale");
                    ui.separator();
                    ui.toggle_value(&mut self.snap, "⌗ Snap");
                    ui.toggle_value(&mut self.show_grid, "Grid");

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
                let model_asset = node
                    .components
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
                    ui.add(egui::TextEdit::singleline(&mut name).desired_width(f32::INFINITY));
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
                        let mut scale = [transform.scale.x, transform.scale.y, transform.scale.z];
                        vector_editor(ui, "Position", &mut position, f64::from(self.grid_spacing));
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
                        ui.small(RichText::new("Inherited through GeometryTree").weak());
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
                        property(ui, "Material", "materials/default");
                        inherited_property(ui, "Visibility", "Visible", "Environment");
                        inherited_property(ui, "Layer", "Default", "Project");
                    });
                egui::CollapsingHeader::new(RichText::new("Script Components").strong())
                    .default_open(true)
                    .show(ui, |ui| {
                        ui.label("◇  spin_controller.rs");
                        ui.small(
                            RichText::new("All attributes script-readable and writable").weak(),
                        );
                    });
                if self.advanced {
                    egui::CollapsingHeader::new("Advanced")
                        .default_open(true)
                        .show(ui, |ui| {
                            property(ui, "Stable ID", &format!("{}:{}", id.slot, id.generation));
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
            ui.label("Import OBJ / PLY / STL");
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
                            .add_filter("Supported 3D models", &["obj", "ply", "stl"])
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
        self.poll_build(ctx);
        self.shortcuts(ctx);
        self.top_bar(ctx);
        self.hierarchy(ctx);
        self.inspector(ctx);
        self.bottom_panel(ctx);
        self.viewport(ctx);
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
            ui.label(value);
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
}

#[derive(Clone, Copy)]
struct PreparedRasterTriangle {
    vertices: [RasterVertex; 3],
    color: Color32,
    min_y: usize,
    max_y: usize,
}

fn rasterize_depth_frame(
    viewport_size: Vec2,
    triangles: &[[[f32; 3]; 3]],
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
    workspace.prepared.clear();
    workspace.clipped.clear();
    workspace.projected.clear();
    workspace.prepared.reserve(triangles.len());
    workspace.clipped.reserve(4);
    workspace.projected.reserve(4);
    for triangle in triangles {
        let world = triangle.map(|point| CoreVec3::new(point[0], point[1], point[2]));
        let normal = (world[1] - world[0])
            .cross(world[2] - world[0])
            .normalized();
        let diffuse = normal.dot(light).abs();
        let band = ((0.25 + diffuse * 0.75) * 3.0).round() / 3.0;
        let triangle_color = Color32::from_rgb(
            (61.0 + 72.0 * band) as u8,
            (92.0 + 96.0 * band) as u8,
            (118.0 + 103.0 * band) as u8,
        );
        clip_polygon_to_near_into(
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
            .extend(workspace.clipped.iter().filter_map(|point| {
                let position = project(
                    *point,
                    center,
                    scale,
                    yaw,
                    pitch,
                    camera_target,
                    projection_mode,
                    grid_spacing,
                )?;
                let camera_depth = PERSPECTIVE_CAMERA_DISTANCE * grid_spacing
                    + view_depth(*point, yaw, pitch, camera_target);
                (position.x.is_finite()
                    && position.y.is_finite()
                    && camera_depth.is_finite()
                    && camera_depth > 0.0)
                    .then_some(RasterVertex {
                        position,
                        camera_depth,
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
                color: triangle_color,
                min_y,
                max_y,
            });
        }
    }
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
    for triangle in workspace.prepared.iter().copied() {
        let first_band = triangle.min_y / rows_per_worker;
        let last_band = (triangle.max_y / rows_per_worker).min(workers - 1);
        for band in &mut workspace.bands[first_band..=last_band] {
            band.push(triangle);
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
                        triangle.color,
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
    triangle_color: Color32,
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
                    linear_depth[pixel] = depth;
                    color[pixel] = triangle_color;
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
    triangles: &[[[f32; 3]; 3]],
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
                        [triangle[a], triangle[b]],
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
        let triangles = [
            [[-2.0, 0.0, -2.0], [2.0, 0.0, -2.0], [0.0, 0.0, 2.0]],
            [[-2.0, 5.0, -2.0], [2.0, 5.0, -2.0], [0.0, 5.0, 2.0]],
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
            &[triangles[1], triangles[0]],
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
}
