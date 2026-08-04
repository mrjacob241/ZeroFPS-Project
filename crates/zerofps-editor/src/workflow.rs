//! General-purpose game orchestration graph.
//!
//! Workflow deliberately contains no racing vocabulary. A checkpoint race is
//! assembled from events, variables, collections, comparisons and proto
//! spawning, and the same nodes can describe doors, missions or score rules.

use std::collections::BTreeMap;

use eframe::egui::{self, Color32, Id, Rect, Sense, Stroke, Vec2};
use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ValueType {
    Execution,
    Event,
    Bool,
    Integer,
    Float,
    String,
    Vector3,
    Transform,
    Object,
    Entity,
    Proto,
    Path,
    Collection,
    Map,
    Variant,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Value {
    None,
    Bool(bool),
    Integer(i64),
    Float(f64),
    String(String),
    Vector3([f64; 3]),
    Entity(u64),
    Collection(Vec<Value>),
    Map(BTreeMap<String, Value>),
}

impl Default for Value {
    fn default() -> Self {
        Self::None
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum NodeKind {
    LifecycleEvent,
    Event,
    Variable,
    Constant,
    Compare,
    Algebra,
    Branch,
    Sequence,
    Gate,
    Timer,
    Collection,
    CollectionGet,
    ForEach,
    SortCollection,
    SceneQuery,
    EntityQuery,
    ObjectProperty,
    TriggerEvent,
    ProtoReference,
    SpawnProto,
    DespawnEntity,
    Map,
    DebugProbe,
    ExposedParameter,
    GetGraphParameter,
    SetGraphParameter,
    BindParameter,
    ObjectInput,
    InputAxis,
    PathReference,
    ClosestPathPoint,
    SamplePath,
    VectorMath,
    Select,
    PidController,
    Raycast,
    ForceOutput,
    TorqueOutput,
    JointControl,
}

impl NodeKind {
    pub const ALL: &'static [Self] = &[
        Self::LifecycleEvent,
        Self::Event,
        Self::Variable,
        Self::Constant,
        Self::Compare,
        Self::Algebra,
        Self::Branch,
        Self::Sequence,
        Self::Gate,
        Self::Timer,
        Self::Collection,
        Self::CollectionGet,
        Self::ForEach,
        Self::SortCollection,
        Self::SceneQuery,
        Self::EntityQuery,
        Self::ObjectProperty,
        Self::TriggerEvent,
        Self::ProtoReference,
        Self::SpawnProto,
        Self::DespawnEntity,
        Self::Map,
        Self::DebugProbe,
        Self::ExposedParameter,
        Self::GetGraphParameter,
        Self::SetGraphParameter,
        Self::BindParameter,
        Self::ObjectInput,
        Self::InputAxis,
        Self::PathReference,
        Self::ClosestPathPoint,
        Self::SamplePath,
        Self::VectorMath,
        Self::Select,
        Self::PidController,
        Self::Raycast,
        Self::ForceOutput,
        Self::TorqueOutput,
        Self::JointControl,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::LifecycleEvent => "Lifecycle Event",
            Self::Event => "Event",
            Self::Variable => "Variable",
            Self::Constant => "Constant",
            Self::Compare => "Compare",
            Self::Algebra => "Algebra",
            Self::Branch => "Branch",
            Self::Sequence => "Sequence",
            Self::Gate => "Gate",
            Self::Timer => "Timer",
            Self::Collection => "Collection",
            Self::CollectionGet => "Collection Get",
            Self::ForEach => "For Each",
            Self::SortCollection => "Sort Collection",
            Self::SceneQuery => "Scene Query",
            Self::EntityQuery => "Entity Query",
            Self::ObjectProperty => "Object Property",
            Self::TriggerEvent => "Trigger Event",
            Self::ProtoReference => "Proto Reference",
            Self::SpawnProto => "Spawn Proto",
            Self::DespawnEntity => "Despawn Entity",
            Self::Map => "Map",
            Self::DebugProbe => "Debug Probe",
            Self::ExposedParameter => "Exposed Parameter",
            Self::GetGraphParameter => "Get Graph Parameter",
            Self::SetGraphParameter => "Set Graph Parameter",
            Self::BindParameter => "Bind Parameter",
            Self::ObjectInput => "Object Input",
            Self::InputAxis => "Input Axis",
            Self::PathReference => "Path Reference",
            Self::ClosestPathPoint => "Closest Path Point",
            Self::SamplePath => "Sample Path",
            Self::VectorMath => "Vector Math",
            Self::Select => "Select",
            Self::PidController => "PID Controller",
            Self::Raycast => "Raycast",
            Self::ForceOutput => "Force Output",
            Self::TorqueOutput => "Torque Output",
            Self::JointControl => "Joint Control",
        }
    }

    pub fn category(self) -> &'static str {
        match self {
            Self::LifecycleEvent
            | Self::Event
            | Self::Branch
            | Self::Sequence
            | Self::Gate
            | Self::Timer
            | Self::ForEach => "Flow",
            Self::Variable
            | Self::Constant
            | Self::Compare
            | Self::Algebra
            | Self::Collection
            | Self::CollectionGet
            | Self::SortCollection
            | Self::Map
            | Self::Select => "Data",
            Self::SceneQuery
            | Self::EntityQuery
            | Self::ObjectProperty
            | Self::TriggerEvent
            | Self::ProtoReference
            | Self::SpawnProto
            | Self::DespawnEntity => "Scene & Entities",
            Self::ExposedParameter
            | Self::GetGraphParameter
            | Self::SetGraphParameter
            | Self::BindParameter
            | Self::ObjectInput => "Parameters",
            Self::InputAxis
            | Self::PathReference
            | Self::ClosestPathPoint
            | Self::SamplePath
            | Self::VectorMath
            | Self::PidController
            | Self::Raycast
            | Self::ForceOutput
            | Self::TorqueOutput
            | Self::JointControl => "Object",
            Self::DebugProbe => "Debug",
        }
    }

    pub fn ports(
        self,
    ) -> (
        &'static [(&'static str, ValueType)],
        &'static [(&'static str, ValueType)],
    ) {
        use ValueType::*;
        match self {
            Self::LifecycleEvent => (
                &[],
                &[
                    ("Started", Event),
                    ("Delta Time", Float),
                    ("Game Time", Float),
                ],
            ),
            Self::Event => (
                &[("Emit", Execution), ("Payload", Variant)],
                &[("Received", Event), ("Payload", Variant)],
            ),
            Self::Variable => (
                &[
                    ("Context", Entity),
                    ("Set", Variant),
                    ("Add", Variant),
                    ("Reset", Execution),
                ],
                &[("Value", Variant), ("Changed", Event)],
            ),
            Self::Constant => (&[], &[("Value", Variant)]),
            Self::Compare => (&[("A", Variant), ("B", Variant)], &[("Result", Bool)]),
            Self::Algebra => (
                &[("X", Variant), ("Y", Variant), ("Z", Variant)],
                &[("Value", Variant)],
            ),
            Self::Branch => (
                &[("In", Execution), ("Condition", Bool)],
                &[("True", Execution), ("False", Execution)],
            ),
            Self::Sequence => (
                &[("In", Execution)],
                &[
                    ("First", Execution),
                    ("Then", Execution),
                    ("Finally", Execution),
                ],
            ),
            Self::Gate => (
                &[("In", Execution), ("Open", Execution), ("Close", Execution)],
                &[("Out", Execution)],
            ),
            Self::Timer => (
                &[
                    ("Start", Execution),
                    ("Stop", Execution),
                    ("Duration", Float),
                ],
                &[("Tick", Event), ("Completed", Event), ("Elapsed", Float)],
            ),
            Self::Collection => (
                &[("Add", Variant), ("Remove", Variant), ("Clear", Execution)],
                &[
                    ("Items", Collection),
                    ("Count", Integer),
                    ("Changed", Event),
                ],
            ),
            Self::CollectionGet => (
                &[("Items", Collection), ("Index", Integer)],
                &[("Item", Variant), ("Valid", Bool)],
            ),
            Self::ForEach => (
                &[("In", Execution), ("Items", Collection)],
                &[
                    ("Loop", Execution),
                    ("Item", Variant),
                    ("Index", Integer),
                    ("Completed", Execution),
                ],
            ),
            Self::SortCollection => (
                &[("Items", Collection), ("Key", Variant)],
                &[("Sorted", Collection)],
            ),
            Self::SceneQuery => (
                &[("Parent", Object)],
                &[("Objects", Collection), ("Count", Integer)],
            ),
            Self::EntityQuery => (&[], &[("Entities", Collection), ("Count", Integer)]),
            Self::ObjectProperty => (
                &[("Target", Variant), ("Set", Variant)],
                &[("Value", Variant), ("Changed", Event), ("Valid", Bool)],
            ),
            Self::TriggerEvent => (
                &[("Triggers", Collection)],
                &[
                    ("Entered", Event),
                    ("Exited", Event),
                    ("Entity", Entity),
                    ("Trigger", Object),
                ],
            ),
            Self::ProtoReference => (&[], &[("Proto", Proto)]),
            Self::SpawnProto => (
                &[
                    ("Spawn", Execution),
                    ("Proto", Proto),
                    ("Transform", Transform),
                    ("Parameters", Map),
                ],
                &[("Spawned", Event), ("Entity", Entity), ("Failed", Event)],
            ),
            Self::DespawnEntity => (
                &[("In", Execution), ("Entity", Entity)],
                &[("Despawned", Event), ("Failed", Event)],
            ),
            Self::Map => (
                &[("Key", String), ("Set", Variant), ("Remove", Execution)],
                &[
                    ("Value", Variant),
                    ("Keys", Collection),
                    ("Values", Collection),
                ],
            ),
            Self::DebugProbe => (&[("Value", Variant)], &[]),
            Self::ExposedParameter | Self::ObjectInput => (
                &[("Default", Variant)],
                &[("Value", Variant), ("Changed", Event)],
            ),
            Self::GetGraphParameter => (
                &[("Entity", Entity), ("Parameter", String)],
                &[("Value", Variant), ("Valid", Bool)],
            ),
            Self::SetGraphParameter => (
                &[
                    ("In", Execution),
                    ("Entity", Entity),
                    ("Parameter", String),
                    ("Value", Variant),
                ],
                &[("Changed", Event), ("Failed", Event)],
            ),
            Self::BindParameter => (
                &[
                    ("Source", Variant),
                    ("Entity", Entity),
                    ("Parameter", String),
                ],
                &[("Changed", Event)],
            ),
            Self::InputAxis => (&[], &[("Value", Float)]),
            Self::PathReference => (&[("Object", Object)], &[("Path", Path)]),
            Self::ClosestPathPoint => (
                &[("Path", Path), ("Position", Vector3)],
                &[
                    ("Coordinate", Float),
                    ("Point", Vector3),
                    ("Tangent", Vector3),
                    ("Distance", Float),
                ],
            ),
            Self::SamplePath => (
                &[("Path", Path), ("Coordinate", Float), ("Offset", Float)],
                &[
                    ("Position", Vector3),
                    ("Tangent", Vector3),
                    ("Curvature", Float),
                    ("Valid", Bool),
                ],
            ),
            Self::VectorMath => (
                &[("A", Vector3), ("B", Vector3)],
                &[("Vector", Vector3), ("Scalar", Float)],
            ),
            Self::Select => (
                &[("Condition", Bool), ("A", Variant), ("B", Variant)],
                &[("Value", Variant)],
            ),
            Self::PidController => (
                &[("Target", Float), ("Current", Float), ("Delta Time", Float)],
                &[("Control", Float)],
            ),
            Self::Raycast => (
                &[
                    ("Origin", Vector3),
                    ("Direction", Vector3),
                    ("Distance", Float),
                ],
                &[
                    ("Hit", Bool),
                    ("Object", Object),
                    ("Position", Vector3),
                    ("Normal", Vector3),
                ],
            ),
            Self::ForceOutput | Self::TorqueOutput => (&[("Value", Vector3)], &[]),
            Self::JointControl => (
                &[
                    ("Joint", Object),
                    ("Velocity", Float),
                    ("Angle", Float),
                    ("Torque", Float),
                    ("Brake", Float),
                ],
                &[],
            ),
        }
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Node {
    pub id: u64,
    pub kind: NodeKind,
    pub position: [f32; 2],
    pub name: String,
    pub settings: BTreeMap<String, Value>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct Link {
    pub from: u64,
    pub output: usize,
    pub to: u64,
    pub input: usize,
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct Graph {
    pub nodes: Vec<Node>,
    pub links: Vec<Link>,
    pub next_id: u64,
}

impl Graph {
    pub fn add(&mut self, kind: NodeKind, position: [f32; 2]) -> u64 {
        let id = self.next_id;
        self.next_id += 1;
        self.nodes.push(Node {
            id,
            kind,
            position,
            name: kind.label().into(),
            settings: default_settings(kind),
        });
        id
    }
    pub fn remove(&mut self, id: u64) {
        self.nodes.retain(|n| n.id != id);
        self.links.retain(|l| l.from != id && l.to != id);
    }
    pub fn connect(
        &mut self,
        from: u64,
        output: usize,
        to: u64,
        input: usize,
    ) -> Result<(), String> {
        let a = self
            .nodes
            .iter()
            .find(|n| n.id == from)
            .ok_or("source missing")?;
        let b = self
            .nodes
            .iter()
            .find(|n| n.id == to)
            .ok_or("target missing")?;
        let out_ty = a.kind.ports().1.get(output).ok_or("output missing")?.1;
        let in_ty = b.kind.ports().0.get(input).ok_or("input missing")?.1;
        let control = |ty| matches!(ty, ValueType::Execution | ValueType::Event);
        let both_control = control(out_ty) && control(in_ty);
        if out_ty != in_ty
            && !both_control
            && (control(out_ty)
                || control(in_ty)
                || (out_ty != ValueType::Variant && in_ty != ValueType::Variant))
        {
            return Err("incompatible socket types".into());
        }
        self.links.retain(|l| l.to != to || l.input != input);
        self.links.push(Link {
            from,
            output,
            to,
            input,
        });
        Ok(())
    }
}

fn default_settings(kind: NodeKind) -> BTreeMap<String, Value> {
    let mut s = BTreeMap::new();
    match kind {
        NodeKind::Variable | NodeKind::ExposedParameter | NodeKind::ObjectInput => {
            s.insert("name".into(), Value::String("value".into()));
            s.insert("default".into(), Value::Float(0.0));
        }
        NodeKind::Algebra => {
            s.insert("expression".into(), Value::String("x".into()));
        }
        NodeKind::Compare => {
            s.insert("operation".into(), Value::String("equal".into()));
        }
        NodeKind::LifecycleEvent => {
            s.insert("event".into(), Value::String("game_started".into()));
        }
        NodeKind::Event => {
            s.insert("name".into(), Value::String("event".into()));
        }
        NodeKind::ProtoReference => {
            s.insert("proto".into(), Value::String(String::new()));
        }
        NodeKind::SceneQuery | NodeKind::EntityQuery => {
            s.insert("tag".into(), Value::String(String::new()));
        }
        NodeKind::ObjectProperty => {
            s.insert("path".into(), Value::String("transform.position".into()));
        }
        NodeKind::InputAxis => {
            s.insert("negative".into(), Value::String("A".into()));
            s.insert("positive".into(), Value::String("D".into()));
        }
        NodeKind::PidController => {
            s.insert("p".into(), Value::Float(1.0));
            s.insert("i".into(), Value::Float(0.0));
            s.insert("d".into(), Value::Float(0.0));
        }
        _ => {}
    }
    s
}

#[derive(Clone, Debug, PartialEq)]
pub enum Command {
    Spawn {
        proto: String,
        transform: Value,
        parameters: BTreeMap<String, Value>,
    },
    Despawn(u64),
    SetParameter {
        entity: u64,
        name: String,
        value: Value,
    },
    Force([f64; 3]),
    Torque([f64; 3]),
}

/// Small deterministic workflow interpreter shared by editor preview and the
/// future standalone player. It intentionally produces commands instead of
/// mutating editor state, keeping orchestration independent from the renderer.
#[derive(Default)]
pub struct Runtime {
    started: bool,
    pub game_time: f64,
    pub variables: BTreeMap<(String, Option<u64>), Value>,
    pub parameters: BTreeMap<(u64, String), Value>,
}

impl Runtime {
    pub fn reset(&mut self) {
        *self = Self::default();
    }

    pub fn tick(&mut self, graph: &Graph, dt: f64) -> Vec<Command> {
        self.game_time += dt.max(0.0);
        let mut commands = Vec::new();
        if !self.started {
            self.started = true;
            for node in graph.nodes.iter().filter(|node| {
                node.kind == NodeKind::LifecycleEvent
                    && setting_string(node, "event") == "game_started"
            }) {
                self.follow(graph, node.id, 0, &mut commands, &mut Vec::new());
            }
        }
        commands
    }

    fn follow(
        &mut self,
        graph: &Graph,
        from: u64,
        output: usize,
        commands: &mut Vec<Command>,
        stack: &mut Vec<u64>,
    ) {
        let targets = graph
            .links
            .iter()
            .filter(|link| link.from == from && link.output == output)
            .map(|link| link.to)
            .collect::<Vec<_>>();
        for target in targets {
            self.execute(graph, target, commands, stack);
        }
    }

    fn execute(
        &mut self,
        graph: &Graph,
        id: u64,
        commands: &mut Vec<Command>,
        stack: &mut Vec<u64>,
    ) {
        if stack.contains(&id) || stack.len() > graph.nodes.len() {
            return;
        }
        let Some(node) = graph.nodes.iter().find(|node| node.id == id) else {
            return;
        };
        stack.push(id);
        match node.kind {
            NodeKind::SpawnProto => {
                let proto = self
                    .input(graph, id, 1, stack)
                    .and_then(value_string)
                    .unwrap_or_else(|| setting_string(node, "proto"));
                if !proto.is_empty() {
                    commands.push(Command::Spawn {
                        proto,
                        transform: self.input(graph, id, 2, stack).unwrap_or(Value::None),
                        parameters: self
                            .input(graph, id, 3, stack)
                            .and_then(value_map)
                            .unwrap_or_default(),
                    });
                    self.follow(graph, id, 0, commands, stack);
                } else {
                    self.follow(graph, id, 2, commands, stack);
                }
            }
            NodeKind::DespawnEntity => {
                if let Some(Value::Entity(entity)) = self.input(graph, id, 1, stack) {
                    commands.push(Command::Despawn(entity));
                    self.follow(graph, id, 0, commands, stack);
                } else {
                    self.follow(graph, id, 1, commands, stack);
                }
            }
            NodeKind::SetGraphParameter => {
                let entity = match self.input(graph, id, 1, stack) {
                    Some(Value::Entity(value)) => value,
                    _ => 0,
                };
                let name = self
                    .input(graph, id, 2, stack)
                    .and_then(value_string)
                    .unwrap_or_else(|| setting_string(node, "parameter"));
                let value = self.input(graph, id, 3, stack).unwrap_or(Value::None);
                self.parameters
                    .insert((entity, name.clone()), value.clone());
                commands.push(Command::SetParameter {
                    entity,
                    name,
                    value,
                });
                self.follow(graph, id, 0, commands, stack);
            }
            NodeKind::Branch => {
                let condition = matches!(self.input(graph, id, 1, stack), Some(Value::Bool(true)));
                self.follow(graph, id, usize::from(!condition), commands, stack);
            }
            NodeKind::Sequence => {
                for output in 0..3 {
                    self.follow(graph, id, output, commands, stack);
                }
            }
            NodeKind::ForceOutput | NodeKind::TorqueOutput => {
                if let Some(Value::Vector3(value)) = self.input(graph, id, 0, stack) {
                    commands.push(if node.kind == NodeKind::ForceOutput {
                        Command::Force(value)
                    } else {
                        Command::Torque(value)
                    });
                }
            }
            _ => self.follow(graph, id, 0, commands, stack),
        }
        stack.pop();
    }

    fn input(&self, graph: &Graph, node: u64, input: usize, stack: &mut Vec<u64>) -> Option<Value> {
        let link = graph
            .links
            .iter()
            .find(|link| link.to == node && link.input == input)?;
        self.output(graph, link.from, link.output, stack)
    }

    fn output(&self, graph: &Graph, id: u64, output: usize, stack: &mut Vec<u64>) -> Option<Value> {
        if stack.contains(&id) {
            return None;
        }
        let node = graph.nodes.iter().find(|node| node.id == id)?;
        stack.push(id);
        let value = match node.kind {
            NodeKind::ProtoReference => Some(Value::String(setting_string(node, "proto"))),
            NodeKind::Constant => node.settings.get("value").cloned(),
            NodeKind::Variable => {
                let name = setting_string(node, "name");
                self.variables
                    .get(&(name, None))
                    .cloned()
                    .or_else(|| node.settings.get("default").cloned())
            }
            NodeKind::Compare => {
                let a = self.input(graph, id, 0, stack).unwrap_or(Value::None);
                let b = self.input(graph, id, 1, stack).unwrap_or(Value::None);
                Some(Value::Bool(compare_values(
                    &a,
                    &b,
                    &setting_string(node, "operation"),
                )))
            }
            NodeKind::Select => {
                let selected = if matches!(self.input(graph, id, 0, stack), Some(Value::Bool(true)))
                {
                    1
                } else {
                    2
                };
                self.input(graph, id, selected, stack)
            }
            NodeKind::LifecycleEvent if output == 1 => Some(Value::Float(0.0)),
            NodeKind::LifecycleEvent if output == 2 => Some(Value::Float(self.game_time)),
            NodeKind::GetGraphParameter => {
                let entity = match self.input(graph, id, 0, stack) {
                    Some(Value::Entity(value)) => value,
                    _ => 0,
                };
                let name = self
                    .input(graph, id, 1, stack)
                    .and_then(value_string)
                    .unwrap_or_else(|| setting_string(node, "parameter"));
                if output == 1 {
                    Some(Value::Bool(self.parameters.contains_key(&(entity, name))))
                } else {
                    self.parameters.get(&(entity, name)).cloned()
                }
            }
            _ => None,
        };
        stack.pop();
        value
    }
}

fn setting_string(node: &Node, key: &str) -> String {
    node.settings
        .get(key)
        .and_then(|value| match value {
            Value::String(value) => Some(value.clone()),
            _ => None,
        })
        .unwrap_or_default()
}

fn value_string(value: Value) -> Option<String> {
    match value {
        Value::String(value) => Some(value),
        _ => None,
    }
}

fn value_map(value: Value) -> Option<BTreeMap<String, Value>> {
    match value {
        Value::Map(value) => Some(value),
        _ => None,
    }
}

fn compare_values(a: &Value, b: &Value, operation: &str) -> bool {
    let numbers = match (a, b) {
        (Value::Float(a), Value::Float(b)) => Some((*a, *b)),
        (Value::Integer(a), Value::Integer(b)) => Some((*a as f64, *b as f64)),
        _ => None,
    };
    match operation {
        "not_equal" => a != b,
        "less" => numbers.is_some_and(|(a, b)| a < b),
        "less_equal" => numbers.is_some_and(|(a, b)| a <= b),
        "greater" => numbers.is_some_and(|(a, b)| a > b),
        "greater_equal" => numbers.is_some_and(|(a, b)| a >= b),
        _ => a == b,
    }
}

#[derive(Default)]
pub struct Editor {
    pub graph: Graph,
    pan: Vec2,
    selected: Option<u64>,
    dragging: Option<(u64, Vec2)>,
    pending: Option<(u64, usize)>,
}

impl Editor {
    pub fn from_graph(graph: Graph) -> Self {
        Self {
            graph,
            ..Default::default()
        }
    }
    pub fn ui(&mut self, ctx: &egui::Context, dirty: &mut bool) {
        egui::SidePanel::right("workflow-inspector")
            .default_width(280.0)
            .resizable(true)
            .show(ctx, |ui| {
                ui.heading("Node");
                let Some(id) = self.selected else {
                    ui.label("Select a node to edit its parameters.");
                    return;
                };
                let Some(node) = self.graph.nodes.iter_mut().find(|node| node.id == id) else {
                    return;
                };
                *dirty |= ui.text_edit_singleline(&mut node.name).changed();
                ui.small(format!("{} · {}", node.kind.category(), node.kind.label()));
                ui.separator();
                for (key, value) in &mut node.settings {
                    ui.horizontal(|ui| {
                        ui.label(key);
                        *dirty |= value_editor(ui, value);
                    });
                }
                if node.settings.is_empty() {
                    ui.weak("This node is configured entirely through its input ports.");
                }
                ui.separator();
                ui.small("Inputs may override these values. Unconnected inputs use the values shown here.");
            });
        egui::CentralPanel::default().show(ctx, |ui| {
            ui.horizontal(|ui| {
                ui.menu_button("Add Node", |ui| {
                    for category in [
                        "Flow",
                        "Data",
                        "Scene & Entities",
                        "Parameters",
                        "Object",
                        "Debug",
                    ] {
                        ui.menu_button(category, |ui| {
                            for &kind in NodeKind::ALL.iter().filter(|k| k.category() == category) {
                                if ui.button(kind.label()).clicked() {
                                    self.graph
                                        .add(kind, [200.0 - self.pan.x, 150.0 - self.pan.y]);
                                    *dirty = true;
                                    ui.close_menu();
                                }
                            }
                        });
                    }
                });
                ui.label("General-purpose orchestration · click output then input to connect");
            });
            let canvas = ui.available_rect_before_wrap();
            let canvas_response =
                ui.interact(canvas, Id::new("workflow-canvas"), Sense::click_and_drag());
            if canvas_response.dragged_by(egui::PointerButton::Middle) {
                self.pan += canvas_response.drag_delta();
            }
            let painter = ui.painter_at(canvas);
            let origin = canvas.min + self.pan;
            for link in &self.graph.links {
                if let (Some(a), Some(b)) = (
                    self.graph.nodes.iter().find(|n| n.id == link.from),
                    self.graph.nodes.iter().find(|n| n.id == link.to),
                ) {
                    painter.line_segment(
                        [
                            origin
                                + Vec2::from(a.position)
                                + Vec2::new(210.0, 42.0 + link.output as f32 * 20.0),
                            origin
                                + Vec2::from(b.position)
                                + Vec2::new(0.0, 42.0 + link.input as f32 * 20.0),
                        ],
                        Stroke::new(2.0, Color32::from_rgb(90, 170, 240)),
                    );
                }
            }
            let ids = self.graph.nodes.iter().map(|n| n.id).collect::<Vec<_>>();
            for id in ids {
                let Some(index) = self.graph.nodes.iter().position(|n| n.id == id) else {
                    continue;
                };
                let node = self.graph.nodes[index].clone();
                let (inputs, outputs) = node.kind.ports();
                let height = 70.0
                    + inputs.len().max(outputs.len()) as f32 * 20.0
                    + node.settings.len() as f32 * 24.0;
                let rect = Rect::from_min_size(
                    origin + Vec2::from(node.position),
                    Vec2::new(210.0, height),
                );
                painter.rect_filled(
                    rect,
                    6.0,
                    if self.selected == Some(id) {
                        Color32::from_rgb(48, 55, 70)
                    } else {
                        Color32::from_rgb(35, 39, 49)
                    },
                );
                painter.rect_stroke(
                    rect,
                    6.0,
                    Stroke::new(1.0, Color32::from_rgb(85, 95, 115)),
                    egui::StrokeKind::Outside,
                );
                let response = ui.interact(
                    rect,
                    Id::new(("workflow-node", id)),
                    Sense::click_and_drag(),
                );
                if response.clicked() {
                    self.selected = Some(id);
                }
                if response.drag_started() {
                    self.dragging = response.interact_pointer_pos().map(|p| (id, p - rect.min));
                }
                if response.dragged()
                    && let (Some((drag, offset)), Some(p)) =
                        (self.dragging, response.interact_pointer_pos())
                    && drag == id
                {
                    self.graph.nodes[index].position = ((p - offset) - origin).into();
                    *dirty = true;
                }
                if response.drag_stopped() {
                    self.dragging = None;
                }
                painter.text(
                    rect.min + Vec2::new(10.0, 10.0),
                    egui::Align2::LEFT_TOP,
                    &node.name,
                    egui::FontId::proportional(14.0),
                    Color32::WHITE,
                );
                for (pin, (name, _)) in inputs.iter().enumerate() {
                    let p = rect.min + Vec2::new(0.0, 42.0 + pin as f32 * 20.0);
                    painter.circle_filled(p, 5.0, Color32::from_rgb(240, 180, 80));
                    painter.text(
                        p + Vec2::new(9.0, 0.0),
                        egui::Align2::LEFT_CENTER,
                        *name,
                        egui::FontId::proportional(11.0),
                        Color32::LIGHT_GRAY,
                    );
                    if ui
                        .interact(
                            Rect::from_center_size(p, Vec2::splat(16.0)),
                            Id::new(("wi", id, pin)),
                            Sense::click(),
                        )
                        .clicked()
                    {
                        if let Some((from, out)) = self.pending.take() {
                            if self.graph.connect(from, out, id, pin).is_ok() {
                                *dirty = true;
                            }
                        } else {
                            let previous = self.graph.links.len();
                            self.graph
                                .links
                                .retain(|link| link.to != id || link.input != pin);
                            if self.graph.links.len() != previous {
                                *dirty = true;
                            }
                        }
                    }
                }
                for (pin, (name, _)) in outputs.iter().enumerate() {
                    let p = rect.min + Vec2::new(rect.width(), 42.0 + pin as f32 * 20.0);
                    painter.circle_filled(p, 5.0, Color32::from_rgb(90, 180, 240));
                    painter.text(
                        p - Vec2::new(9.0, 0.0),
                        egui::Align2::RIGHT_CENTER,
                        *name,
                        egui::FontId::proportional(11.0),
                        Color32::LIGHT_GRAY,
                    );
                    if ui
                        .interact(
                            Rect::from_center_size(p, Vec2::splat(16.0)),
                            Id::new(("wo", id, pin)),
                            Sense::click(),
                        )
                        .clicked()
                    {
                        self.pending = Some((id, pin));
                    }
                }
            }
            if ctx.input(|i| i.key_pressed(egui::Key::Delete))
                && let Some(id) = self.selected.take()
            {
                self.graph.remove(id);
                *dirty = true;
            }
        });
    }
}

fn value_editor(ui: &mut egui::Ui, value: &mut Value) -> bool {
    match value {
        Value::None => {
            ui.weak("connected");
            false
        }
        Value::Bool(value) => ui.checkbox(value, "").changed(),
        Value::Integer(value) => ui.add(egui::DragValue::new(value)).changed(),
        Value::Float(value) => ui.add(egui::DragValue::new(value).speed(0.05)).changed(),
        Value::String(value) => ui.text_edit_singleline(value).changed(),
        Value::Vector3(value) => {
            let mut changed = false;
            for component in value {
                changed |= ui
                    .add(egui::DragValue::new(component).speed(0.05).max_decimals(4))
                    .changed();
            }
            changed
        }
        Value::Entity(value) => ui.add(egui::DragValue::new(value)).changed(),
        Value::Collection(values) => {
            ui.label(format!("{} items", values.len()));
            false
        }
        Value::Map(values) => {
            ui.label(format!("{} fields", values.len()));
            false
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn rejects_incompatible_links() {
        let mut g = Graph::default();
        let a = g.add(NodeKind::Constant, [0.0; 2]);
        let b = g.add(NodeKind::Branch, [0.0; 2]);
        assert!(g.connect(a, 0, b, 0).is_err());
        assert!(g.connect(a, 0, b, 1).is_ok());
    }
    #[test]
    fn graph_round_trips() {
        let mut g = Graph::default();
        g.add(NodeKind::SpawnProto, [3.0, 4.0]);
        let s = serde_json::to_string(&g).unwrap();
        assert_eq!(serde_json::from_str::<Graph>(&s).unwrap().nodes.len(), 1);
    }

    #[test]
    fn game_started_spawns_connected_proto_once() {
        let mut graph = Graph::default();
        let started = graph.add(NodeKind::LifecycleEvent, [0.0; 2]);
        let proto = graph.add(NodeKind::ProtoReference, [0.0; 2]);
        let spawn = graph.add(NodeKind::SpawnProto, [0.0; 2]);
        graph
            .nodes
            .iter_mut()
            .find(|node| node.id == proto)
            .unwrap()
            .settings
            .insert("proto".into(), Value::String("Car-Proto".into()));
        graph.connect(started, 0, spawn, 0).unwrap();
        graph.connect(proto, 0, spawn, 1).unwrap();

        let mut runtime = Runtime::default();
        let commands = runtime.tick(&graph, 1.0 / 60.0);
        assert!(matches!(
            commands.as_slice(),
            [Command::Spawn { proto, .. }] if proto == "Car-Proto"
        ));
        assert!(runtime.tick(&graph, 1.0 / 60.0).is_empty());
    }
}
