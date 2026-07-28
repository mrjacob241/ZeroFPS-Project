//! Shared authoring/runtime scene model for ZeroFPS.
//!
//! The editor and the compiled player deliberately use this exact data model.
//! Nodes form a stable, generation-checked geometry tree; capabilities are
//! components, not subclasses.

use serde::{Deserialize, Serialize};
use std::collections::{BTreeMap, BTreeSet};
use thiserror::Error;

const EPSILON: f32 = 1.0e-6;

#[derive(Clone, Copy, Debug, Default, PartialEq, Serialize, Deserialize)]
pub struct Vec3 {
    pub x: f32,
    pub y: f32,
    pub z: f32,
}

impl Vec3 {
    pub const ZERO: Self = Self::new(0.0, 0.0, 0.0);
    pub const ONE: Self = Self::new(1.0, 1.0, 1.0);
    pub const X: Self = Self::new(1.0, 0.0, 0.0);
    pub const Y: Self = Self::new(0.0, 1.0, 0.0);
    pub const Z: Self = Self::new(0.0, 0.0, 1.0);

    pub const fn new(x: f32, y: f32, z: f32) -> Self {
        Self { x, y, z }
    }
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z
    }
    pub fn cross(self, rhs: Self) -> Self {
        Self::new(
            self.y * rhs.z - self.z * rhs.y,
            self.z * rhs.x - self.x * rhs.z,
            self.x * rhs.y - self.y * rhs.x,
        )
    }
    pub fn length(self) -> f32 {
        self.dot(self).sqrt()
    }
    pub fn normalized(self) -> Self {
        let n = self.length();
        if n <= EPSILON { Self::ZERO } else { self / n }
    }
    pub fn component_mul(self, rhs: Self) -> Self {
        Self::new(self.x * rhs.x, self.y * rhs.y, self.z * rhs.z)
    }
    pub fn component_div(self, rhs: Self) -> Option<Self> {
        if rhs.x.abs() <= EPSILON || rhs.y.abs() <= EPSILON || rhs.z.abs() <= EPSILON {
            None
        } else {
            Some(Self::new(self.x / rhs.x, self.y / rhs.y, self.z / rhs.z))
        }
    }
    pub fn approx_eq(self, rhs: Self, epsilon: f32) -> bool {
        (self - rhs).length() <= epsilon
    }
}

impl std::ops::Add for Vec3 {
    type Output = Self;
    fn add(self, rhs: Self) -> Self {
        Self::new(self.x + rhs.x, self.y + rhs.y, self.z + rhs.z)
    }
}
impl std::ops::Sub for Vec3 {
    type Output = Self;
    fn sub(self, rhs: Self) -> Self {
        Self::new(self.x - rhs.x, self.y - rhs.y, self.z - rhs.z)
    }
}
impl std::ops::Mul<f32> for Vec3 {
    type Output = Self;
    fn mul(self, rhs: f32) -> Self {
        Self::new(self.x * rhs, self.y * rhs, self.z * rhs)
    }
}
impl std::ops::Div<f32> for Vec3 {
    type Output = Self;
    fn div(self, rhs: f32) -> Self {
        self * (1.0 / rhs)
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Quat {
    pub x: f32,
    pub y: f32,
    pub z: f32,
    pub w: f32,
}

impl Default for Quat {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Quat {
    pub const IDENTITY: Self = Self {
        x: 0.0,
        y: 0.0,
        z: 0.0,
        w: 1.0,
    };
    pub fn from_axis_angle(axis: Vec3, radians: f32) -> Self {
        let axis = axis.normalized();
        let (s, c) = (radians * 0.5).sin_cos();
        Self {
            x: axis.x * s,
            y: axis.y * s,
            z: axis.z * s,
            w: c,
        }
        .normalized()
    }
    /// Builds a rotation by applying local X, then Y, then Z rotations.
    pub fn from_euler_xyz(radians: Vec3) -> Self {
        let x = Self::from_axis_angle(Vec3::X, radians.x);
        let y = Self::from_axis_angle(Vec3::Y, radians.y);
        let z = Self::from_axis_angle(Vec3::Z, radians.z);
        z * y * x
    }
    /// Returns XYZ Euler angles in radians for the equivalent `Z * Y * X`
    /// quaternion composition. At gimbal lock, Y is clamped to ±π/2.
    pub fn to_euler_xyz(self) -> Vec3 {
        let q = self.normalized();
        Vec3::new(
            (2.0 * (q.w * q.x + q.y * q.z)).atan2(1.0 - 2.0 * (q.x * q.x + q.y * q.y)),
            (2.0 * (q.w * q.y - q.z * q.x)).clamp(-1.0, 1.0).asin(),
            (2.0 * (q.w * q.z + q.x * q.y)).atan2(1.0 - 2.0 * (q.y * q.y + q.z * q.z)),
        )
    }
    pub fn dot(self, rhs: Self) -> f32 {
        self.x * rhs.x + self.y * rhs.y + self.z * rhs.z + self.w * rhs.w
    }
    pub fn normalized(self) -> Self {
        let n = self.dot(self).sqrt();
        if n <= EPSILON {
            Self::IDENTITY
        } else {
            Self {
                x: self.x / n,
                y: self.y / n,
                z: self.z / n,
                w: self.w / n,
            }
        }
    }
    pub fn inverse(self) -> Self {
        let n = self.dot(self);
        if n <= EPSILON {
            Self::IDENTITY
        } else {
            Self {
                x: -self.x / n,
                y: -self.y / n,
                z: -self.z / n,
                w: self.w / n,
            }
        }
    }
    pub fn rotate(self, value: Vec3) -> Vec3 {
        let q = self.normalized();
        let v = Vec3::new(q.x, q.y, q.z);
        value + v.cross(value) * (2.0 * q.w) + v.cross(v.cross(value)) * 2.0
    }
    pub fn approx_eq(self, rhs: Self, epsilon: f32) -> bool {
        self.dot(rhs).abs() >= 1.0 - epsilon
    }
}

impl std::ops::Mul for Quat {
    type Output = Self;
    fn mul(self, rhs: Self) -> Self {
        Self {
            w: self.w * rhs.w - self.x * rhs.x - self.y * rhs.y - self.z * rhs.z,
            x: self.w * rhs.x + self.x * rhs.w + self.y * rhs.z - self.z * rhs.y,
            y: self.w * rhs.y - self.x * rhs.z + self.y * rhs.w + self.z * rhs.x,
            z: self.w * rhs.z + self.x * rhs.y - self.y * rhs.x + self.z * rhs.w,
        }
        .normalized()
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Serialize, Deserialize)]
pub struct Transform {
    pub translation: Vec3,
    pub rotation: Quat,
    /// User-authored per-axis scale. Rendering must not add further distortion.
    pub scale: Vec3,
}

impl Default for Transform {
    fn default() -> Self {
        Self::IDENTITY
    }
}

impl Transform {
    pub const IDENTITY: Self = Self {
        translation: Vec3::ZERO,
        rotation: Quat::IDENTITY,
        scale: Vec3::ONE,
    };
    /// Compose `local` below `self` (the parent).
    pub fn compose(self, local: Self) -> Self {
        Self {
            translation: self.translation
                + self
                    .rotation
                    .rotate(self.scale.component_mul(local.translation)),
            rotation: self.rotation * local.rotation,
            scale: self.scale.component_mul(local.scale),
        }
    }
    /// Express a world transform relative to this parent.
    pub fn relative_to(self, world: Self) -> Result<Self, SceneError> {
        let inv_rotation = self.rotation.inverse();
        let translation = inv_rotation
            .rotate(world.translation - self.translation)
            .component_div(self.scale)
            .ok_or(SceneError::NonInvertibleTransform)?;
        let scale = world
            .scale
            .component_div(self.scale)
            .ok_or(SceneError::NonInvertibleTransform)?;
        Ok(Self {
            translation,
            rotation: inv_rotation * world.rotation,
            scale,
        })
    }
    pub fn approx_eq(self, rhs: Self, epsilon: f32) -> bool {
        self.translation.approx_eq(rhs.translation, epsilon)
            && self.rotation.approx_eq(rhs.rotation, epsilon)
            && self.scale.approx_eq(rhs.scale, epsilon)
    }
}

/// A slot plus generation. Stale handles never alias newly-created nodes.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash, Serialize, Deserialize)]
pub struct NodeId {
    pub slot: u32,
    pub generation: u32,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Component {
    Model {
        asset: String,
        /// Restrict this instance to one primitive of a hierarchical asset.
        /// `None` preserves the historic behavior of drawing the whole asset.
        #[serde(default)]
        primitive: Option<usize>,
    },
    Camera {
        field_of_view_degrees: f32,
    },
    Light {
        intensity: f32,
        color: [f32; 3],
    },
    Collider {
        shape: String,
        #[serde(default)]
        center: Vec3,
        #[serde(default = "default_collider_half_extents")]
        half_extents: Vec3,
        #[serde(default = "default_collider_radius")]
        radius: f32,
        #[serde(default = "default_collider_height")]
        height: f32,
        #[serde(default = "default_collider_coupling_stiffness")]
        coupling_stiffness: f32,
        #[serde(default = "default_collider_coupling_damping")]
        coupling_damping: f32,
        #[serde(default = "default_collider_elasticity_stiffness")]
        elasticity_stiffness: f32,
        #[serde(default = "default_collider_elasticity_damping")]
        elasticity_damping: f32,
        /// Normal-velocity retention after an impulse contact, in [0, 1].
        #[serde(default)]
        restitution: f32,
        #[serde(default = "default_collider_force_cutoff")]
        force_cutoff: f32,
        #[serde(default = "default_collider_force_cutoff")]
        collision_force_cutoff: f32,
        #[serde(default = "default_collider_density")]
        density: f32,
        #[serde(default = "default_collider_mass")]
        mass: f32,
        #[serde(default = "default_true")]
        automatic_mass: bool,
        #[serde(default = "default_collider_friction")]
        friction: Vec3,
        /// Cylinder-only rotational joint: "none", "wheel", or "engine".
        #[serde(default)]
        joint: String,
    },
    RigidBody {
        dynamic: bool,
    },
    Script {
        source: String,
    },
    AudioSource {
        asset: String,
    },
    Marker {
        kind: String,
    },
}

fn default_collider_half_extents() -> Vec3 {
    Vec3::new(0.5, 0.5, 0.5)
}

fn default_collider_radius() -> f32 {
    0.5
}

fn default_collider_height() -> f32 {
    1.0
}

fn default_collider_coupling_stiffness() -> f32 {
    0.5
}

fn default_collider_coupling_damping() -> f32 {
    1.0
}

fn default_collider_elasticity_stiffness() -> f32 {
    0.5
}

fn default_collider_elasticity_damping() -> f32 {
    1.0
}

fn default_collider_force_cutoff() -> f32 {
    0.01
}

fn default_collider_density() -> f32 {
    1_000.0
}

fn default_collider_mass() -> f32 {
    1.0
}

fn default_collider_friction() -> Vec3 {
    Vec3::new(0.8, 0.8, 0.8)
}

fn default_true() -> bool {
    true
}

#[derive(Clone, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum AttributeKey {
    Visible,
    Enabled,
    Material,
    MaterialShader,
    MaterialSmoothNormals,
    MaterialUseImportedOptics,
    RenderLayer,
    Tint,
    CastShadows,
    CollisionGroup,
    Custom(String),
}

// Attribute keys are used as JSON object keys in GeometryTree. Serde's default
// externally-tagged representation for `Custom(String)` is an object, which
// JSON cannot use as a map key. Keep the historic names for built-in keys and
// give custom keys an explicitly string-valued representation.
impl Serialize for AttributeKey {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        match self {
            Self::Visible => serializer.serialize_str("Visible"),
            Self::Enabled => serializer.serialize_str("Enabled"),
            Self::Material => serializer.serialize_str("Material"),
            Self::MaterialShader => serializer.serialize_str("MaterialShader"),
            Self::MaterialSmoothNormals => serializer.serialize_str("MaterialSmoothNormals"),
            Self::MaterialUseImportedOptics => {
                serializer.serialize_str("MaterialUseImportedOptics")
            }
            Self::RenderLayer => serializer.serialize_str("RenderLayer"),
            Self::Tint => serializer.serialize_str("Tint"),
            Self::CastShadows => serializer.serialize_str("CastShadows"),
            Self::CollisionGroup => serializer.serialize_str("CollisionGroup"),
            Self::Custom(name) => serializer.serialize_str(&format!("Custom:{name}")),
        }
    }
}

impl<'de> Deserialize<'de> for AttributeKey {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        let value = String::deserialize(deserializer)?;
        match value.as_str() {
            "Visible" => Ok(Self::Visible),
            "Enabled" => Ok(Self::Enabled),
            "Material" => Ok(Self::Material),
            "MaterialShader" => Ok(Self::MaterialShader),
            "MaterialSmoothNormals" => Ok(Self::MaterialSmoothNormals),
            "MaterialUseImportedOptics" => Ok(Self::MaterialUseImportedOptics),
            "RenderLayer" => Ok(Self::RenderLayer),
            "Tint" => Ok(Self::Tint),
            "CastShadows" => Ok(Self::CastShadows),
            "CollisionGroup" => Ok(Self::CollisionGroup),
            _ => value
                .strip_prefix("Custom:")
                .map(|name| Self::Custom(name.to_owned()))
                .ok_or_else(|| {
                    <D::Error as serde::de::Error>::custom(format!(
                        "unknown geometry attribute key `{value}`"
                    ))
                }),
        }
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum Attribute {
    Bool(bool),
    Number(f64),
    Text(String),
    Color([f32; 4]),
    Integer(i64),
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub enum AttributeDeclaration {
    /// Consult the closest ancestor, then the project default.
    Inherit,
    Value(Attribute),
    /// Deliberately bypass ancestors and use the project default.
    Initial,
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum AttributeOrigin {
    Node(NodeId),
    Initial(NodeId),
    Default,
}

#[derive(Clone, Debug, PartialEq)]
pub struct ResolvedAttribute {
    pub value: Attribute,
    pub origin: AttributeOrigin,
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct GeometryNode {
    pub name: String,
    parent: Option<NodeId>,
    children: Vec<NodeId>,
    local: Transform,
    global: Transform,
    pub components: Vec<Component>,
    declarations: BTreeMap<AttributeKey, AttributeDeclaration>,
}

impl GeometryNode {
    pub fn parent(&self) -> Option<NodeId> {
        self.parent
    }
    pub fn children(&self) -> &[NodeId] {
        &self.children
    }
    pub fn local_transform(&self) -> Transform {
        self.local
    }
    pub fn global_transform(&self) -> Transform {
        self.global
    }
    pub fn declaration(&self, key: &AttributeKey) -> Option<&AttributeDeclaration> {
        self.declarations.get(key)
    }
}

#[derive(Clone, Debug, Serialize, Deserialize)]
struct Slot {
    generation: u32,
    node: Option<GeometryNode>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct GeometryTree {
    slots: Vec<Slot>,
    free: Vec<u32>,
    roots: Vec<NodeId>,
    defaults: BTreeMap<AttributeKey, Attribute>,
}

impl Default for GeometryTree {
    fn default() -> Self {
        Self::new()
    }
}

impl GeometryTree {
    pub fn new() -> Self {
        let mut defaults = BTreeMap::new();
        defaults.insert(AttributeKey::Visible, Attribute::Bool(true));
        defaults.insert(AttributeKey::Enabled, Attribute::Bool(true));
        defaults.insert(AttributeKey::CastShadows, Attribute::Bool(true));
        defaults.insert(AttributeKey::RenderLayer, Attribute::Integer(0));
        defaults.insert(AttributeKey::Tint, Attribute::Color([1.0; 4]));
        defaults.insert(AttributeKey::Material, Attribute::Text("default".into()));
        defaults.insert(
            AttributeKey::MaterialShader,
            Attribute::Text("diffuse".into()),
        );
        defaults.insert(AttributeKey::MaterialSmoothNormals, Attribute::Bool(true));
        defaults.insert(
            AttributeKey::MaterialUseImportedOptics,
            Attribute::Bool(true),
        );
        defaults.insert(AttributeKey::CollisionGroup, Attribute::Integer(0));
        Self {
            slots: Vec::new(),
            free: Vec::new(),
            roots: Vec::new(),
            defaults,
        }
    }
    pub fn roots(&self) -> &[NodeId] {
        &self.roots
    }
    pub fn contains(&self, id: NodeId) -> bool {
        self.node(id).is_ok()
    }
    pub fn node(&self, id: NodeId) -> Result<&GeometryNode, SceneError> {
        self.slots
            .get(id.slot as usize)
            .filter(|s| s.generation == id.generation)
            .and_then(|s| s.node.as_ref())
            .ok_or(SceneError::InvalidNode(id))
    }
    pub fn node_mut(&mut self, id: NodeId) -> Result<&mut GeometryNode, SceneError> {
        self.slots
            .get_mut(id.slot as usize)
            .filter(|s| s.generation == id.generation)
            .and_then(|s| s.node.as_mut())
            .ok_or(SceneError::InvalidNode(id))
    }
    pub fn iter(&self) -> impl Iterator<Item = (NodeId, &GeometryNode)> {
        self.slots.iter().enumerate().filter_map(|(slot, entry)| {
            entry.node.as_ref().map(|node| {
                (
                    NodeId {
                        slot: slot as u32,
                        generation: entry.generation,
                    },
                    node,
                )
            })
        })
    }
    pub fn create(
        &mut self,
        name: impl Into<String>,
        parent: Option<NodeId>,
    ) -> Result<NodeId, SceneError> {
        if let Some(parent) = parent {
            self.node(parent)?;
        }
        let slot = self.free.pop().unwrap_or(self.slots.len() as u32);
        if slot as usize == self.slots.len() {
            self.slots.push(Slot {
                generation: 0,
                node: None,
            });
        }
        let id = NodeId {
            slot,
            generation: self.slots[slot as usize].generation,
        };
        self.slots[slot as usize].node = Some(GeometryNode {
            name: name.into(),
            parent,
            children: Vec::new(),
            local: Transform::IDENTITY,
            global: Transform::IDENTITY,
            components: Vec::new(),
            declarations: BTreeMap::new(),
        });
        if let Some(parent) = parent {
            self.node_mut(parent)?.children.push(id);
        } else {
            self.roots.push(id);
        }
        self.propagate_from(id)?;
        Ok(id)
    }
    /// Remove a whole subtree and invalidate every handle in it.
    pub fn remove(&mut self, id: NodeId) -> Result<Vec<NodeId>, SceneError> {
        self.node(id)?;
        self.unlink(id)?;
        let mut order = Vec::new();
        self.collect_subtree(id, &mut order)?;
        for victim in order.iter().rev().copied() {
            let slot = &mut self.slots[victim.slot as usize];
            slot.node = None;
            slot.generation = slot.generation.wrapping_add(1);
            self.free.push(victim.slot);
        }
        Ok(order)
    }
    fn collect_subtree(&self, id: NodeId, out: &mut Vec<NodeId>) -> Result<(), SceneError> {
        out.push(id);
        for child in self.node(id)?.children.clone() {
            self.collect_subtree(child, out)?;
        }
        Ok(())
    }
    fn unlink(&mut self, id: NodeId) -> Result<(), SceneError> {
        if let Some(parent) = self.node(id)?.parent {
            self.node_mut(parent)?.children.retain(|child| *child != id);
        } else {
            self.roots.retain(|root| *root != id);
        }
        Ok(())
    }
    pub fn set_local_transform(
        &mut self,
        id: NodeId,
        transform: Transform,
    ) -> Result<(), SceneError> {
        self.node_mut(id)?.local = transform;
        self.propagate_from(id)
    }
    pub fn set_global_transform(
        &mut self,
        id: NodeId,
        transform: Transform,
    ) -> Result<(), SceneError> {
        let local = match self.node(id)?.parent {
            Some(parent) => self.node(parent)?.global.relative_to(transform)?,
            None => transform,
        };
        self.set_local_transform(id, local)
    }
    pub fn propagate_all(&mut self) -> Result<(), SceneError> {
        for root in self.roots.clone() {
            self.propagate_from(root)?;
        }
        Ok(())
    }
    pub fn propagate_from(&mut self, id: NodeId) -> Result<(), SceneError> {
        let global = match self.node(id)?.parent {
            Some(parent) => self.node(parent)?.global.compose(self.node(id)?.local),
            None => self.node(id)?.local,
        };
        let children = self.node(id)?.children.clone();
        self.node_mut(id)?.global = global;
        for child in children {
            self.propagate_from(child)?;
        }
        Ok(())
    }
    pub fn reparent(
        &mut self,
        id: NodeId,
        new_parent: Option<NodeId>,
        mode: ReparentMode,
    ) -> Result<(), SceneError> {
        self.node(id)?;
        if new_parent == Some(id) {
            return Err(SceneError::Cycle);
        }
        if let Some(parent) = new_parent {
            self.node(parent)?;
            let mut cursor = Some(parent);
            while let Some(current) = cursor {
                if current == id {
                    return Err(SceneError::Cycle);
                }
                cursor = self.node(current)?.parent;
            }
        }
        let old_global = self.node(id)?.global;
        // Calculate first so a failed inversion is atomic.
        let new_local = match mode {
            ReparentMode::KeepLocal => self.node(id)?.local,
            ReparentMode::KeepGlobal => match new_parent {
                Some(parent) => self.node(parent)?.global.relative_to(old_global)?,
                None => old_global,
            },
        };
        self.unlink(id)?;
        self.node_mut(id)?.parent = new_parent;
        self.node_mut(id)?.local = new_local;
        if let Some(parent) = new_parent {
            self.node_mut(parent)?.children.push(id);
        } else {
            self.roots.push(id);
        }
        self.propagate_from(id)
    }
    pub fn add_component(&mut self, id: NodeId, component: Component) -> Result<usize, SceneError> {
        let node = self.node_mut(id)?;
        node.components.push(component);
        Ok(node.components.len() - 1)
    }
    pub fn set_default(&mut self, key: AttributeKey, value: Attribute) {
        self.defaults.insert(key, value);
    }
    pub fn set_attribute(
        &mut self,
        id: NodeId,
        key: AttributeKey,
        declaration: AttributeDeclaration,
    ) -> Result<(), SceneError> {
        self.node_mut(id)?.declarations.insert(key, declaration);
        Ok(())
    }
    pub fn clear_attribute(
        &mut self,
        id: NodeId,
        key: &AttributeKey,
    ) -> Result<Option<AttributeDeclaration>, SceneError> {
        Ok(self.node_mut(id)?.declarations.remove(key))
    }
    pub fn resolve_attribute(
        &self,
        id: NodeId,
        key: &AttributeKey,
    ) -> Result<ResolvedAttribute, SceneError> {
        let mut cursor = Some(id);
        while let Some(current) = cursor {
            let node = self.node(current)?;
            match node.declarations.get(key) {
                Some(AttributeDeclaration::Value(value)) => {
                    return Ok(ResolvedAttribute {
                        value: value.clone(),
                        origin: AttributeOrigin::Node(current),
                    });
                }
                Some(AttributeDeclaration::Initial) => {
                    let value = self
                        .defaults
                        .get(key)
                        .cloned()
                        .ok_or_else(|| SceneError::MissingDefault(key.clone()))?;
                    return Ok(ResolvedAttribute {
                        value,
                        origin: AttributeOrigin::Initial(current),
                    });
                }
                Some(AttributeDeclaration::Inherit) | None => cursor = node.parent,
            }
        }
        self.defaults
            .get(key)
            .cloned()
            .map(|value| ResolvedAttribute {
                value,
                origin: AttributeOrigin::Default,
            })
            .ok_or_else(|| SceneError::MissingDefault(key.clone()))
    }
    pub fn apply_commands(
        &mut self,
        commands: impl IntoIterator<Item = SceneCommand>,
    ) -> CommandReport {
        let mut report = CommandReport::default();
        for command in commands {
            let result = self.apply_command(command);
            match result {
                Ok(()) => report.applied += 1,
                Err(error) => report.errors.push(error),
            }
        }
        report
    }
    fn apply_command(&mut self, command: SceneCommand) -> Result<(), SceneError> {
        match command {
            SceneCommand::SetLocal { node, transform } => self.set_local_transform(node, transform),
            SceneCommand::SetGlobal { node, transform } => {
                self.set_global_transform(node, transform)
            }
            SceneCommand::Reparent { node, parent, mode } => self.reparent(node, parent, mode),
            SceneCommand::SetAttribute {
                node,
                key,
                declaration,
            } => self.set_attribute(node, key, declaration),
            SceneCommand::AddComponent { node, component } => {
                self.add_component(node, component).map(|_| ())
            }
            SceneCommand::Remove { node } => self.remove(node).map(|_| ()),
        }
    }
    /// Drops invalid selection handles after object deletion/reload.
    pub fn sanitize_selection(&self, selection: &mut BTreeSet<NodeId>) {
        selection.retain(|id| self.contains(*id));
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum ReparentMode {
    KeepLocal,
    KeepGlobal,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub enum SceneCommand {
    SetLocal {
        node: NodeId,
        transform: Transform,
    },
    SetGlobal {
        node: NodeId,
        transform: Transform,
    },
    Reparent {
        node: NodeId,
        parent: Option<NodeId>,
        mode: ReparentMode,
    },
    SetAttribute {
        node: NodeId,
        key: AttributeKey,
        declaration: AttributeDeclaration,
    },
    AddComponent {
        node: NodeId,
        component: Component,
    },
    Remove {
        node: NodeId,
    },
}

#[derive(Debug, Default)]
pub struct CommandReport {
    pub applied: usize,
    pub errors: Vec<SceneError>,
}

#[derive(Clone, Debug, Error, PartialEq)]
pub enum SceneError {
    #[error("invalid or stale node handle {0:?}")]
    InvalidNode(NodeId),
    #[error("reparenting would create a geometry cycle")]
    Cycle,
    #[error("transform has a zero scale component and cannot be inverted")]
    NonInvertibleTransform,
    #[error("attribute {0:?} has no registered project default")]
    MissingDefault(AttributeKey),
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::f32::consts::{FRAC_PI_2, PI};

    fn t(x: f32, y: f32, z: f32) -> Transform {
        Transform {
            translation: Vec3::new(x, y, z),
            ..Transform::IDENTITY
        }
    }

    #[test]
    fn quaternion_rotates_vector() {
        let q = Quat::from_axis_angle(Vec3::Z, FRAC_PI_2);
        assert!(q.rotate(Vec3::X).approx_eq(Vec3::Y, 1e-5));
    }

    #[test]
    fn euler_xyz_round_trips_through_quaternion() {
        let quaternion = Quat::from_euler_xyz(Vec3::new(0.31, -0.47, 1.12));
        let rebuilt = Quat::from_euler_xyz(quaternion.to_euler_xyz());
        assert!(quaternion.approx_eq(rebuilt, 1e-5));
    }

    #[test]
    fn euler_xyz_applies_local_x_then_y_then_z() {
        let from_euler = Quat::from_euler_xyz(Vec3::new(FRAC_PI_2, FRAC_PI_2, 0.0));
        let composed =
            Quat::from_axis_angle(Vec3::Y, FRAC_PI_2) * Quat::from_axis_angle(Vec3::X, FRAC_PI_2);
        assert!(from_euler.approx_eq(composed, 1e-5));
    }

    #[test]
    fn ragdoll_cascade_moves_and_rotates_hand() {
        let mut tree = GeometryTree::new();
        let torso = tree.create("Torso", None).unwrap();
        let arm = tree.create("Upper arm", Some(torso)).unwrap();
        let hand = tree.create("Hand", Some(arm)).unwrap();
        tree.set_local_transform(arm, t(1.0, 0.0, 0.0)).unwrap();
        tree.set_local_transform(hand, t(1.0, 0.0, 0.0)).unwrap();
        tree.set_local_transform(torso, t(5.0, 0.0, 0.0)).unwrap();
        assert!(
            tree.node(hand)
                .unwrap()
                .global
                .translation
                .approx_eq(Vec3::new(7.0, 0.0, 0.0), 1e-5)
        );
        let arm_pose = Transform {
            translation: Vec3::X,
            rotation: Quat::from_axis_angle(Vec3::Z, FRAC_PI_2),
            scale: Vec3::ONE,
        };
        tree.set_local_transform(arm, arm_pose).unwrap();
        assert!(
            tree.node(hand)
                .unwrap()
                .global
                .translation
                .approx_eq(Vec3::new(6.0, 1.0, 0.0), 1e-5)
        );
    }

    #[test]
    fn stale_ids_do_not_alias_reused_slots() {
        let mut tree = GeometryTree::new();
        let old = tree.create("old", None).unwrap();
        tree.remove(old).unwrap();
        let new = tree.create("new", None).unwrap();
        assert_eq!(old.slot, new.slot);
        assert_ne!(old.generation, new.generation);
        assert_eq!(tree.node(old), Err(SceneError::InvalidNode(old)));
    }

    #[test]
    fn remove_is_recursive_and_selection_is_safe() {
        let mut tree = GeometryTree::new();
        let root = tree.create("root", None).unwrap();
        let child = tree.create("child", Some(root)).unwrap();
        let survivor = tree.create("survivor", None).unwrap();
        tree.remove(root).unwrap();
        let mut selection = BTreeSet::from([root, child, survivor]);
        tree.sanitize_selection(&mut selection);
        assert_eq!(selection, BTreeSet::from([survivor]));
    }

    #[test]
    fn cycles_are_rejected_without_mutation() {
        let mut tree = GeometryTree::new();
        let a = tree.create("a", None).unwrap();
        let b = tree.create("b", Some(a)).unwrap();
        assert_eq!(
            tree.reparent(a, Some(b), ReparentMode::KeepLocal),
            Err(SceneError::Cycle)
        );
        assert_eq!(tree.node(a).unwrap().parent(), None);
        assert_eq!(tree.node(b).unwrap().parent(), Some(a));
    }

    #[test]
    fn reparent_modes_have_expected_semantics() {
        let mut tree = GeometryTree::new();
        let a = tree.create("a", None).unwrap();
        let b = tree.create("b", None).unwrap();
        let child = tree.create("child", Some(a)).unwrap();
        tree.set_local_transform(a, t(10.0, 0.0, 0.0)).unwrap();
        tree.set_local_transform(b, t(20.0, 0.0, 0.0)).unwrap();
        tree.set_local_transform(child, t(1.0, 0.0, 0.0)).unwrap();
        tree.reparent(child, Some(b), ReparentMode::KeepGlobal)
            .unwrap();
        assert!(
            tree.node(child)
                .unwrap()
                .global
                .translation
                .approx_eq(Vec3::new(11.0, 0.0, 0.0), 1e-5)
        );
        assert!(
            tree.node(child)
                .unwrap()
                .local
                .translation
                .approx_eq(Vec3::new(-9.0, 0.0, 0.0), 1e-5)
        );
        tree.reparent(child, Some(a), ReparentMode::KeepLocal)
            .unwrap();
        assert!(
            tree.node(child)
                .unwrap()
                .global
                .translation
                .approx_eq(Vec3::new(1.0, 0.0, 0.0), 1e-5)
        );
    }

    #[test]
    fn keep_global_handles_parent_rotation() {
        let mut tree = GeometryTree::new();
        let parent = tree.create("parent", None).unwrap();
        tree.set_local_transform(
            parent,
            Transform {
                translation: Vec3::new(2.0, 3.0, 0.0),
                rotation: Quat::from_axis_angle(Vec3::Z, PI),
                scale: Vec3::ONE,
            },
        )
        .unwrap();
        let child = tree.create("child", None).unwrap();
        tree.set_global_transform(child, t(4.0, 3.0, 0.0)).unwrap();
        tree.reparent(child, Some(parent), ReparentMode::KeepGlobal)
            .unwrap();
        assert!(
            tree.node(child)
                .unwrap()
                .global_transform()
                .approx_eq(t(4.0, 3.0, 0.0), 1e-4)
        );
    }

    #[test]
    fn attributes_cascade_and_initial_stops_cascade() {
        let mut tree = GeometryTree::new();
        let root = tree.create("root", None).unwrap();
        let child = tree.create("child", Some(root)).unwrap();
        let leaf = tree.create("leaf", Some(child)).unwrap();
        tree.set_attribute(
            root,
            AttributeKey::Visible,
            AttributeDeclaration::Value(Attribute::Bool(false)),
        )
        .unwrap();
        assert_eq!(
            tree.resolve_attribute(leaf, &AttributeKey::Visible)
                .unwrap(),
            ResolvedAttribute {
                value: Attribute::Bool(false),
                origin: AttributeOrigin::Node(root)
            }
        );
        tree.set_attribute(child, AttributeKey::Visible, AttributeDeclaration::Initial)
            .unwrap();
        assert_eq!(
            tree.resolve_attribute(leaf, &AttributeKey::Visible)
                .unwrap(),
            ResolvedAttribute {
                value: Attribute::Bool(true),
                origin: AttributeOrigin::Initial(child)
            }
        );
        tree.set_attribute(
            leaf,
            AttributeKey::Visible,
            AttributeDeclaration::Value(Attribute::Bool(false)),
        )
        .unwrap();
        assert_eq!(
            tree.resolve_attribute(leaf, &AttributeKey::Visible)
                .unwrap()
                .origin,
            AttributeOrigin::Node(leaf)
        );
        tree.clear_attribute(leaf, &AttributeKey::Visible).unwrap();
        assert_eq!(
            tree.resolve_attribute(leaf, &AttributeKey::Visible)
                .unwrap()
                .origin,
            AttributeOrigin::Initial(child)
        );
    }

    #[test]
    fn reparent_changes_inherited_attribute_source() {
        let mut tree = GeometryTree::new();
        let red = tree.create("red", None).unwrap();
        let blue = tree.create("blue", None).unwrap();
        let leaf = tree.create("leaf", Some(red)).unwrap();
        tree.set_attribute(
            red,
            AttributeKey::Tint,
            AttributeDeclaration::Value(Attribute::Color([1., 0., 0., 1.])),
        )
        .unwrap();
        tree.set_attribute(
            blue,
            AttributeKey::Tint,
            AttributeDeclaration::Value(Attribute::Color([0., 0., 1., 1.])),
        )
        .unwrap();
        tree.reparent(leaf, Some(blue), ReparentMode::KeepGlobal)
            .unwrap();
        assert_eq!(
            tree.resolve_attribute(leaf, &AttributeKey::Tint)
                .unwrap()
                .value,
            Attribute::Color([0., 0., 1., 1.])
        );
    }

    #[test]
    fn buffered_commands_continue_after_invalid_write() {
        let mut tree = GeometryTree::new();
        let node = tree.create("node", None).unwrap();
        let stale = NodeId {
            slot: 99,
            generation: 0,
        };
        let report = tree.apply_commands([
            SceneCommand::SetLocal {
                node: stale,
                transform: t(1.0, 0.0, 0.0),
            },
            SceneCommand::SetAttribute {
                node,
                key: AttributeKey::Enabled,
                declaration: AttributeDeclaration::Value(Attribute::Bool(false)),
            },
        ]);
        assert_eq!(report.applied, 1);
        assert_eq!(report.errors.len(), 1);
        assert_eq!(
            tree.resolve_attribute(node, &AttributeKey::Enabled)
                .unwrap()
                .value,
            Attribute::Bool(false)
        );
    }

    #[test]
    fn zero_scale_keep_global_fails_atomically() {
        let mut tree = GeometryTree::new();
        let child = tree.create("child", None).unwrap();
        let parent = tree.create("parent", None).unwrap();
        tree.set_local_transform(
            parent,
            Transform {
                scale: Vec3::new(0.0, 1.0, 1.0),
                ..Transform::IDENTITY
            },
        )
        .unwrap();
        assert_eq!(
            tree.reparent(child, Some(parent), ReparentMode::KeepGlobal),
            Err(SceneError::NonInvertibleTransform)
        );
        assert_eq!(tree.node(child).unwrap().parent(), None);
    }

    #[test]
    fn transform_scale_composes_per_axis() {
        let parent = Transform {
            scale: Vec3::new(2.0, 3.0, 4.0),
            ..Transform::IDENTITY
        };
        let child = Transform {
            translation: Vec3::new(1.0, 1.0, 1.0),
            scale: Vec3::new(5.0, 6.0, 7.0),
            ..Transform::IDENTITY
        };
        let world = parent.compose(child);
        assert_eq!(world.scale, Vec3::new(10.0, 18.0, 28.0));
        assert_eq!(world.translation, Vec3::new(2.0, 3.0, 4.0));
    }
}
