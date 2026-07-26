use std::collections::{BTreeSet, HashMap};

use zerofps_assets::MeshAsset;
use zerofps_core::Vec3;

#[derive(Clone)]
pub struct MeshScalarField {
    pub name: String,
    pub values: Vec<f32>,
    pub default: f32,
    pub display_min: f32,
    pub display_max: f32,
    pub revision: u64,
}

impl MeshScalarField {
    pub fn mobility_for_mesh(mesh: &MeshAsset) -> Self {
        if let Some(imported) = mesh
            .vertex_scalar_fields
            .get("dynamics.mobility")
            .filter(|field| field.values.len() == mesh.vertices.len())
        {
            return Self {
                name: "dynamics.mobility".into(),
                values: imported
                    .values
                    .iter()
                    .map(|value| finite_or(*value, 0.0).clamp(0.0, 1.0))
                    .collect(),
                default: 0.0,
                display_min: 0.0,
                display_max: 1.0,
                revision: 1,
            };
        }
        let mut values = vec![0.0; mesh.vertices.len()];
        for primitive in &mesh.primitives {
            if !primitive.name.to_ascii_lowercase().contains("cloth") {
                continue;
            }
            let minimum_x = primitive
                .indices
                .iter()
                .filter_map(|index| mesh.vertices.get(*index as usize))
                .map(|vertex| vertex.position[0])
                .fold(f32::INFINITY, f32::min);
            let maximum_x = primitive
                .indices
                .iter()
                .filter_map(|index| mesh.vertices.get(*index as usize))
                .map(|vertex| vertex.position[0])
                .fold(f32::NEG_INFINITY, f32::max);
            let transition = ((maximum_x - minimum_x) / 16.0).max(1.0e-4);
            for index in &primitive.indices {
                if let Some(vertex) = mesh.vertices.get(*index as usize) {
                    values[*index as usize] =
                        ((vertex.position[0] - minimum_x) / transition).clamp(0.0, 1.0);
                }
            }
        }
        Self {
            name: "dynamics.mobility".into(),
            values,
            default: 0.0,
            display_min: 0.0,
            display_max: 1.0,
            revision: 1,
        }
    }

    pub fn paint(
        &mut self,
        vertices: impl IntoIterator<Item = (usize, f32)>,
        value: f32,
        strength: f32,
        mode: PaintMode,
    ) {
        let vertices = vertices.into_iter().collect::<Vec<_>>();
        let smooth_target = if mode == PaintMode::Smooth {
            let (weighted_sum, total_weight) = vertices
                .iter()
                .filter_map(|(index, influence)| {
                    let value = self.values.get(*index).copied()?;
                    (value.is_finite() && influence.is_finite())
                        .then_some((value, influence.max(0.0)))
                })
                .fold((0.0, 0.0), |(sum, weight), (value, influence)| {
                    (sum + value * influence, weight + influence)
                });
            (total_weight > f32::EPSILON).then(|| weighted_sum / total_weight)
        } else {
            None
        };
        let display_min = finite_or(self.display_min, 0.0);
        let display_max = finite_or(self.display_max, 1.0);
        let minimum = display_min.min(display_max);
        let maximum = display_min.max(display_max);
        let default = if self.default.is_finite() {
            self.default.clamp(minimum, maximum)
        } else {
            minimum
        };
        let value = if value.is_finite() { value } else { default };
        let strength = if strength.is_finite() { strength } else { 0.0 };
        let mut changed = false;
        for (index, influence) in vertices {
            let Some(current) = self.values.get_mut(index) else {
                continue;
            };
            if !influence.is_finite() {
                continue;
            }
            let amount = (influence * strength).clamp(0.0, 1.0);
            let next = match mode {
                PaintMode::Replace => *current + (value - *current) * amount,
                PaintMode::Add => *current + value * amount,
                PaintMode::Subtract => *current - value * amount,
                PaintMode::Smooth => {
                    let target = smooth_target.unwrap_or(*current);
                    *current + (target - *current) * amount
                }
                PaintMode::Erase => *current + (default - *current) * amount,
            }
            .clamp(minimum, maximum);
            changed |= next != *current;
            *current = next;
        }
        if changed {
            self.revision = self.revision.wrapping_add(1);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PaintMode {
    Replace,
    Add,
    Subtract,
    Smooth,
    Erase,
}

impl PaintMode {
    pub const ALL: [Self; 5] = [
        Self::Replace,
        Self::Add,
        Self::Subtract,
        Self::Smooth,
        Self::Erase,
    ];

    pub fn label(self) -> &'static str {
        match self {
            Self::Replace => "Replace",
            Self::Add => "Add",
            Self::Subtract => "Subtract",
            Self::Smooth => "Smooth",
            Self::Erase => "Erase",
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct ClothSettings {
    pub particle_mass: f32,
    pub stretch_compliance: f32,
    pub bend_compliance: f32,
    pub damping: f32,
    pub iterations: usize,
    pub wind_drag: f32,
    pub collision_radius: f32,
    pub ground_z: f32,
}

impl Default for ClothSettings {
    fn default() -> Self {
        Self {
            particle_mass: 0.08,
            stretch_compliance: 0.000_002,
            bend_compliance: 0.000_8,
            damping: 0.025,
            iterations: 7,
            wind_drag: 1.2,
            collision_radius: 0.13,
            ground_z: 0.0,
        }
    }
}

#[derive(Clone, PartialEq)]
pub struct WindField {
    pub enabled: bool,
    pub velocity: Vec3,
    pub gust_strength: f32,
    pub gust_frequency: f32,
    pub turbulence: f32,
}

impl Default for WindField {
    fn default() -> Self {
        Self {
            enabled: true,
            velocity: Vec3::new(0.4, -4.0, 0.2),
            gust_strength: 0.8,
            gust_frequency: 0.65,
            turbulence: 0.25,
        }
    }
}

impl WindField {
    fn sample(&self, position: Vec3, time: f32) -> Vec3 {
        if !self.enabled {
            return Vec3::ZERO;
        }
        let time = finite_or(time, 0.0);
        let position = finite_vec3_or(position, Vec3::ZERO);
        let gust_strength = finite_or(self.gust_strength, 0.0).max(0.0);
        let gust_frequency = finite_or(self.gust_frequency, 0.0).max(0.0);
        let turbulence = finite_or(self.turbulence, 0.0).max(0.0);
        let velocity = finite_vec3_or(self.velocity, Vec3::ZERO);
        let gust = 1.0 + gust_strength * (time * gust_frequency * std::f32::consts::TAU).sin();
        let noise = Vec3::new(
            (position.y * 1.7 + time * 1.3).sin(),
            (position.z * 1.1 - time * 0.9).cos(),
            (position.x * 1.9 + time * 0.7).sin(),
        ) * turbulence;
        velocity * gust.max(0.0) + noise
    }
}

#[derive(Clone)]
struct Particle {
    position: Vec3,
    previous: Vec3,
    rest: Vec3,
    inverse_mass: f32,
}

#[derive(Clone)]
struct DistanceConstraint {
    a: usize,
    b: usize,
    rest_length: f32,
    kind: ConstraintKind,
    lambda: f32,
}

#[derive(Clone, Copy)]
enum ConstraintKind {
    Stretch,
    Bend,
}

#[derive(Clone)]
pub struct DeformationSnapshot {
    pub positions: Vec<[f32; 3]>,
    pub normals: Vec<[f32; 3]>,
    pub revision: u64,
}

pub struct ClothState {
    particles: Vec<Particle>,
    triangles: Vec<[usize; 3]>,
    constraints: Vec<DistanceConstraint>,
    rest_normals: Vec<[f32; 3]>,
    pub snapshot: DeformationSnapshot,
    field_revision: u64,
    particle_mass: f32,
    particle_masses: Vec<f32>,
    pub settings: ClothSettings,
}

impl ClothState {
    pub fn new(mesh: &MeshAsset, mobility: &MeshScalarField, settings: ClothSettings) -> Self {
        let particles = mesh
            .vertices
            .iter()
            .enumerate()
            .map(|(index, vertex)| {
                let position = from_array(vertex.position);
                let mobility = mobility.values.get(index).copied().unwrap_or(0.0);
                Particle {
                    position,
                    previous: position,
                    rest: position,
                    inverse_mass: finite_or(mobility, 0.0).max(0.0)
                        / finite_or(settings.particle_mass, 0.08).max(1.0e-5),
                }
            })
            .collect::<Vec<_>>();
        let triangles = mesh
            .primitives
            .iter()
            .flat_map(|primitive| {
                primitive.indices.chunks_exact(3).map(|triangle| {
                    [
                        triangle[0] as usize,
                        triangle[1] as usize,
                        triangle[2] as usize,
                    ]
                })
            })
            .filter(|triangle| {
                triangle.iter().all(|index| *index < particles.len())
                    && triangle[0] != triangle[1]
                    && triangle[1] != triangle[2]
                    && triangle[2] != triangle[0]
            })
            .collect::<Vec<_>>();
        let mut edges = BTreeSet::new();
        for triangle in &triangles {
            for (a, b) in [
                (triangle[0], triangle[1]),
                (triangle[1], triangle[2]),
                (triangle[2], triangle[0]),
            ] {
                edges.insert((a.min(b), a.max(b)));
            }
        }
        let mut adjacency = HashMap::<usize, BTreeSet<usize>>::new();
        for &(a, b) in &edges {
            adjacency.entry(a).or_default().insert(b);
            adjacency.entry(b).or_default().insert(a);
        }
        let mut constraints = edges
            .iter()
            .map(|&(a, b)| DistanceConstraint {
                a,
                b,
                rest_length: (particles[b].rest - particles[a].rest).length(),
                kind: ConstraintKind::Stretch,
                lambda: 0.0,
            })
            .collect::<Vec<_>>();
        let mut bend_edges = BTreeSet::new();
        for (&center, neighbors) in &adjacency {
            for &near in neighbors {
                if let Some(second_ring) = adjacency.get(&near) {
                    for &far in second_ring {
                        if far != center && !neighbors.contains(&far) {
                            bend_edges.insert((center.min(far), center.max(far)));
                        }
                    }
                }
            }
        }
        constraints.extend(bend_edges.into_iter().map(|(a, b)| DistanceConstraint {
            a,
            b,
            rest_length: (particles[b].rest - particles[a].rest).length(),
            kind: ConstraintKind::Bend,
            lambda: 0.0,
        }));
        let snapshot = DeformationSnapshot {
            positions: mesh.vertices.iter().map(|vertex| vertex.position).collect(),
            normals: mesh.vertices.iter().map(|vertex| vertex.normal).collect(),
            revision: 1,
        };
        Self {
            particles,
            triangles,
            constraints,
            rest_normals: mesh.vertices.iter().map(|vertex| vertex.normal).collect(),
            snapshot,
            field_revision: mobility.revision,
            particle_mass: settings.particle_mass,
            particle_masses: vec![
                finite_or(settings.particle_mass, 0.08).max(1.0e-5);
                mesh.vertices.len()
            ],
            settings,
        }
    }

    pub fn reset(&mut self) {
        for particle in &mut self.particles {
            particle.position = particle.rest;
            particle.previous = particle.rest;
        }
        self.publish();
    }

    pub fn sync_mobility(&mut self, field: &MeshScalarField) {
        if self.field_revision == field.revision
            && self.particle_mass.to_bits() == self.settings.particle_mass.to_bits()
        {
            return;
        }
        if self.particle_mass.to_bits() != self.settings.particle_mass.to_bits() {
            let uniform_mass = finite_or(self.settings.particle_mass, 0.08).max(1.0e-5);
            self.particle_masses.fill(uniform_mass);
        }
        for (index, particle) in self.particles.iter_mut().enumerate() {
            let mobility = field.values.get(index).copied().unwrap_or(0.0).max(0.0);
            let particle_mass = self
                .particle_masses
                .get(index)
                .copied()
                .unwrap_or(self.settings.particle_mass)
                .max(1.0e-5);
            particle.inverse_mass = mobility / particle_mass;
            if mobility <= 1.0e-5 {
                particle.position = particle.rest;
                particle.previous = particle.rest;
            }
        }
        self.field_revision = field.revision;
        self.particle_mass = self.settings.particle_mass;
    }

    pub fn sync_particle_masses(&mut self, field: &MeshScalarField, masses: &[f32]) {
        if masses.len() != self.particles.len() {
            return;
        }
        self.particle_masses.clear();
        self.particle_masses.extend(
            masses
                .iter()
                .map(|mass| finite_or(*mass, self.settings.particle_mass).max(1.0e-6)),
        );
        self.particle_mass = f32::NAN;
        self.field_revision = field.revision;
        for (index, particle) in self.particles.iter_mut().enumerate() {
            let mobility = field.values.get(index).copied().unwrap_or(0.0).max(0.0);
            particle.inverse_mass = mobility / self.particle_masses[index];
            if mobility <= 1.0e-5 {
                particle.position = particle.rest;
                particle.previous = particle.rest;
            }
        }
    }

    pub fn step(&mut self, dt: f32, time: f32, wind: &WindField) {
        self.step_with_forces(dt, time, wind, None);
    }

    pub fn step_with_forces(
        &mut self,
        dt: f32,
        time: f32,
        wind: &WindField,
        external_forces: Option<&[Vec3]>,
    ) {
        self.step_with_fields(dt, time, wind, external_forces, true);
    }

    pub fn step_with_fields(
        &mut self,
        dt: f32,
        time: f32,
        wind: &WindField,
        external_forces: Option<&[Vec3]>,
        gravity_enabled: bool,
    ) {
        let dt = finite_or(dt, 1.0 / 60.0).clamp(1.0 / 240.0, 1.0 / 20.0);
        let damping = finite_or(self.settings.damping, 0.0).clamp(0.0, 1.0);
        let wind_drag = finite_or(self.settings.wind_drag, 0.0).max(0.0);
        let gravity = if gravity_enabled {
            Vec3::new(0.0, 0.0, -9.81)
        } else {
            Vec3::ZERO
        };
        let mut accelerations = vec![gravity; self.particles.len()];
        if let Some(forces) = external_forces {
            for (index, acceleration) in accelerations.iter_mut().enumerate() {
                if let Some(force) = forces.get(index).copied() {
                    *acceleration = *acceleration
                        + finite_vec3_or(force, Vec3::ZERO) * self.particles[index].inverse_mass;
                }
            }
        }
        for triangle in &self.triangles {
            let [a, b, c] = *triangle;
            let pa = self.particles[a].position;
            let pb = self.particles[b].position;
            let pc = self.particles[c].position;
            let area_normal = (pb - pa).cross(pc - pa) * 0.5;
            let area = area_normal.length();
            if area <= 1.0e-8 {
                continue;
            }
            let normal = area_normal / area;
            let center = (pa + pb + pc) / 3.0;
            let air = wind.sample(center, time);
            let surface_velocity = ((pa - self.particles[a].previous)
                + (pb - self.particles[b].previous)
                + (pc - self.particles[c].previous))
                / (3.0 * dt);
            let relative = air - surface_velocity;
            let normal_speed = relative.dot(normal);
            let aerodynamic = normal * normal_speed * normal_speed.abs() * area * wind_drag;
            for index in [a, b, c] {
                accelerations[index] =
                    accelerations[index] + aerodynamic * (self.particles[index].inverse_mass / 3.0);
            }
        }
        for (index, particle) in self.particles.iter_mut().enumerate() {
            if particle.inverse_mass <= 0.0 {
                particle.position = particle.rest;
                particle.previous = particle.rest;
                continue;
            }
            let velocity = (particle.position - particle.previous) * (1.0 - damping);
            particle.previous = particle.position;
            particle.position = particle.position + velocity + accelerations[index] * (dt * dt);
        }
        for constraint in &mut self.constraints {
            constraint.lambda = 0.0;
        }
        for _ in 0..self.settings.iterations.max(1) {
            for constraint in &mut self.constraints {
                let a = constraint.a;
                let b = constraint.b;
                let delta = self.particles[b].position - self.particles[a].position;
                let length = delta.length();
                if length <= 1.0e-7 {
                    continue;
                }
                let wa = self.particles[a].inverse_mass;
                let wb = self.particles[b].inverse_mass;
                let compliance = match constraint.kind {
                    ConstraintKind::Stretch => self.settings.stretch_compliance,
                    ConstraintKind::Bend => self.settings.bend_compliance,
                };
                let alpha = finite_or(compliance, 0.0).max(0.0) / (dt * dt);
                let denominator = wa + wb + alpha;
                if denominator <= 1.0e-8 {
                    continue;
                }
                let constraint_value = length - constraint.rest_length;
                let delta_lambda = (-constraint_value - alpha * constraint.lambda) / denominator;
                constraint.lambda += delta_lambda;
                let direction = delta / length;
                self.particles[a].position =
                    self.particles[a].position - direction * (wa * delta_lambda);
                self.particles[b].position =
                    self.particles[b].position + direction * (wb * delta_lambda);
            }
            self.solve_collisions();
            for particle in &mut self.particles {
                if particle.inverse_mass <= 0.0 {
                    particle.position = particle.rest;
                }
            }
        }
        self.publish();
    }

    pub fn apply_velocity_field(&mut self, dt: f32, velocities: &[Vec3], blend: f32, mode: usize) {
        let dt = finite_or(dt, 1.0 / 60.0).max(1.0e-6);
        let blend = finite_or(blend, 1.0).clamp(0.0, 1.0);
        for (index, particle) in self.particles.iter_mut().enumerate() {
            if particle.inverse_mass <= 0.0 {
                continue;
            }
            let target = velocities
                .get(index)
                .copied()
                .map(|value| finite_vec3_or(value, Vec3::ZERO))
                .unwrap_or(Vec3::ZERO);
            let current = (particle.position - particle.previous) / dt;
            let velocity = match mode {
                0 => current + (target - current) * blend,
                1 => current + target * blend,
                _ => current + (target - current) * (blend * dt).clamp(0.0, 1.0),
            };
            particle.previous = particle.position - velocity * dt;
        }
    }

    /// Applies `x <- x + dt * V(p)` without adding `V(p)` to the persistent
    /// particle momentum encoded by the Verlet position pair.
    pub fn advect_velocity_field(&mut self, dt: f32, velocities: &[Vec3], blend: f32) {
        let dt = finite_or(dt, 1.0 / 60.0).max(0.0);
        let blend = finite_or(blend, 1.0).clamp(0.0, 1.0);
        for (index, particle) in self.particles.iter_mut().enumerate() {
            if particle.inverse_mass <= 0.0 {
                continue;
            }
            let field_velocity = velocities
                .get(index)
                .copied()
                .map(|value| finite_vec3_or(value, Vec3::ZERO))
                .unwrap_or(Vec3::ZERO);
            let displacement = field_velocity * (dt * blend);
            particle.position = particle.position + displacement;
            particle.previous = particle.previous + displacement;
        }
    }

    fn solve_collisions(&mut self) {
        for particle in &mut self.particles {
            if particle.inverse_mass <= 0.0 {
                continue;
            }
            particle.position.z = particle.position.z.max(self.settings.ground_z);
            if particle.position.z >= 0.25 && particle.position.z <= 6.4 {
                let radial = Vec3::new(particle.position.x, particle.position.y, 0.0);
                let length = radial.length();
                let collision_radius = finite_or(self.settings.collision_radius, 0.0).max(0.0);
                if length < collision_radius && length > 1.0e-7 {
                    let pushed = radial / length * collision_radius;
                    particle.position.x = pushed.x;
                    particle.position.y = pushed.y;
                }
            }
        }
    }

    fn publish(&mut self) {
        self.snapshot.positions = self
            .particles
            .iter()
            .map(|particle| to_array(particle.position))
            .collect();
        self.snapshot.normals.clone_from(&self.rest_normals);
        let mut cloth_vertices = vec![false; self.particles.len()];
        for triangle in &self.triangles {
            for index in triangle {
                cloth_vertices[*index] = true;
                self.snapshot.normals[*index] = [0.0; 3];
            }
        }
        for [a, b, c] in &self.triangles {
            let normal = (self.particles[*b].position - self.particles[*a].position)
                .cross(self.particles[*c].position - self.particles[*a].position);
            for index in [*a, *b, *c] {
                let previous = from_array(self.snapshot.normals[index]);
                self.snapshot.normals[index] = to_array(previous + normal);
            }
        }
        for (index, normal) in self.snapshot.normals.iter_mut().enumerate() {
            if cloth_vertices[index] {
                let accumulated = from_array(*normal);
                let normalized = if accumulated.length() > 1.0e-7 {
                    accumulated.normalized()
                } else {
                    from_array(self.rest_normals[index]).normalized()
                };
                *normal = to_array(normalized);
            }
        }
        self.snapshot.revision = self.snapshot.revision.wrapping_add(1);
    }
}

pub fn heatmap(value: f32, minimum: f32, maximum: f32) -> [f32; 4] {
    if !value.is_finite() || !minimum.is_finite() || !maximum.is_finite() {
        return [1.0, 0.0, 1.0, 1.0];
    }
    let t = if (maximum - minimum).abs() <= f32::EPSILON {
        0.0
    } else {
        ((value - minimum) / (maximum - minimum)).clamp(0.0, 1.0)
    };
    let red = (1.5 - (4.0 * t - 3.0).abs()).clamp(0.0, 1.0);
    let green = (1.5 - (4.0 * t - 2.0).abs()).clamp(0.0, 1.0);
    let blue = (1.5 - (4.0 * t - 1.0).abs()).clamp(0.0, 1.0);
    [red, green, blue, 1.0]
}

fn from_array(value: [f32; 3]) -> Vec3 {
    Vec3::new(value[0], value[1], value[2])
}

fn to_array(value: Vec3) -> [f32; 3] {
    [value.x, value.y, value.z]
}

fn finite_or(value: f32, fallback: f32) -> f32 {
    if value.is_finite() { value } else { fallback }
}

fn finite_vec3_or(value: Vec3, fallback: Vec3) -> Vec3 {
    if value.x.is_finite() && value.y.is_finite() && value.z.is_finite() {
        value
    } else {
        fallback
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use zerofps_assets::{Primitive, Vertex, VertexScalarField};

    fn grid() -> MeshAsset {
        MeshAsset {
            vertices: vec![
                Vertex {
                    position: [0.0, 0.0, 1.0],
                    ..Vertex::default()
                },
                Vertex {
                    position: [1.0, 0.0, 1.0],
                    ..Vertex::default()
                },
                Vertex {
                    position: [0.0, 0.0, 0.0],
                    ..Vertex::default()
                },
                Vertex {
                    position: [1.0, 0.0, 0.0],
                    ..Vertex::default()
                },
            ],
            primitives: vec![Primitive {
                name: "cloth".into(),
                indices: vec![0, 2, 1, 1, 2, 3],
                ..Primitive::default()
            }],
            ..MeshAsset::default()
        }
    }

    #[test]
    fn mobility_initialization_pins_minimum_x() {
        let field = MeshScalarField::mobility_for_mesh(&grid());
        assert_eq!(field.values[0], 0.0);
        assert_eq!(field.values[2], 0.0);
        assert_eq!(field.values[1], 1.0);
    }

    #[test]
    fn particle_mass_field_changes_inverse_mass_without_unpinning() {
        let mesh = grid();
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        cloth.sync_particle_masses(&field, &[1.0, 2.0, 1.0, 4.0]);
        assert_eq!(cloth.particles[0].inverse_mass, 0.0);
        assert!((cloth.particles[1].inverse_mass - 0.5).abs() < 1.0e-6);
        assert!((cloth.particles[3].inverse_mass - 0.25).abs() < 1.0e-6);
    }

    #[test]
    fn external_cartesian_force_moves_only_mobile_particles() {
        let mesh = grid();
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        let mut baseline = ClothState::new(&mesh, &field, ClothSettings::default());
        let forces = vec![Vec3::new(100.0, 0.0, 0.0); mesh.vertices.len()];
        let wind = WindField {
            enabled: false,
            ..WindField::default()
        };
        baseline.step(1.0 / 60.0, 0.0, &wind);
        cloth.step_with_forces(1.0 / 60.0, 0.0, &wind, Some(&forces));
        assert_eq!(cloth.snapshot.positions[0], mesh.vertices[0].position);
        assert_ne!(cloth.snapshot.positions[1], baseline.snapshot.positions[1]);
    }

    #[test]
    fn velocity_field_sets_mobile_velocity_without_moving_pins() {
        let mesh = grid();
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        let velocities = vec![Vec3::new(3.0, 0.0, 0.0); mesh.vertices.len()];
        let pinned_previous = cloth.particles[0].previous;
        cloth.apply_velocity_field(0.25, &velocities, 1.0, 0);
        assert_eq!(cloth.particles[0].previous, pinned_previous);
        let velocity = (cloth.particles[1].position - cloth.particles[1].previous) / 0.25;
        assert!((velocity.x - 3.0).abs() < 1.0e-6);
    }

    #[test]
    fn additive_velocity_field_accumulates_once_per_tick() {
        let mesh = grid();
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        let velocities = vec![Vec3::new(2.0, -1.0, 0.5); mesh.vertices.len()];
        let dt = 0.25;
        cloth.apply_velocity_field(dt, &velocities, 1.0, 1);
        cloth.apply_velocity_field(dt, &velocities, 1.0, 1);
        let velocity = (cloth.particles[1].position - cloth.particles[1].previous) / dt;
        assert!((velocity.x - 4.0).abs() < 1.0e-6);
        assert!((velocity.y + 2.0).abs() < 1.0e-6);
        assert!((velocity.z - 1.0).abs() < 1.0e-6);
    }

    #[test]
    fn velocity_field_advects_without_becoming_particle_momentum() {
        let mesh = grid();
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        let velocities = vec![Vec3::new(2.0, -1.0, 0.5); mesh.vertices.len()];
        let dt = 0.25;
        let before = cloth.particles[1].position;
        let momentum_before = cloth.particles[1].position - cloth.particles[1].previous;
        cloth.advect_velocity_field(dt, &velocities, 1.0);
        let displacement = cloth.particles[1].position - before;
        let momentum_after = cloth.particles[1].position - cloth.particles[1].previous;
        assert!((displacement.x - 0.5).abs() < 1.0e-6);
        assert!((displacement.y + 0.25).abs() < 1.0e-6);
        assert!((displacement.z - 0.125).abs() < 1.0e-6);
        assert_eq!(momentum_after, momentum_before);
        assert_eq!(
            cloth.particles[0].position,
            from_array(mesh.vertices[0].position)
        );
    }

    #[test]
    fn damping_monotonically_reduces_particle_velocity() {
        let mesh = grid();
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut undamped_settings = ClothSettings::default();
        undamped_settings.damping = 0.0;
        let mut damped_settings = undamped_settings.clone();
        damped_settings.damping = 0.5;
        let mut undamped = ClothState::new(&mesh, &field, undamped_settings);
        let mut damped = ClothState::new(&mesh, &field, damped_settings);
        let initial = vec![Vec3::new(1.0, 0.0, 0.0); mesh.vertices.len()];
        let dt = 1.0 / 60.0;
        undamped.apply_velocity_field(dt, &initial, 1.0, 0);
        damped.apply_velocity_field(dt, &initial, 1.0, 0);
        let wind = WindField {
            enabled: false,
            ..WindField::default()
        };
        undamped.step_with_fields(dt, 0.0, &wind, None, false);
        damped.step_with_fields(dt, 0.0, &wind, None, false);
        let undamped_speed =
            (undamped.particles[1].position - undamped.particles[1].previous).length() / dt;
        let damped_speed =
            (damped.particles[1].position - damped.particles[1].previous).length() / dt;
        assert!(damped_speed < undamped_speed);
    }

    #[test]
    fn imported_mobility_takes_precedence_and_is_sanitized() {
        let mut mesh = grid();
        mesh.vertex_scalar_fields.insert(
            "dynamics.mobility".into(),
            VertexScalarField {
                values: vec![-1.0, 0.25, f32::NAN, 2.0],
            },
        );
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        assert_eq!(field.values, vec![0.0, 0.25, 0.0, 1.0]);
    }

    #[test]
    fn paint_replace_and_subtract_are_bounded() {
        let mut field = MeshScalarField::mobility_for_mesh(&grid());
        field.paint([(0, 1.0)], 1.0, 0.5, PaintMode::Replace);
        assert_eq!(field.values[0], 0.5);
        field.paint([(0, 1.0)], 2.0, 1.0, PaintMode::Subtract);
        assert_eq!(field.values[0], 0.0);
    }

    #[test]
    fn paint_smooth_moves_values_toward_stroke_average() {
        let mut field = MeshScalarField::mobility_for_mesh(&grid());
        let original_revision = field.revision;
        field.values = vec![0.0, 1.0, 0.0, 1.0];
        field.paint([(0, 1.0), (1, 1.0)], 0.0, 1.0, PaintMode::Smooth);
        assert_eq!(field.values[0], 0.5);
        assert_eq!(field.values[1], 0.5);
        assert_eq!(field.revision, original_revision + 1);

        let revision = field.revision;
        field.paint([(99, 1.0)], 1.0, 1.0, PaintMode::Replace);
        assert_eq!(field.revision, revision);
    }

    #[test]
    fn cloth_keeps_pinned_vertices_fixed() {
        let mesh = grid();
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        for step in 0..10 {
            cloth.step(1.0 / 60.0, step as f32 / 60.0, &WindField::default());
        }
        assert_eq!(cloth.snapshot.positions[0], mesh.vertices[0].position);
        assert_ne!(cloth.snapshot.positions[1], mesh.vertices[1].position);
    }

    #[test]
    fn mobility_sync_applies_particle_mass_changes() {
        let mesh = grid();
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        let before = cloth.particles[1].inverse_mass;
        cloth.settings.particle_mass *= 2.0;
        cloth.sync_mobility(&field);
        assert!((cloth.particles[1].inverse_mass - before * 0.5).abs() < 1.0e-6);
    }

    #[test]
    fn cloth_ignores_invalid_and_degenerate_triangles() {
        let mut mesh = grid();
        mesh.primitives[0].indices.extend([0, 0, 1, 0, 1, 99]);
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        cloth.step(1.0 / 60.0, 0.0, &WindField::default());
        assert_eq!(cloth.triangles.len(), 2);
        assert!(
            cloth
                .snapshot
                .positions
                .iter()
                .flatten()
                .all(|component| component.is_finite())
        );
    }

    #[test]
    fn publishing_cloth_preserves_non_cloth_normals() {
        let mut mesh = grid();
        mesh.vertices.push(Vertex {
            position: [2.0, 0.0, 0.0],
            normal: [1.0, 0.0, 0.0],
            ..Vertex::default()
        });
        mesh.primitives.push(Primitive {
            name: "pole".into(),
            indices: vec![4, 4, 4],
            ..Primitive::default()
        });
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        cloth.step(1.0 / 60.0, 0.0, &WindField::default());
        assert_eq!(cloth.snapshot.normals[4], [1.0, 0.0, 0.0]);
    }

    #[test]
    fn heatmap_marks_invalid_values_as_debug_magenta() {
        assert_eq!(heatmap(f32::NAN, 0.0, 1.0), [1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn generated_flag_field_drives_finite_cloth_when_fixture_exists() {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../appdata/models/Tripo/flag_model.glb");
        if !path.is_file() {
            return;
        }
        let mesh = zerofps_assets::import_file(path).expect("generated flag imports");
        let field = MeshScalarField::mobility_for_mesh(&mesh);
        let fixed = field
            .values
            .iter()
            .enumerate()
            .find_map(|(index, value)| (*value == 0.0).then_some(index))
            .unwrap();
        let free = field
            .values
            .iter()
            .enumerate()
            .find_map(|(index, value)| (*value >= 1.0).then_some(index))
            .unwrap();
        let mut cloth = ClothState::new(&mesh, &field, ClothSettings::default());
        for step in 0..30 {
            cloth.step(1.0 / 60.0, step as f32 / 60.0, &WindField::default());
        }
        assert_eq!(
            cloth.snapshot.positions[fixed],
            mesh.vertices[fixed].position
        );
        assert_ne!(cloth.snapshot.positions[free], mesh.vertices[free].position);
        assert!(
            cloth
                .snapshot
                .positions
                .iter()
                .flatten()
                .all(|value| value.is_finite())
        );
    }
}
