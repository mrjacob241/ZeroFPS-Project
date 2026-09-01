//! Reusable conservative triangle-mesh voxelization.
//!
//! The crate deliberately accepts plain triangles and has no dependency on a
//! renderer, asset format, or FEM implementation. The current backend marks
//! every grid cell touched by a triangle and classifies the remaining cells by
//! a six-connected exterior flood fill.

use std::collections::{HashMap, VecDeque};

use thiserror::Error;

pub type Point3 = [f32; 3];
pub type Triangle = [Point3; 3];

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
#[repr(u8)]
pub enum Cell {
    Exterior = 0,
    Boundary = 1,
    Interior = 2,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct VoxelizeOptions {
    /// Number of cells along the longest unpadded mesh axis.
    pub longest_axis_cells: u32,
    /// Empty cells added on every side for reliable exterior flood filling.
    pub padding: u32,
}

impl Default for VoxelizeOptions {
    fn default() -> Self {
        Self {
            longest_axis_cells: 64,
            padding: 1,
        }
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct VoxelGrid {
    pub origin: Point3,
    pub cell_size: f32,
    pub dimensions: [u32; 3],
    mesh_bounds: [Point3; 2],
    cells: Vec<Cell>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct CellCounts {
    pub exterior: usize,
    pub boundary: usize,
    pub interior: usize,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SurfaceMesh {
    pub positions: Vec<Point3>,
    pub normals: Vec<Point3>,
    pub indices: Vec<u32>,
}

/// Conforming volume mesh derived from occupied voxel cells. Neighboring
/// cubes share vertices and face diagonals, so the result has no T-junctions.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct TetrahedralMesh {
    pub positions: Vec<Point3>,
    pub tetrahedra: Vec<[u32; 4]>,
}

#[derive(Debug, Error, PartialEq)]
pub enum VoxelizeError {
    #[error("mesh contains no triangles")]
    EmptyMesh,
    #[error("mesh contains a non-finite coordinate")]
    NonFiniteCoordinate,
    #[error("mesh has zero extent")]
    ZeroExtent,
    #[error("longest_axis_cells must be greater than zero")]
    ZeroResolution,
    #[error("voxel grid is too large for this platform")]
    GridTooLarge,
}

impl VoxelGrid {
    pub fn cells(&self) -> &[Cell] {
        &self.cells
    }

    pub fn get(&self, coordinate: [u32; 3]) -> Option<Cell> {
        self.index(coordinate).map(|index| self.cells[index])
    }

    pub fn coordinate_of(&self, point: Point3) -> Option<[u32; 3]> {
        let mut coordinate = [0; 3];
        for axis in 0..3 {
            let value = ((point[axis] - self.origin[axis]) / self.cell_size).floor();
            if value < 0.0 || value >= self.dimensions[axis] as f32 {
                return None;
            }
            coordinate[axis] = value as u32;
        }
        Some(coordinate)
    }

    pub fn cell_center(&self, coordinate: [u32; 3]) -> Option<Point3> {
        self.index(coordinate)?;
        Some(std::array::from_fn(|axis| {
            self.origin[axis] + (coordinate[axis] as f32 + 0.5) * self.cell_size
        }))
    }

    pub fn sample(&self, point: Point3) -> Option<Cell> {
        self.coordinate_of(point)
            .and_then(|coordinate| self.get(coordinate))
    }

    pub fn counts(&self) -> CellCounts {
        let mut counts = CellCounts::default();
        for cell in &self.cells {
            match cell {
                Cell::Exterior => counts.exterior += 1,
                Cell::Boundary => counts.boundary += 1,
                Cell::Interior => counts.interior += 1,
            }
        }
        counts
    }

    pub fn occupied_volume(&self) -> f64 {
        let counts = self.counts();
        (counts.boundary + counts.interior) as f64 * f64::from(self.cell_size).powi(3)
    }

    /// Extracts only faces separating occupied cells from exterior space.
    /// Vertices are intentionally face-local, providing flat normals and a
    /// simple interchange mesh suitable for rendering or export.
    pub fn surface_mesh(&self) -> SurfaceMesh {
        const FACES: [([i32; 3], [usize; 4], Point3); 6] = [
            ([-1, 0, 0], [0, 4, 7, 3], [-1.0, 0.0, 0.0]),
            ([1, 0, 0], [1, 2, 6, 5], [1.0, 0.0, 0.0]),
            ([0, -1, 0], [0, 1, 5, 4], [0.0, -1.0, 0.0]),
            ([0, 1, 0], [3, 7, 6, 2], [0.0, 1.0, 0.0]),
            ([0, 0, -1], [0, 3, 2, 1], [0.0, 0.0, -1.0]),
            ([0, 0, 1], [4, 5, 6, 7], [0.0, 0.0, 1.0]),
        ];
        const CORNERS: [[f32; 3]; 8] = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        let mut mesh = SurfaceMesh::default();
        for z in 0..self.dimensions[2] {
            for y in 0..self.dimensions[1] {
                for x in 0..self.dimensions[0] {
                    let coordinate = [x, y, z];
                    if self.get(coordinate) == Some(Cell::Exterior) {
                        continue;
                    }
                    for (offset, corners, normal) in FACES {
                        let neighbor = std::array::from_fn(|axis| {
                            coordinate[axis] as i64 + offset[axis] as i64
                        });
                        let exposed = neighbor
                            .iter()
                            .zip(self.dimensions)
                            .any(|(&value, limit)| value < 0 || value >= limit as i64)
                            || self.get(neighbor.map(|value| value as u32)) == Some(Cell::Exterior);
                        if !exposed {
                            continue;
                        }
                        let base = mesh.positions.len() as u32;
                        for corner in corners {
                            mesh.positions.push(std::array::from_fn(|axis| {
                                self.origin[axis]
                                    + (coordinate[axis] as f32 + CORNERS[corner][axis])
                                        * self.cell_size
                            }));
                            mesh.normals.push(normal);
                        }
                        mesh.indices
                            .extend([base, base + 1, base + 2, base, base + 2, base + 3]);
                    }
                }
            }
        }
        mesh
    }

    /// Splits every occupied cubic cell into six conforming tetrahedra around
    /// the same body diagonal. The outer coordinates are clamped to the input
    /// triangle mesh bounds, avoiding the voxel padding/half-cell overshoot.
    pub fn tetrahedral_mesh(&self) -> Result<TetrahedralMesh, VoxelizeError> {
        const CORNERS: [[u32; 3]; 8] = [
            [0, 0, 0],
            [1, 0, 0],
            [1, 1, 0],
            [0, 1, 0],
            [0, 0, 1],
            [1, 0, 1],
            [1, 1, 1],
            [0, 1, 1],
        ];
        const SPLIT: [[usize; 4]; 6] = [
            [0, 1, 2, 6],
            [0, 2, 3, 6],
            [0, 3, 7, 6],
            [0, 7, 4, 6],
            [0, 4, 5, 6],
            [0, 5, 1, 6],
        ];
        const MAX_TETRAHEDRA: usize = 4_000_000;

        let occupied = self
            .cells
            .iter()
            .filter(|cell| **cell != Cell::Exterior)
            .count();
        let requested = occupied
            .checked_mul(SPLIT.len())
            .ok_or(VoxelizeError::GridTooLarge)?;
        if requested > MAX_TETRAHEDRA {
            return Err(VoxelizeError::GridTooLarge);
        }

        let mut mesh = TetrahedralMesh {
            positions: Vec::with_capacity(occupied.saturating_mul(2)),
            tetrahedra: Vec::with_capacity(requested),
        };
        let mut vertices = HashMap::<[u32; 3], u32>::new();
        let minimum_volume6 = self.cell_size.powi(3) * 1.0e-6;
        for z in 0..self.dimensions[2] {
            for y in 0..self.dimensions[1] {
                for x in 0..self.dimensions[0] {
                    if self.get([x, y, z]) == Some(Cell::Exterior) {
                        continue;
                    }
                    let lattice =
                        CORNERS.map(|corner| [x + corner[0], y + corner[1], z + corner[2]]);
                    let indices = lattice.map(|coordinate| {
                        *vertices.entry(coordinate).or_insert_with(|| {
                            let index = mesh.positions.len() as u32;
                            mesh.positions.push(std::array::from_fn(|axis| {
                                (self.origin[axis] + coordinate[axis] as f32 * self.cell_size)
                                    .clamp(self.mesh_bounds[0][axis], self.mesh_bounds[1][axis])
                            }));
                            index
                        })
                    });
                    for tetrahedron in SPLIT {
                        let mut candidate = tetrahedron.map(|corner| indices[corner]);
                        let points = candidate.map(|index| mesh.positions[index as usize]);
                        let signed_volume6 = tetrahedron_signed_volume6(points);
                        if signed_volume6.abs() > minimum_volume6 {
                            if signed_volume6 < 0.0 {
                                candidate.swap(2, 3);
                            }
                            mesh.tetrahedra.push(candidate);
                        }
                    }
                }
            }
        }
        Ok(mesh)
    }

    /// Builds the voxel-derived volume and snaps only its exterior vertices to
    /// the closest points on the source triangle mesh. Interior lattice points
    /// remain regular, preserving useful element quality away from the skin.
    pub fn fitted_tetrahedral_mesh(
        &self,
        source: &[Triangle],
    ) -> Result<TetrahedralMesh, VoxelizeError> {
        let mut mesh = self.tetrahedral_mesh()?;
        let buckets = triangle_buckets(self, source);
        let mut incident = vec![Vec::<usize>::new(); mesh.positions.len()];
        for (tetrahedron_index, tetrahedron) in mesh.tetrahedra.iter().enumerate() {
            for &vertex in tetrahedron {
                incident[vertex as usize].push(tetrahedron_index);
            }
        }
        let mut boundary_vertices = mesh
            .boundary_faces()
            .into_iter()
            .flatten()
            .collect::<std::collections::HashSet<_>>();
        let mut boundary_vertices = boundary_vertices.drain().collect::<Vec<_>>();
        boundary_vertices.sort_unstable();
        let minimum_volume6 = self.cell_size.powi(3) * 1.0e-7;
        let targets = boundary_vertices
            .into_iter()
            .filter_map(|index| {
                let point = mesh.positions[index as usize];
                let nearby = nearby_triangle_indices(self, point, &buckets);
                let projected = if nearby.is_empty() {
                    closest_point_on_mesh(point, source)
                } else {
                    closest_point_on_triangles(point, nearby.into_iter().map(|index| source[index]))
                };
                projected.map(|target| (index, target))
            })
            .collect::<Vec<_>>();
        for pass in 0..6 {
            let visit: Box<dyn Iterator<Item = &(u32, Point3)>> = if pass % 2 == 0 {
                Box::new(targets.iter())
            } else {
                Box::new(targets.iter().rev())
            };
            let mut largest_step = 0.0_f32;
            for &(index, target) in visit {
                let start = mesh.positions[index as usize];
                let fitted = inversion_safe_position(
                    &mesh,
                    index,
                    start,
                    target,
                    &incident[index as usize],
                    minimum_volume6,
                );
                largest_step = largest_step.max(squared_distance(start, fitted));
                mesh.positions[index as usize] = fitted;
            }
            if largest_step <= self.cell_size.powi(2) * 1.0e-10 {
                break;
            }
        }
        Ok(mesh)
    }

    fn index(&self, coordinate: [u32; 3]) -> Option<usize> {
        if coordinate
            .iter()
            .zip(self.dimensions)
            .any(|(&value, limit)| value >= limit)
        {
            return None;
        }
        Some(
            coordinate[0] as usize
                + self.dimensions[0] as usize
                    * (coordinate[1] as usize
                        + self.dimensions[1] as usize * coordinate[2] as usize),
        )
    }
}

impl TetrahedralMesh {
    fn boundary_faces(&self) -> Vec<[u32; 3]> {
        let mut faces = HashMap::<[u32; 3], ([u32; 3], usize)>::new();
        for &[a, b, c, d] in &self.tetrahedra {
            for face in [[b, c, d], [a, d, c], [a, b, d], [a, c, b]] {
                let mut key = face;
                key.sort_unstable();
                faces
                    .entry(key)
                    .and_modify(|entry| entry.1 += 1)
                    .or_insert((face, 1));
            }
        }
        faces
            .into_values()
            .filter_map(|(face, count)| (count == 1).then_some(face))
            .collect()
    }

    pub fn surface_mesh(&self) -> SurfaceMesh {
        let mut surface = SurfaceMesh::default();
        let faces = self.boundary_faces();
        let mut vertex_normals = vec![[0.0; 3]; self.positions.len()];
        for face in &faces {
            let points = face.map(|index| self.positions[index as usize]);
            let weighted_normal = cross(sub(points[1], points[0]), sub(points[2], points[0]));
            for &index in face {
                vertex_normals[index as usize] =
                    add(vertex_normals[index as usize], weighted_normal);
            }
        }
        for face in faces {
            let points = face.map(|index| self.positions[index as usize]);
            let base = surface.positions.len() as u32;
            surface.positions.extend(points);
            surface
                .normals
                .extend(face.map(|index| normalized(vertex_normals[index as usize])));
            surface.indices.extend([base, base + 1, base + 2]);
        }
        surface
    }
}

/// Conservatively voxelizes a triangle surface and fills its enclosed volume.
///
/// The mesh should be closed for meaningful interior classification. Open
/// surfaces are still rasterized, but flood fill can pass through their gaps.
pub fn voxelize(
    triangles: &[Triangle],
    options: VoxelizeOptions,
) -> Result<VoxelGrid, VoxelizeError> {
    if triangles.is_empty() {
        return Err(VoxelizeError::EmptyMesh);
    }
    if options.longest_axis_cells == 0 {
        return Err(VoxelizeError::ZeroResolution);
    }

    let mut minimum = [f32::INFINITY; 3];
    let mut maximum = [f32::NEG_INFINITY; 3];
    for point in triangles.iter().flatten() {
        if point.iter().any(|value| !value.is_finite()) {
            return Err(VoxelizeError::NonFiniteCoordinate);
        }
        for axis in 0..3 {
            minimum[axis] = minimum[axis].min(point[axis]);
            maximum[axis] = maximum[axis].max(point[axis]);
        }
    }
    let extent: Point3 = std::array::from_fn(|axis| maximum[axis] - minimum[axis]);
    let longest = extent.into_iter().fold(0.0_f32, f32::max);
    if longest <= f32::EPSILON {
        return Err(VoxelizeError::ZeroExtent);
    }
    let cell_size = longest / options.longest_axis_cells as f32;
    let padding_twice = options
        .padding
        .checked_mul(2)
        .ok_or(VoxelizeError::GridTooLarge)?;
    let dimensions =
        std::array::from_fn(|axis| (extent[axis] / cell_size).ceil() as u32 + padding_twice);
    let origin = std::array::from_fn(|axis| minimum[axis] - options.padding as f32 * cell_size);
    let cell_count = dimensions.into_iter().try_fold(1_usize, |product, value| {
        product.checked_mul(value as usize)
    });
    let mut grid = VoxelGrid {
        origin,
        cell_size,
        dimensions,
        mesh_bounds: [minimum, maximum],
        cells: vec![Cell::Interior; cell_count.ok_or(VoxelizeError::GridTooLarge)?],
    };

    rasterize_surface(&mut grid, triangles);
    flood_fill_exterior(&mut grid);
    Ok(grid)
}

fn tetrahedron_signed_volume6(points: [Point3; 4]) -> f32 {
    dot(
        sub(points[1], points[0]),
        cross(sub(points[2], points[0]), sub(points[3], points[0])),
    )
}

fn inversion_safe_position(
    mesh: &TetrahedralMesh,
    vertex: u32,
    start: Point3,
    target: Point3,
    incident: &[usize],
    minimum_volume6: f32,
) -> Point3 {
    let valid = |candidate: Point3| {
        incident.iter().all(|&tetrahedron_index| {
            let tetrahedron = mesh.tetrahedra[tetrahedron_index];
            let points = tetrahedron.map(|index| {
                if index == vertex {
                    candidate
                } else {
                    mesh.positions[index as usize]
                }
            });
            tetrahedron_signed_volume6(points) > minimum_volume6
        })
    };
    if valid(target) {
        return target;
    }

    let mut low = 0.0_f32;
    let mut high = 1.0_f32;
    for _ in 0..24 {
        let fraction = (low + high) * 0.5;
        let candidate =
            std::array::from_fn(|axis| start[axis] + (target[axis] - start[axis]) * fraction);
        if valid(candidate) {
            low = fraction;
        } else {
            high = fraction;
        }
    }
    std::array::from_fn(|axis| start[axis] + (target[axis] - start[axis]) * low)
}

fn closest_point_on_mesh(point: Point3, triangles: &[Triangle]) -> Option<Point3> {
    closest_point_on_triangles(point, triangles.iter().copied())
}

fn closest_point_on_triangles(
    point: Point3,
    triangles: impl Iterator<Item = Triangle>,
) -> Option<Point3> {
    triangles
        .map(|triangle| closest_point_on_triangle(point, triangle))
        .min_by(|left, right| {
            squared_distance(point, *left)
                .partial_cmp(&squared_distance(point, *right))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
}

fn triangle_buckets(grid: &VoxelGrid, triangles: &[Triangle]) -> HashMap<[u32; 3], Vec<usize>> {
    let mut buckets = HashMap::<[u32; 3], Vec<usize>>::new();
    let half = [grid.cell_size * 0.5; 3];
    for (triangle_index, triangle) in triangles.iter().enumerate() {
        let triangle_min: [f32; 3] = std::array::from_fn(|axis| {
            triangle
                .iter()
                .map(|point| point[axis])
                .fold(f32::INFINITY, f32::min)
        });
        let triangle_max: [f32; 3] = std::array::from_fn(|axis| {
            triangle
                .iter()
                .map(|point| point[axis])
                .fold(f32::NEG_INFINITY, f32::max)
        });
        let first: [u32; 3] = std::array::from_fn(|axis| {
            (((triangle_min[axis] - grid.origin[axis]) / grid.cell_size).floor() as i64)
                .clamp(0, grid.dimensions[axis] as i64 - 1) as u32
        });
        let last: [u32; 3] = std::array::from_fn(|axis| {
            (((triangle_max[axis] - grid.origin[axis]) / grid.cell_size).floor() as i64)
                .clamp(0, grid.dimensions[axis] as i64 - 1) as u32
        });
        for z in first[2]..=last[2] {
            for y in first[1]..=last[1] {
                for x in first[0]..=last[0] {
                    let coordinate = [x, y, z];
                    let center = grid.cell_center(coordinate).expect("bounded coordinate");
                    if triangle_box_overlap(center, half, *triangle) {
                        buckets.entry(coordinate).or_default().push(triangle_index);
                    }
                }
            }
        }
    }
    buckets
}

fn nearby_triangle_indices(
    grid: &VoxelGrid,
    point: Point3,
    buckets: &HashMap<[u32; 3], Vec<usize>>,
) -> Vec<usize> {
    let center: [i64; 3] = std::array::from_fn(|axis| {
        (((point[axis] - grid.origin[axis]) / grid.cell_size).floor() as i64)
            .clamp(0, grid.dimensions[axis] as i64 - 1)
    });
    let mut indices = Vec::new();
    for radius in 0..=2_i64 {
        for z in center[2] - radius..=center[2] + radius {
            for y in center[1] - radius..=center[1] + radius {
                for x in center[0] - radius..=center[0] + radius {
                    if [x, y, z]
                        .iter()
                        .zip(grid.dimensions)
                        .any(|(&value, limit)| value < 0 || value >= limit as i64)
                    {
                        continue;
                    }
                    if let Some(bucket) = buckets.get(&[x as u32, y as u32, z as u32]) {
                        indices.extend(bucket);
                    }
                }
            }
        }
        if !indices.is_empty() {
            break;
        }
    }
    indices.sort_unstable();
    indices.dedup();
    indices
}

// Closest-point regions from Real-Time Collision Detection, Christer Ericson.
fn closest_point_on_triangle(point: Point3, triangle: Triangle) -> Point3 {
    let [a, b, c] = triangle;
    let ab = sub(b, a);
    let ac = sub(c, a);
    let ap = sub(point, a);
    let d1 = dot(ab, ap);
    let d2 = dot(ac, ap);
    if d1 <= 0.0 && d2 <= 0.0 {
        return a;
    }
    let bp = sub(point, b);
    let d3 = dot(ab, bp);
    let d4 = dot(ac, bp);
    if d3 >= 0.0 && d4 <= d3 {
        return b;
    }
    let vc = d1 * d4 - d3 * d2;
    if vc <= 0.0 && d1 >= 0.0 && d3 <= 0.0 {
        let v = d1 / (d1 - d3);
        return add(a, scale(ab, v));
    }
    let cp = sub(point, c);
    let d5 = dot(ab, cp);
    let d6 = dot(ac, cp);
    if d6 >= 0.0 && d5 <= d6 {
        return c;
    }
    let vb = d5 * d2 - d1 * d6;
    if vb <= 0.0 && d2 >= 0.0 && d6 <= 0.0 {
        let w = d2 / (d2 - d6);
        return add(a, scale(ac, w));
    }
    let va = d3 * d6 - d5 * d4;
    if va <= 0.0 && d4 - d3 >= 0.0 && d5 - d6 >= 0.0 {
        let w = (d4 - d3) / ((d4 - d3) + (d5 - d6));
        return add(b, scale(sub(c, b), w));
    }
    let denominator = (va + vb + vc).recip();
    add(
        a,
        add(scale(ab, vb * denominator), scale(ac, vc * denominator)),
    )
}

fn squared_distance(left: Point3, right: Point3) -> f32 {
    dot(sub(left, right), sub(left, right))
}

fn add(left: Point3, right: Point3) -> Point3 {
    std::array::from_fn(|axis| left[axis] + right[axis])
}

fn scale(value: Point3, factor: f32) -> Point3 {
    value.map(|component| component * factor)
}

fn normalized(value: Point3) -> Point3 {
    let length = dot(value, value).sqrt();
    if length <= f32::EPSILON {
        [0.0, 0.0, 1.0]
    } else {
        scale(value, length.recip())
    }
}

fn rasterize_surface(grid: &mut VoxelGrid, triangles: &[Triangle]) {
    let half = [grid.cell_size * 0.5; 3];
    for triangle in triangles {
        let triangle_min: Point3 = std::array::from_fn(|axis| {
            triangle
                .iter()
                .map(|point| point[axis])
                .fold(f32::INFINITY, f32::min)
        });
        let triangle_max: Point3 = std::array::from_fn(|axis| {
            triangle
                .iter()
                .map(|point| point[axis])
                .fold(f32::NEG_INFINITY, f32::max)
        });
        let mut first = [0; 3];
        let mut last = [0; 3];
        for axis in 0..3 {
            first[axis] = (((triangle_min[axis] - grid.origin[axis]) / grid.cell_size).floor()
                as i64)
                .clamp(0, grid.dimensions[axis] as i64 - 1) as u32;
            last[axis] = (((triangle_max[axis] - grid.origin[axis]) / grid.cell_size).floor()
                as i64)
                .clamp(0, grid.dimensions[axis] as i64 - 1) as u32;
        }
        for z in first[2]..=last[2] {
            for y in first[1]..=last[1] {
                for x in first[0]..=last[0] {
                    let coordinate = [x, y, z];
                    let center = grid.cell_center(coordinate).expect("bounded coordinate");
                    if triangle_box_overlap(center, half, *triangle) {
                        let index = grid.index(coordinate).expect("bounded coordinate");
                        grid.cells[index] = Cell::Boundary;
                    }
                }
            }
        }
    }
}

fn flood_fill_exterior(grid: &mut VoxelGrid) {
    let mut queue = VecDeque::new();
    for z in 0..grid.dimensions[2] {
        for y in 0..grid.dimensions[1] {
            for x in 0..grid.dimensions[0] {
                if x != 0
                    && y != 0
                    && z != 0
                    && x + 1 != grid.dimensions[0]
                    && y + 1 != grid.dimensions[1]
                    && z + 1 != grid.dimensions[2]
                {
                    continue;
                }
                let coordinate = [x, y, z];
                let index = grid.index(coordinate).expect("bounded coordinate");
                if grid.cells[index] == Cell::Interior {
                    grid.cells[index] = Cell::Exterior;
                    queue.push_back(coordinate);
                }
            }
        }
    }

    const NEIGHBORS: [[i32; 3]; 6] = [
        [-1, 0, 0],
        [1, 0, 0],
        [0, -1, 0],
        [0, 1, 0],
        [0, 0, -1],
        [0, 0, 1],
    ];
    while let Some(current) = queue.pop_front() {
        for offset in NEIGHBORS {
            let next = std::array::from_fn(|axis| current[axis] as i64 + offset[axis] as i64);
            if next
                .iter()
                .zip(grid.dimensions)
                .any(|(&value, limit)| value < 0 || value >= limit as i64)
            {
                continue;
            }
            let next = next.map(|value| value as u32);
            let index = grid.index(next).expect("bounded neighbor");
            if grid.cells[index] == Cell::Interior {
                grid.cells[index] = Cell::Exterior;
                queue.push_back(next);
            }
        }
    }
}

// Akenine-Moller style triangle/AABB SAT test. The thirteen possible
// separating axes are the three box axes, the triangle normal, and the nine
// cross products of triangle edges with box axes.
fn triangle_box_overlap(center: Point3, half: Point3, triangle: Triangle) -> bool {
    let vertices: [Point3; 3] = triangle.map(|point| sub(point, center));
    let edges = [
        sub(vertices[1], vertices[0]),
        sub(vertices[2], vertices[1]),
        sub(vertices[0], vertices[2]),
    ];

    for axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
        if separated(axis, &vertices, half) {
            return false;
        }
    }
    let normal = cross(edges[0], edges[1]);
    if separated(normal, &vertices, half) {
        return false;
    }
    for edge in edges {
        for box_axis in [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]] {
            if separated(cross(edge, box_axis), &vertices, half) {
                return false;
            }
        }
    }
    true
}

fn separated(axis: Point3, vertices: &[Point3; 3], half: Point3) -> bool {
    if dot(axis, axis) <= f32::EPSILON {
        return false;
    }
    let projections = vertices.map(|vertex| dot(vertex, axis));
    let minimum = projections.into_iter().fold(f32::INFINITY, f32::min);
    let maximum = projections.into_iter().fold(f32::NEG_INFINITY, f32::max);
    let radius = half[0] * axis[0].abs() + half[1] * axis[1].abs() + half[2] * axis[2].abs();
    minimum > radius || maximum < -radius
}

fn sub(left: Point3, right: Point3) -> Point3 {
    std::array::from_fn(|axis| left[axis] - right[axis])
}

fn cross(left: Point3, right: Point3) -> Point3 {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: Point3, right: Point3) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rejects_invalid_inputs() {
        assert_eq!(
            voxelize(&[], VoxelizeOptions::default()),
            Err(VoxelizeError::EmptyMesh)
        );
        assert_eq!(
            voxelize(
                &[[[0.0; 3], [1.0, 0.0, 0.0], [0.0, f32::NAN, 0.0]]],
                VoxelizeOptions::default()
            ),
            Err(VoxelizeError::NonFiniteCoordinate)
        );
    }

    #[test]
    fn closed_cube_has_interior_and_exterior() {
        let triangles = cube_triangles();
        let grid = voxelize(
            &triangles,
            VoxelizeOptions {
                longest_axis_cells: 16,
                padding: 1,
            },
        )
        .unwrap();
        let counts = grid.counts();
        assert!(counts.boundary > 0);
        assert!(counts.interior > 0);
        assert!(counts.exterior > 0);
        assert_eq!(grid.sample([0.5; 3]), Some(Cell::Interior));
        assert_eq!(grid.sample([-0.04; 3]), Some(Cell::Exterior));
        let surface = grid.surface_mesh();
        assert!(!surface.positions.is_empty());
        assert_eq!(surface.positions.len(), surface.normals.len());
        assert_eq!(surface.indices.len() % 3, 0);
        let tetrahedral = grid.tetrahedral_mesh().unwrap();
        assert!(!tetrahedral.tetrahedra.is_empty());
        assert!(
            tetrahedral
                .tetrahedra
                .iter()
                .flatten()
                .all(|&index| (index as usize) < tetrahedral.positions.len())
        );
        let bounds = |axis: usize, select: fn(f32, f32) -> f32, initial: f32| {
            tetrahedral
                .positions
                .iter()
                .map(|point| point[axis])
                .fold(initial, select)
        };
        for axis in 0..3 {
            assert_eq!(bounds(axis, f32::min, f32::INFINITY), 0.0);
            assert_eq!(bounds(axis, f32::max, f32::NEG_INFINITY), 1.0);
        }
        assert!(tetrahedral.tetrahedra.iter().all(|tetrahedron| {
            tetrahedron_signed_volume6(
                tetrahedron.map(|index| tetrahedral.positions[index as usize]),
            ) > 0.0
        }));

        let fitted = grid.fitted_tetrahedral_mesh(&triangles).unwrap();
        assert_eq!(fitted.tetrahedra.len(), tetrahedral.tetrahedra.len());
        assert!(fitted.tetrahedra.iter().all(|tetrahedron| {
            tetrahedron_signed_volume6(tetrahedron.map(|index| fitted.positions[index as usize]))
                > 0.0
        }));
        let mut boundary_edges = HashMap::<[u32; 2], usize>::new();
        for [a, b, c] in fitted.boundary_faces() {
            for mut edge in [[a, b], [b, c], [c, a]] {
                edge.sort_unstable();
                *boundary_edges.entry(edge).or_default() += 1;
            }
        }
        assert!(boundary_edges.values().all(|count| *count == 2));
    }

    #[test]
    fn procedural_torus_preserves_hole_and_converges_in_volume() {
        let major_radius = 0.7_f32;
        let minor_radius = 0.25_f32;
        let triangles = torus_triangles(major_radius, minor_radius, 48, 20);
        let analytic_volume = std::f64::consts::TAU
            * std::f64::consts::PI
            * f64::from(major_radius)
            * f64::from(minor_radius).powi(2);
        let mut errors = Vec::new();
        for resolution in [32, 64] {
            let grid = voxelize(
                &triangles,
                VoxelizeOptions {
                    longest_axis_cells: resolution,
                    padding: 2,
                },
            )
            .unwrap();
            assert_eq!(grid.sample([0.0; 3]), Some(Cell::Exterior));
            assert_eq!(grid.sample([major_radius, 0.0, 0.0]), Some(Cell::Interior));
            errors.push((grid.occupied_volume() - analytic_volume).abs());
        }
        assert!(
            errors[1] < errors[0],
            "volume error should decrease with resolution: {errors:?}"
        );
    }

    fn cube_triangles() -> Vec<Triangle> {
        let p = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 1.0],
            [1.0, 1.0, 1.0],
            [0.0, 1.0, 1.0],
        ];
        [
            [0, 2, 1],
            [0, 3, 2],
            [4, 5, 6],
            [4, 6, 7],
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
        ]
        .map(|face| face.map(|index| p[index]))
        .to_vec()
    }

    fn torus_triangles(major: f32, minor: f32, rings: usize, sides: usize) -> Vec<Triangle> {
        let point = |ring: usize, side: usize| {
            let u = std::f32::consts::TAU * ring as f32 / rings as f32;
            let v = std::f32::consts::TAU * side as f32 / sides as f32;
            [
                (major + minor * v.cos()) * u.cos(),
                (major + minor * v.cos()) * u.sin(),
                minor * v.sin(),
            ]
        };
        let mut triangles = Vec::with_capacity(rings * sides * 2);
        for ring in 0..rings {
            for side in 0..sides {
                let next_ring = (ring + 1) % rings;
                let next_side = (side + 1) % sides;
                let a = point(ring, side);
                let b = point(next_ring, side);
                let c = point(next_ring, next_side);
                let d = point(ring, next_side);
                triangles.extend([[a, b, c], [a, c, d]]);
            }
        }
        triangles
    }
}
