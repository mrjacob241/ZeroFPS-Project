//! Reusable conservative triangle-mesh voxelization.
//!
//! The crate deliberately accepts plain triangles and has no dependency on a
//! renderer, asset format, or FEM implementation. The current backend marks
//! every grid cell touched by a triangle and classifies the remaining cells by
//! a six-connected exterior flood fill.

use std::collections::VecDeque;

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
        cells: vec![Cell::Interior; cell_count.ok_or(VoxelizeError::GridTooLarge)?],
    };

    rasterize_surface(&mut grid, triangles);
    flood_fill_exterior(&mut grid);
    Ok(grid)
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
