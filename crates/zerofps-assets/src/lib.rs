//! Interoperable mesh import for ZeroFPS.
//!
//! Codecs translate external files into this canonical representation. Runtime
//! and scene code never depend on source-format details.

pub mod gltf;
pub mod obj;
pub mod ply;
pub mod stl;

use std::{
    collections::{BTreeMap, BTreeSet},
    path::Path,
};
use thiserror::Error;

#[derive(Clone, Copy, Debug, Default, PartialEq)]
pub struct Vertex {
    pub position: [f32; 3],
    pub normal: [f32; 3],
    pub uv: [f32; 2],
    pub color: [f32; 4],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Material {
    pub name: String,
    pub base_color: [f32; 4],
    pub emissive: [f32; 3],
    pub specular: [f32; 3],
    pub shininess: f32,
    pub opacity: f32,
    pub base_color_texture: Option<String>,
    pub transmission: Option<f32>,
    pub ior: Option<f32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextureAsset {
    pub name: String,
    pub width: u32,
    pub height: u32,
    /// Row-major RGBA8 pixels.
    pub pixels: Vec<u8>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Primitive {
    pub name: String,
    pub material: Option<String>,
    pub indices: Vec<u32>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshAsset {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub primitives: Vec<Primitive>,
    pub materials: BTreeMap<String, Material>,
    pub textures: BTreeMap<String, TextureAsset>,
    pub source: SourceInfo,
    pub warnings: Vec<String>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct SourceInfo {
    pub format: String,
    pub path: Option<String>,
    pub up_axis: AxisConvention,
    pub handedness: Handedness,
    /// Meters represented by one source unit, when the format declares it.
    pub unit_scale_meters: Option<f32>,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum AxisConvention {
    XUp,
    YUp,
    ZUp,
    #[default]
    Unspecified,
}

impl AxisConvention {
    pub fn label(self) -> &'static str {
        match self {
            Self::XUp => "X-up",
            Self::YUp => "Y-up",
            Self::ZUp => "Z-up",
            Self::Unspecified => "axis unspecified",
        }
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum Handedness {
    Left,
    Right,
    #[default]
    Unspecified,
}

impl MeshAsset {
    pub fn triangle_count(&self) -> usize {
        self.primitives.iter().map(|p| p.indices.len() / 3).sum()
    }

    pub fn validate(&self) -> Result<(), ImportError> {
        if self.vertices.is_empty() {
            return Err(ImportError::InvalidData("mesh has no vertices".into()));
        }
        for (index, vertex) in self.vertices.iter().enumerate() {
            if vertex
                .position
                .iter()
                .chain(vertex.normal.iter())
                .chain(vertex.uv.iter())
                .chain(vertex.color.iter())
                .any(|value| !value.is_finite())
            {
                return Err(ImportError::InvalidData(format!(
                    "vertex {index} contains a non-finite value"
                )));
            }
        }
        for primitive in &self.primitives {
            if primitive.indices.len() % 3 != 0 {
                return Err(ImportError::InvalidData(format!(
                    "primitive `{}` index count is not divisible by three",
                    primitive.name
                )));
            }
            if let Some(index) = primitive
                .indices
                .iter()
                .find(|index| **index as usize >= self.vertices.len())
            {
                return Err(ImportError::InvalidData(format!(
                    "primitive `{}` references vertex {index} outside {} vertices",
                    primitive.name,
                    self.vertices.len()
                )));
            }
        }
        Ok(())
    }
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MeshAutofixReport {
    pub boundary_loops: usize,
    pub filled_loops: usize,
    pub added_triangles: usize,
}

/// Produce a non-destructive repaired derivative of an imported mesh.
///
/// Simple manifold boundary loops are projected to their dominant plane and
/// triangulated with ear clipping. Original vertices and triangles are kept
/// unchanged, then area-weighted vertex normals are regenerated.
pub fn autofix_mesh(source: &MeshAsset) -> (MeshAsset, MeshAutofixReport) {
    const MAX_HOLE_EDGES: usize = 64;
    const MAX_HOLE_PERIMETER_TO_DIAGONAL: f32 = 0.30;
    let mut mesh = source.clone();
    let mut edge_counts: BTreeMap<(u32, u32), usize> = BTreeMap::new();
    for triangle in mesh
        .primitives
        .iter()
        .flat_map(|primitive| primitive.indices.chunks_exact(3))
    {
        for (a, b) in [
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ] {
            *edge_counts.entry((a.min(b), a.max(b))).or_default() += 1;
        }
    }
    let mut adjacency: BTreeMap<u32, Vec<u32>> = BTreeMap::new();
    for (&(a, b), &count) in &edge_counts {
        if count == 1 {
            adjacency.entry(a).or_default().push(b);
            adjacency.entry(b).or_default().push(a);
        }
    }
    let mut unused: BTreeSet<(u32, u32)> = edge_counts
        .iter()
        .filter_map(|(&edge, &count)| (count == 1).then_some(edge))
        .collect();
    let mut loops = Vec::new();
    while let Some(&(start, next)) = unused.first() {
        let mut boundary = vec![start];
        let mut previous = start;
        let mut current = next;
        unused.remove(&(start.min(next), start.max(next)));
        while current != start {
            boundary.push(current);
            let Some(neighbors) = adjacency.get(&current) else {
                boundary.clear();
                break;
            };
            if neighbors.len() != 2 {
                boundary.clear();
                break;
            }
            let following = if neighbors[0] == previous {
                neighbors[1]
            } else {
                neighbors[0]
            };
            if !unused.remove(&(current.min(following), current.max(following))) {
                boundary.clear();
                break;
            }
            previous = current;
            current = following;
        }
        if boundary.len() >= 3 {
            loops.push(boundary);
        }
    }
    let mut repair_indices = Vec::new();
    let boundary_loops = loops.len();
    let mut filled_loops = 0;
    let bounds = mesh.vertices.iter().fold(
        ([f32::INFINITY; 3], [f32::NEG_INFINITY; 3]),
        |(mut minimum, mut maximum), vertex| {
            for axis in 0..3 {
                minimum[axis] = minimum[axis].min(vertex.position[axis]);
                maximum[axis] = maximum[axis].max(vertex.position[axis]);
            }
            (minimum, maximum)
        },
    );
    let diagonal = (0..3)
        .map(|axis| (bounds.1[axis] - bounds.0[axis]).powi(2))
        .sum::<f32>()
        .sqrt();
    for boundary in loops {
        let perimeter = boundary
            .iter()
            .zip(boundary.iter().cycle().skip(1))
            .map(|(a, b)| {
                let start = mesh.vertices[*a as usize].position;
                let end = mesh.vertices[*b as usize].position;
                (0..3)
                    .map(|axis| (end[axis] - start[axis]).powi(2))
                    .sum::<f32>()
                    .sqrt()
            })
            .sum::<f32>();
        if boundary.len() > MAX_HOLE_EDGES
            || diagonal <= f32::EPSILON
            || perimeter > diagonal * MAX_HOLE_PERIMETER_TO_DIAGONAL
        {
            continue;
        }
        if let Some(triangles) = triangulate_boundary(&mesh.vertices, &boundary) {
            repair_indices.extend(triangles);
            filled_loops += 1;
        }
    }
    let added_triangles = repair_indices.len() / 3;
    if !repair_indices.is_empty() {
        mesh.primitives.push(Primitive {
            name: "mesh-autofix".into(),
            material: None,
            indices: repair_indices,
        });
    }
    regenerate_vertex_normals(&mut mesh);
    (
        mesh,
        MeshAutofixReport {
            boundary_loops,
            filled_loops,
            added_triangles,
        },
    )
}

fn triangulate_boundary(vertices: &[Vertex], boundary: &[u32]) -> Option<Vec<u32>> {
    let points: Vec<[f32; 3]> = boundary
        .iter()
        .map(|index| vertices[*index as usize].position)
        .collect();
    let mut normal = [0.0_f32; 3];
    for (current, next) in points.iter().zip(points.iter().cycle().skip(1)) {
        normal[0] += (current[1] - next[1]) * (current[2] + next[2]);
        normal[1] += (current[2] - next[2]) * (current[0] + next[0]);
        normal[2] += (current[0] - next[0]) * (current[1] + next[1]);
    }
    let dropped = normal
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.abs().total_cmp(&right.abs()))?
        .0;
    let projected: Vec<[f32; 2]> = points
        .iter()
        .map(|point| match dropped {
            0 => [point[1], point[2]],
            1 => [point[0], point[2]],
            _ => [point[0], point[1]],
        })
        .collect();
    let signed_area: f32 = projected
        .iter()
        .zip(projected.iter().cycle().skip(1))
        .map(|(a, b)| a[0] * b[1] - b[0] * a[1])
        .sum();
    if signed_area.abs() <= f32::EPSILON {
        return None;
    }
    let orientation = signed_area.signum();
    let cross = |a: [f32; 2], b: [f32; 2], c: [f32; 2]| {
        (b[0] - a[0]) * (c[1] - a[1]) - (b[1] - a[1]) * (c[0] - a[0])
    };
    let mut polygon: Vec<usize> = (0..boundary.len()).collect();
    let mut output = Vec::with_capacity((boundary.len() - 2) * 3);
    while polygon.len() > 3 {
        let mut ear = None;
        for index in 0..polygon.len() {
            let previous = polygon[(index + polygon.len() - 1) % polygon.len()];
            let current = polygon[index];
            let next = polygon[(index + 1) % polygon.len()];
            if cross(projected[previous], projected[current], projected[next]) * orientation
                <= 1.0e-12
            {
                continue;
            }
            let contains_vertex = polygon.iter().copied().any(|candidate| {
                if matches!(candidate, value if value == previous || value == current || value == next)
                {
                    return false;
                }
                let point = projected[candidate];
                let c0 = cross(projected[previous], projected[current], point) * orientation;
                let c1 = cross(projected[current], projected[next], point) * orientation;
                let c2 = cross(projected[next], projected[previous], point) * orientation;
                c0 >= 0.0 && c1 >= 0.0 && c2 >= 0.0
            });
            if !contains_vertex {
                ear = Some((index, [previous, current, next]));
                break;
            }
        }
        let (index, triangle) = ear?;
        output.extend(triangle.map(|corner| boundary[corner]));
        polygon.remove(index);
    }
    output.extend(polygon.into_iter().map(|corner| boundary[corner]));
    Some(output)
}

fn regenerate_vertex_normals(mesh: &mut MeshAsset) {
    let mut normals = vec![[0.0_f32; 3]; mesh.vertices.len()];
    for triangle in mesh
        .primitives
        .iter()
        .flat_map(|primitive| primitive.indices.chunks_exact(3))
    {
        let [a, b, c] = [
            triangle[0] as usize,
            triangle[1] as usize,
            triangle[2] as usize,
        ];
        let pa = mesh.vertices[a].position;
        let pb = mesh.vertices[b].position;
        let pc = mesh.vertices[c].position;
        let ab = [pb[0] - pa[0], pb[1] - pa[1], pb[2] - pa[2]];
        let ac = [pc[0] - pa[0], pc[1] - pa[1], pc[2] - pa[2]];
        let face = [
            ab[1] * ac[2] - ab[2] * ac[1],
            ab[2] * ac[0] - ab[0] * ac[2],
            ab[0] * ac[1] - ab[1] * ac[0],
        ];
        for index in [a, b, c] {
            for axis in 0..3 {
                normals[index][axis] += face[axis];
            }
        }
    }
    for (vertex, normal) in mesh.vertices.iter_mut().zip(normals) {
        let length = normal.iter().map(|value| value * value).sum::<f32>().sqrt();
        if length > f32::EPSILON {
            vertex.normal = normal.map(|value| value / length);
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshFormat {
    Obj,
    Ply,
    Stl,
    Gltf,
    Glb,
}

impl MeshFormat {
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "obj" => Some(Self::Obj),
            "ply" => Some(Self::Ply),
            "stl" => Some(Self::Stl),
            "gltf" => Some(Self::Gltf),
            "glb" => Some(Self::Glb),
            _ => None,
        }
    }
}

/// Imports a self-contained byte stream. OBJ material libraries are imported
/// separately with `obj::import_obj_with_materials`.
pub fn import_bytes(
    format: MeshFormat,
    bytes: &[u8],
    source_name: &str,
) -> Result<MeshAsset, ImportError> {
    let mut asset = match format {
        MeshFormat::Obj => obj::import_obj(bytes, source_name)?,
        MeshFormat::Ply => ply::import_ply(bytes, source_name)?,
        MeshFormat::Stl => stl::import_stl(bytes, source_name)?,
        MeshFormat::Gltf | MeshFormat::Glb => {
            return Err(ImportError::InvalidData(
                "glTF byte import requires a resource resolver; use import_file".into(),
            ));
        }
    };
    asset.source.path = Some(source_name.into());
    asset.validate()?;
    Ok(asset)
}

pub fn import_file(path: impl AsRef<Path>) -> Result<MeshAsset, ImportError> {
    let path = path.as_ref();
    let format = MeshFormat::from_path(path).ok_or_else(|| {
        ImportError::UnsupportedFormat(
            path.extension()
                .and_then(|value| value.to_str())
                .unwrap_or("<none>")
                .into(),
        )
    })?;
    if matches!(format, MeshFormat::Gltf | MeshFormat::Glb) {
        let mut asset = gltf::import_gltf(path)?;
        asset.source.path = Some(path.to_string_lossy().into_owned());
        asset.validate()?;
        return Ok(asset);
    }
    let bytes = std::fs::read(path)?;
    let source_name = path.to_string_lossy();
    let mut asset = if format == MeshFormat::Obj {
        let base = path.parent().unwrap_or_else(|| Path::new("."));
        obj::import_obj_with_materials(&bytes, &source_name, |companion| {
            Ok(std::fs::read(base.join(companion))?)
        })?
    } else {
        import_bytes(format, &bytes, &source_name)?
    };
    asset.source.path = Some(source_name.into_owned());
    asset.validate()?;
    Ok(asset)
}

#[derive(Debug, Error)]
pub enum ImportError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("unsupported mesh format `{0}`")]
    UnsupportedFormat(String),
    #[error("{format} parse error at {location}: {message}")]
    Parse {
        format: &'static str,
        location: String,
        message: String,
    },
    #[error("invalid mesh data: {0}")]
    InvalidData(String),
    #[error("referenced companion file `{0}` was not supplied")]
    MissingCompanion(String),
    #[error("glTF import error: {0}")]
    Gltf(String),
}

pub(crate) fn parse_f32(
    value: &str,
    format: &'static str,
    location: impl Into<String>,
) -> Result<f32, ImportError> {
    value
        .parse::<f32>()
        .map_err(|_| ImportError::Parse {
            format,
            location: location.into(),
            message: format!("expected finite number, found `{value}`"),
        })
        .and_then(|number| {
            if number.is_finite() {
                Ok(number)
            } else {
                Err(ImportError::InvalidData(format!(
                    "{format} contains non-finite number `{value}`"
                )))
            }
        })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn registry_recognizes_extensions_case_insensitively() {
        assert_eq!(
            MeshFormat::from_path(Path::new("model.OBJ")),
            Some(MeshFormat::Obj)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("scan.Ply")),
            Some(MeshFormat::Ply)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("part.stl")),
            Some(MeshFormat::Stl)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("scene.gltf")),
            Some(MeshFormat::Gltf)
        );
        assert_eq!(
            MeshFormat::from_path(Path::new("packed.GLB")),
            Some(MeshFormat::Glb)
        );
        assert_eq!(MeshFormat::from_path(Path::new("model.fbx")), None);
    }

    #[test]
    fn file_import_resolves_obj_material_companion() {
        let unique = format!(
            "zerofps-assets-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&directory).unwrap();
        let obj_path = directory.join("mesh.obj");
        std::fs::write(
            &obj_path,
            b"mtllib mesh.mtl\nv 0 0 0\nv 1 0 0\nv 0 1 0\nusemtl red\nf 1 2 3\n",
        )
        .unwrap();
        std::fs::write(directory.join("mesh.mtl"), b"newmtl red\nKd 1 0 0\n").unwrap();

        let asset = import_file(&obj_path).unwrap();
        assert_eq!(asset.triangle_count(), 1);
        assert_eq!(asset.materials["red"].base_color, [1.0, 0.0, 0.0, 1.0]);

        std::fs::remove_dir_all(directory).unwrap();
    }

    #[test]
    fn autofix_fills_only_object_relative_small_boundary_loops() {
        let vertices = [
            [0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
            [1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.5, 0.5, -1.0],
            [100.0, 0.0, 0.0],
        ]
        .map(|position| Vertex {
            position,
            color: [1.0; 4],
            ..Vertex::default()
        });
        let source = MeshAsset {
            name: "open-pyramid".into(),
            vertices: vertices.into(),
            primitives: vec![Primitive {
                name: "sides".into(),
                material: None,
                indices: vec![0, 1, 4, 1, 2, 4, 2, 3, 4, 3, 0, 4],
            }],
            ..MeshAsset::default()
        };
        let (fixed, report) = autofix_mesh(&source);
        assert_eq!(source.triangle_count(), 4);
        assert_eq!(report.boundary_loops, 1);
        assert_eq!(report.filled_loops, 1);
        assert_eq!(report.added_triangles, 2);
        assert_eq!(fixed.triangle_count(), 6);
        assert!(fixed.vertices[..5].iter().all(|vertex| {
            let length = vertex.normal.iter().map(|value| value * value).sum::<f32>();
            (length - 1.0).abs() < 1.0e-5
        }));

        let mut large = source;
        large.vertices.pop();
        let (_, large_report) = autofix_mesh(&large);
        assert_eq!(large_report.boundary_loops, 1);
        assert_eq!(large_report.filled_loops, 0);
    }
}
