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
    ffi::OsString,
    io,
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
    /// Validated derived levels loaded from an optional sidecar cache.
    pub cached_mips: Vec<TextureMipLevel>,
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct TextureMipLevel {
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<u8>,
}

impl TextureAsset {
    /// Builds a bounded RGBA8 mip pyramid. RGB is averaged in linear light and
    /// encoded back to sRGB; alpha remains linear.
    pub fn mip_chain(&self, maximum_levels: usize) -> Vec<Self> {
        let maximum_levels = maximum_levels.max(1);
        let expected = self.width as usize * self.height as usize * 4;
        let mut base = self.clone();
        base.cached_mips.clear();
        let mut levels = vec![base];
        if self.width == 0 || self.height == 0 || self.pixels.len() < expected {
            return levels;
        }
        let mut expected_width = self.width;
        let mut expected_height = self.height;
        for cached in &self.cached_mips {
            if levels.len() >= maximum_levels || (expected_width == 1 && expected_height == 1) {
                break;
            }
            expected_width = expected_width.div_ceil(2);
            expected_height = expected_height.div_ceil(2);
            if cached.width != expected_width
                || cached.height != expected_height
                || cached.pixels.len() != cached.width as usize * cached.height as usize * 4
            {
                break;
            }
            levels.push(Self {
                name: format!("{} mip {}", self.name, levels.len()),
                width: cached.width,
                height: cached.height,
                pixels: cached.pixels.clone(),
                cached_mips: Vec::new(),
            });
        }
        while levels.len() < maximum_levels {
            let source = levels.last().expect("mip chain always has a base level");
            if source.width == 1 && source.height == 1 {
                break;
            }
            let width = source.width.div_ceil(2);
            let height = source.height.div_ceil(2);
            let mut pixels = vec![0_u8; width as usize * height as usize * 4];
            for y in 0..height {
                for x in 0..width {
                    let mut rgb = [0.0_f32; 3];
                    let mut alpha = 0.0_f32;
                    let mut samples = 0.0_f32;
                    for source_y in y * 2..(y * 2 + 2).min(source.height) {
                        for source_x in x * 2..(x * 2 + 2).min(source.width) {
                            let source_offset =
                                (source_y as usize * source.width as usize + source_x as usize) * 4;
                            for channel in 0..3 {
                                rgb[channel] +=
                                    srgb_to_linear(source.pixels[source_offset + channel]);
                            }
                            alpha += source.pixels[source_offset + 3] as f32 / 255.0;
                            samples += 1.0;
                        }
                    }
                    let target = (y as usize * width as usize + x as usize) * 4;
                    for channel in 0..3 {
                        pixels[target + channel] = linear_to_srgb(rgb[channel] / samples);
                    }
                    pixels[target + 3] = ((alpha / samples) * 255.0).round() as u8;
                }
            }
            levels.push(Self {
                name: format!("{} mip {}", self.name, levels.len()),
                width,
                height,
                pixels,
                cached_mips: Vec::new(),
            });
        }
        levels
    }
}

fn srgb_to_linear(value: u8) -> f32 {
    let value = value as f32 / 255.0;
    if value <= 0.04045 {
        value / 12.92
    } else {
        ((value + 0.055) / 1.055).powf(2.4)
    }
}

fn linear_to_srgb(value: f32) -> u8 {
    let value = value.clamp(0.0, 1.0);
    let encoded = if value <= 0.003_130_8 {
        value * 12.92
    } else {
        1.055 * value.powf(1.0 / 2.4) - 0.055
    };
    (encoded * 255.0).round() as u8
}

const MIP_CACHE_MAGIC: &[u8; 8] = b"ZFPMIP01";

pub fn mip_cache_path(model_path: impl AsRef<Path>) -> std::path::PathBuf {
    let model_path = model_path.as_ref();
    let mut name: OsString = model_path.as_os_str().to_owned();
    name.push(".zfp-cache");
    name.into()
}

pub fn prepare_texture_mips(asset: &mut MeshAsset) {
    for texture in asset.textures.values_mut() {
        texture.cached_mips = texture
            .mip_chain(3)
            .into_iter()
            .skip(1)
            .map(|level| TextureMipLevel {
                width: level.width,
                height: level.height,
                pixels: level.pixels,
            })
            .collect();
    }
}

pub fn save_texture_mip_cache(
    model_path: impl AsRef<Path>,
    asset: &MeshAsset,
) -> io::Result<std::path::PathBuf> {
    let path = mip_cache_path(model_path);
    let mut bytes = Vec::new();
    bytes.extend_from_slice(MIP_CACHE_MAGIC);
    write_u32(&mut bytes, asset.textures.len() as u32);
    for (name, texture) in &asset.textures {
        write_bytes(&mut bytes, name.as_bytes())?;
        write_u64(&mut bytes, texture_fingerprint(texture));
        let levels = texture.mip_chain(3);
        write_u32(&mut bytes, levels.len().saturating_sub(1) as u32);
        for level in levels.iter().skip(1) {
            write_u32(&mut bytes, level.width);
            write_u32(&mut bytes, level.height);
            write_bytes(&mut bytes, &level.pixels)?;
        }
    }
    let mut temporary = path.as_os_str().to_owned();
    temporary.push(".tmp");
    let temporary = std::path::PathBuf::from(temporary);
    std::fs::write(&temporary, bytes)?;
    if let Err(error) = std::fs::rename(&temporary, &path) {
        let _ = std::fs::remove_file(&temporary);
        return Err(error);
    }
    Ok(path)
}

pub fn load_texture_mip_cache(
    model_path: impl AsRef<Path>,
    asset: &mut MeshAsset,
) -> io::Result<bool> {
    let path = mip_cache_path(model_path);
    let bytes = match std::fs::read(&path) {
        Ok(bytes) => bytes,
        Err(error) if error.kind() == io::ErrorKind::NotFound => return Ok(false),
        Err(error) => return Err(error),
    };
    let mut cursor = CacheCursor::new(&bytes);
    if cursor.take(MIP_CACHE_MAGIC.len())? != MIP_CACHE_MAGIC {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "invalid ZeroFPS mip-cache header",
        ));
    }
    let texture_count = cursor.u32()? as usize;
    if texture_count > 16_384 {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mip cache declares too many textures",
        ));
    }
    let mut cached = BTreeMap::<String, (u64, Vec<TextureMipLevel>)>::new();
    for _ in 0..texture_count {
        let name = String::from_utf8(cursor.bytes(1 << 20)?).map_err(|_| {
            io::Error::new(
                io::ErrorKind::InvalidData,
                "mip-cache texture name is not UTF-8",
            )
        })?;
        let fingerprint = cursor.u64()?;
        let level_count = cursor.u32()? as usize;
        if level_count > 2 {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mip cache has more than two derived levels",
            ));
        }
        let mut levels = Vec::with_capacity(level_count);
        for _ in 0..level_count {
            let width = cursor.u32()?;
            let height = cursor.u32()?;
            let expected = (width as usize)
                .checked_mul(height as usize)
                .and_then(|pixels| pixels.checked_mul(4))
                .filter(|bytes| *bytes <= 1 << 30)
                .ok_or_else(|| {
                    io::Error::new(
                        io::ErrorKind::InvalidData,
                        "mip-cache level dimensions are too large",
                    )
                })?;
            let pixels = cursor.bytes(expected)?;
            if pixels.len() != expected {
                return Err(io::Error::new(
                    io::ErrorKind::InvalidData,
                    "mip-cache level has an invalid byte count",
                ));
            }
            levels.push(TextureMipLevel {
                width,
                height,
                pixels,
            });
        }
        if cached.insert(name, (fingerprint, levels)).is_some() {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mip cache contains a duplicate texture name",
            ));
        }
    }
    if !cursor.is_empty() {
        return Err(io::Error::new(
            io::ErrorKind::InvalidData,
            "mip cache contains trailing data",
        ));
    }
    let mut applied = 0;
    for (name, texture) in &mut asset.textures {
        let Some((fingerprint, levels)) = cached.get(name) else {
            continue;
        };
        if *fingerprint != texture_fingerprint(texture) {
            continue;
        }
        if cached_mips_valid(texture, levels) {
            texture.cached_mips = levels.clone();
            applied += 1;
        }
    }
    Ok(applied == asset.textures.len())
}

fn cached_mips_valid(texture: &TextureAsset, levels: &[TextureMipLevel]) -> bool {
    let mut width = texture.width;
    let mut height = texture.height;
    let expected_levels = if width == 1 && height == 1 {
        0
    } else if width <= 2 && height <= 2 {
        1
    } else {
        2
    };
    if levels.len() != expected_levels {
        return false;
    }
    levels.iter().all(|level| {
        width = width.div_ceil(2);
        height = height.div_ceil(2);
        level.width == width
            && level.height == height
            && level.pixels.len() == width as usize * height as usize * 4
    })
}

fn texture_fingerprint(texture: &TextureAsset) -> u64 {
    let mut hash = 0xcbf2_9ce4_8422_2325_u64;
    for byte in texture
        .width
        .to_le_bytes()
        .into_iter()
        .chain(texture.height.to_le_bytes())
        .chain(texture.pixels.iter().copied())
    {
        hash ^= byte as u64;
        hash = hash.wrapping_mul(0x100_0000_01b3);
    }
    hash
}

fn write_u32(target: &mut Vec<u8>, value: u32) {
    target.extend_from_slice(&value.to_le_bytes());
}

fn write_u64(target: &mut Vec<u8>, value: u64) {
    target.extend_from_slice(&value.to_le_bytes());
}

fn write_bytes(target: &mut Vec<u8>, value: &[u8]) -> io::Result<()> {
    let length = u32::try_from(value.len())
        .map_err(|_| io::Error::new(io::ErrorKind::InvalidInput, "cache field exceeds 4 GiB"))?;
    write_u32(target, length);
    target.extend_from_slice(value);
    Ok(())
}

struct CacheCursor<'a> {
    remaining: &'a [u8],
}

impl<'a> CacheCursor<'a> {
    fn new(bytes: &'a [u8]) -> Self {
        Self { remaining: bytes }
    }

    fn take(&mut self, length: usize) -> io::Result<&'a [u8]> {
        if length > self.remaining.len() {
            return Err(io::Error::new(
                io::ErrorKind::UnexpectedEof,
                "truncated ZeroFPS mip cache",
            ));
        }
        let (value, remaining) = self.remaining.split_at(length);
        self.remaining = remaining;
        Ok(value)
    }

    fn u32(&mut self) -> io::Result<u32> {
        Ok(u32::from_le_bytes(
            self.take(4)?.try_into().expect("four bytes"),
        ))
    }

    fn u64(&mut self) -> io::Result<u64> {
        Ok(u64::from_le_bytes(
            self.take(8)?.try_into().expect("eight bytes"),
        ))
    }

    fn bytes(&mut self, maximum: usize) -> io::Result<Vec<u8>> {
        let length = self.u32()? as usize;
        if length > maximum {
            return Err(io::Error::new(
                io::ErrorKind::InvalidData,
                "mip-cache field exceeds its allowed size",
            ));
        }
        Ok(self.take(length)?.to_vec())
    }

    fn is_empty(&self) -> bool {
        self.remaining.is_empty()
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct Primitive {
    pub name: String,
    pub material: Option<String>,
    pub indices: Vec<u32>,
}

/// An auxiliary scalar sampled once per canonical mesh vertex.
///
/// Importers use these fields for source data that has no built-in [`Vertex`]
/// attribute, such as simulation mobility, mass, or stiffness. Values always
/// follow `MeshAsset::vertices` ordering.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct VertexScalarField {
    pub values: Vec<f32>,
}

/// One object in an imported scene hierarchy.
///
/// Geometry referenced by `primitives` remains in this node's local space.
/// Consumers compose `local` through `parent` rather than baking transforms
/// into vertices.
#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssetNode {
    pub name: String,
    pub parent: Option<usize>,
    pub children: Vec<usize>,
    pub local: AssetTransform,
    pub primitives: Vec<usize>,
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct AssetTransform {
    pub translation: [f32; 3],
    /// Unit quaternion in `(w, x, y, z)` order.
    pub rotation: [f32; 4],
    pub scale: [f32; 3],
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct AssetAnimationClip {
    pub name: String,
    pub channels: Vec<AssetAnimationChannel>,
}

#[derive(Clone, Debug, PartialEq)]
pub enum AssetAnimationChannel {
    Translation {
        node: usize,
        times: Vec<f32>,
        values: Vec<[f32; 3]>,
    },
    Rotation {
        node: usize,
        times: Vec<f32>,
        values: Vec<[f32; 4]>,
    },
    Scale {
        node: usize,
        times: Vec<f32>,
        values: Vec<[f32; 3]>,
    },
}

impl Default for AssetTransform {
    fn default() -> Self {
        Self {
            translation: [0.0; 3],
            rotation: [1.0, 0.0, 0.0, 0.0],
            scale: [1.0; 3],
        }
    }
}

#[derive(Clone, Debug, Default, PartialEq)]
pub struct MeshAsset {
    pub name: String,
    pub vertices: Vec<Vertex>,
    pub primitives: Vec<Primitive>,
    pub materials: BTreeMap<String, Material>,
    pub textures: BTreeMap<String, TextureAsset>,
    pub vertex_scalar_fields: BTreeMap<String, VertexScalarField>,
    pub nodes: Vec<AssetNode>,
    pub animations: Vec<AssetAnimationClip>,
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
        for (node_index, node) in self.nodes.iter().enumerate() {
            if node
                .local
                .translation
                .iter()
                .chain(node.local.rotation.iter())
                .chain(node.local.scale.iter())
                .any(|value| !value.is_finite())
            {
                return Err(ImportError::InvalidData(format!(
                    "asset node `{}` contains a non-finite transform",
                    node.name
                )));
            }
            if let Some(parent) = node.parent {
                let parent_node = self.nodes.get(parent).ok_or_else(|| {
                    ImportError::InvalidData(format!(
                        "asset node `{}` references missing parent {parent}",
                        node.name
                    ))
                })?;
                if parent == node_index || !parent_node.children.contains(&node_index) {
                    return Err(ImportError::InvalidData(format!(
                        "asset node `{}` has an inconsistent parent link",
                        node.name
                    )));
                }
            }
            for child in &node.children {
                if self.nodes.get(*child).and_then(|child| child.parent) != Some(node_index) {
                    return Err(ImportError::InvalidData(format!(
                        "asset node `{}` has an inconsistent child link to {child}",
                        node.name
                    )));
                }
            }
            for primitive in &node.primitives {
                if *primitive >= self.primitives.len() {
                    return Err(ImportError::InvalidData(format!(
                        "asset node `{}` references missing primitive {primitive}",
                        node.name
                    )));
                }
            }
            let mut ancestor = node.parent;
            for _ in 0..self.nodes.len() {
                ancestor = ancestor.and_then(|index| self.nodes.get(index)?.parent);
            }
            if ancestor.is_some() {
                return Err(ImportError::InvalidData(format!(
                    "asset node `{}` belongs to a parent cycle",
                    node.name
                )));
            }
        }
        for (name, field) in &self.vertex_scalar_fields {
            if field.values.len() != self.vertices.len() {
                return Err(ImportError::InvalidData(format!(
                    "vertex scalar field `{name}` has {} values for {} vertices",
                    field.values.len(),
                    self.vertices.len()
                )));
            }
            if field.values.iter().any(|value| !value.is_finite()) {
                return Err(ImportError::InvalidData(format!(
                    "vertex scalar field `{name}` contains a non-finite value"
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
    orient_strongly_inverted_winding(&mut mesh);
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

/// Some public mesh files (including the Bunny fixture) contain a consistently
/// reversed winding and no normals. Only flip when the evidence is strong so
/// concave shapes and ordinary local winding defects are left untouched.
fn orient_strongly_inverted_winding(mesh: &mut MeshAsset) -> bool {
    if mesh.vertices.is_empty() {
        return false;
    }
    let mut center = [0.0_f32; 3];
    for vertex in &mesh.vertices {
        for axis in 0..3 {
            center[axis] += vertex.position[axis] / mesh.vertices.len() as f32;
        }
    }
    let mut inward = 0usize;
    let mut usable = 0usize;
    for triangle in mesh
        .primitives
        .iter()
        .flat_map(|primitive| primitive.indices.chunks_exact(3))
    {
        let positions = [
            mesh.vertices[triangle[0] as usize].position,
            mesh.vertices[triangle[1] as usize].position,
            mesh.vertices[triangle[2] as usize].position,
        ];
        let ab = subtract3(positions[1], positions[0]);
        let ac = subtract3(positions[2], positions[0]);
        let face = cross3(ab, ac);
        // EPSILON would reject valid faces in small models such as Bunny.
        if dot3(face, face) <= f32::MIN_POSITIVE {
            continue;
        }
        let face_center = [0, 1, 2]
            .map(|axis| (positions[0][axis] + positions[1][axis] + positions[2][axis]) / 3.0);
        inward += usize::from(dot3(face, subtract3(face_center, center)) < 0.0);
        usable += 1;
    }
    if usable == 0 || inward * 4 <= usable * 3 {
        return false;
    }
    for primitive in &mut mesh.primitives {
        for triangle in primitive.indices.chunks_exact_mut(3) {
            triangle.swap(1, 2);
        }
    }
    true
}

fn subtract3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
}

fn cross3(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot3(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
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

    #[test]
    fn texture_mips_are_bounded_handle_odd_sizes_and_average_in_linear_light() {
        let texture = TextureAsset {
            name: "checker".into(),
            width: 3,
            height: 2,
            pixels: vec![
                0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255, 255, 255, 255, 255, 0, 0, 0, 255,
                255, 255, 255, 255,
            ],
            cached_mips: Vec::new(),
        };
        let levels = texture.mip_chain(3);
        assert_eq!(
            levels
                .iter()
                .map(|level| (level.width, level.height))
                .collect::<Vec<_>>(),
            vec![(3, 2), (2, 1), (1, 1)]
        );
        assert_eq!(levels[1].pixels.len(), 8);
        assert!(
            levels[1].pixels[0] > 180,
            "linear-light black/white averaging should encode brighter than byte averaging"
        );
        assert_eq!(levels[1].pixels[3], 255);
        assert_eq!(texture.mip_chain(1), vec![texture]);
    }

    #[test]
    fn texture_mip_sidecar_round_trips_and_rejects_stale_or_corrupt_data() {
        let unique = format!(
            "zerofps-mip-cache-{}-{}",
            std::process::id(),
            std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .unwrap()
                .as_nanos()
        );
        let directory = std::env::temp_dir().join(unique);
        std::fs::create_dir_all(&directory).unwrap();
        let model = directory.join("model.glb");
        let source_texture = TextureAsset {
            name: "base".into(),
            width: 4,
            height: 4,
            pixels: (0..4 * 4 * 4).map(|value| value as u8).collect(),
            cached_mips: Vec::new(),
        };
        let mut prepared = MeshAsset::default();
        prepared
            .textures
            .insert("base".into(), source_texture.clone());
        prepare_texture_mips(&mut prepared);
        assert_eq!(prepared.textures["base"].cached_mips.len(), 2);
        let cache_path = save_texture_mip_cache(&model, &prepared).unwrap();
        assert_eq!(cache_path, mip_cache_path(&model));

        let mut loaded = MeshAsset::default();
        loaded
            .textures
            .insert("base".into(), source_texture.clone());
        assert!(load_texture_mip_cache(&model, &mut loaded).unwrap());
        assert_eq!(
            loaded.textures["base"].mip_chain(3),
            prepared.textures["base"].mip_chain(3)
        );

        let mut stale = MeshAsset::default();
        let mut changed = source_texture;
        changed.pixels[0] ^= 0xff;
        stale.textures.insert("base".into(), changed);
        assert!(!load_texture_mip_cache(&model, &mut stale).unwrap());
        assert!(stale.textures["base"].cached_mips.is_empty());

        std::fs::write(&cache_path, b"broken").unwrap();
        assert_eq!(
            load_texture_mip_cache(&model, &mut stale)
                .unwrap_err()
                .kind(),
            io::ErrorKind::UnexpectedEof
        );
        std::fs::remove_dir_all(directory).unwrap();
    }
}
