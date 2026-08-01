//! Wavefront OBJ and MTL import.
//!
//! The parser deliberately implements the small, interoperable core of the
//! formats rather than exporter-specific extensions. Unknown statements are
//! ignored, while malformed statements that we do understand report their
//! source line.

use crate::{ImportError, Material, MeshAsset, Primitive, SourceInfo, Vertex, parse_f32};
use std::collections::{BTreeMap, HashMap};

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct VertexKey {
    position: usize,
    uv: Option<usize>,
    normal: Option<usize>,
    // Missing normals are flat-shaded. Keeping each polygon corner separate
    // also avoids accidentally smoothing across object boundaries.
    generated_normal_face: Option<usize>,
}

#[derive(Clone, Copy, Debug)]
struct FaceVertex {
    position: usize,
    uv: Option<usize>,
    normal: Option<usize>,
}

#[derive(Debug)]
struct Face {
    vertices: Vec<FaceVertex>,
    primitive: usize,
    line: usize,
}

/// Imports OBJ geometry without resolving material libraries.
///
/// Material names assigned with `usemtl` are preserved. Referenced `mtllib`
/// files are listed as warnings so callers can choose whether to resolve them.
pub fn import_obj(bytes: &[u8], source_name: &str) -> Result<MeshAsset, ImportError> {
    import_obj_impl(
        bytes,
        source_name,
        None::<fn(&str) -> Result<Vec<u8>, ImportError>>,
    )
}

/// Imports OBJ geometry and resolves every `mtllib` through `load_material`.
///
/// The callback receives the companion path exactly as written in the OBJ.
/// This keeps filesystem, archive, and in-memory loading policies outside the
/// codec. Return [`ImportError::MissingCompanion`] when a requested file is
/// unavailable.
pub fn import_obj_with_materials<F>(
    bytes: &[u8],
    source_name: &str,
    load_material: F,
) -> Result<MeshAsset, ImportError>
where
    F: FnMut(&str) -> Result<Vec<u8>, ImportError>,
{
    import_obj_impl(bytes, source_name, Some(load_material))
}

fn import_obj_impl<F>(
    bytes: &[u8],
    source_name: &str,
    mut load_material: Option<F>,
) -> Result<MeshAsset, ImportError>
where
    F: FnMut(&str) -> Result<Vec<u8>, ImportError>,
{
    let text = std::str::from_utf8(bytes).map_err(|error| ImportError::Parse {
        format: "OBJ",
        location: source_name.into(),
        message: format!("input is not UTF-8: {error}"),
    })?;
    let mut positions = Vec::<[f32; 3]>::new();
    let mut colors = Vec::<[f32; 4]>::new();
    let mut uvs = Vec::<[f32; 2]>::new();
    let mut normals = Vec::<[f32; 3]>::new();
    let mut primitives = vec![Primitive {
        name: "default".into(),
        material: None,
        indices: Vec::new(),
    }];
    let mut active_name = "default".to_owned();
    let mut active_material = None::<String>;
    let mut active_primitive = 0usize;
    let mut faces = Vec::<Face>::new();
    let mut libraries = Vec::<String>::new();

    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut words = line.split_whitespace();
        let statement = words.next().unwrap_or("");
        let args: Vec<_> = words.collect();
        match statement {
            "v" => {
                require_count(&args, 3, "OBJ", line_number, "vertex")?;
                positions.push([
                    number(args[0], "OBJ", line_number)?,
                    number(args[1], "OBJ", line_number)?,
                    number(args[2], "OBJ", line_number)?,
                ]);
                let color = if args.len() >= 6 {
                    [
                        number(args[3], "OBJ", line_number)?,
                        number(args[4], "OBJ", line_number)?,
                        number(args[5], "OBJ", line_number)?,
                        if args.len() >= 7 {
                            number(args[6], "OBJ", line_number)?
                        } else {
                            1.0
                        },
                    ]
                } else {
                    [1.0; 4]
                };
                colors.push(color);
            }
            "vt" => {
                require_count(&args, 2, "OBJ", line_number, "texture coordinate")?;
                uvs.push([
                    number(args[0], "OBJ", line_number)?,
                    number(args[1], "OBJ", line_number)?,
                ]);
            }
            "vn" => {
                require_count(&args, 3, "OBJ", line_number, "normal")?;
                normals.push(normalize([
                    number(args[0], "OBJ", line_number)?,
                    number(args[1], "OBJ", line_number)?,
                    number(args[2], "OBJ", line_number)?,
                ]));
            }
            "f" => {
                if args.len() < 3 {
                    return parse_error("OBJ", line_number, "face needs at least three vertices");
                }
                let vertices = args
                    .iter()
                    .map(|value| {
                        parse_face_vertex(
                            value,
                            positions.len(),
                            uvs.len(),
                            normals.len(),
                            line_number,
                        )
                    })
                    .collect::<Result<Vec<_>, _>>()?;
                faces.push(Face {
                    vertices,
                    primitive: active_primitive,
                    line: line_number,
                });
            }
            "o" | "g" => {
                active_name = if args.is_empty() {
                    "unnamed".into()
                } else {
                    args.join(" ")
                };
                active_primitive = find_or_create_primitive(
                    &mut primitives,
                    &active_name,
                    active_material.as_deref(),
                );
            }
            "usemtl" => {
                active_material = if args.is_empty() {
                    None
                } else {
                    Some(args.join(" "))
                };
                active_primitive = find_or_create_primitive(
                    &mut primitives,
                    &active_name,
                    active_material.as_deref(),
                );
            }
            "mtllib" => libraries.extend(args.into_iter().map(str::to_owned)),
            _ => {}
        }
    }

    let mut materials = BTreeMap::new();
    let mut warnings = Vec::new();
    if let Some(loader) = load_material.as_mut() {
        for library in &libraries {
            let mtl = loader(library)?;
            materials.extend(import_mtl(&mtl, library)?);
        }
    } else {
        warnings.extend(
            libraries
                .iter()
                .map(|name| format!("material library `{name}` was not loaded")),
        );
    }

    let mut vertices = Vec::<Vertex>::new();
    let mut lookup = HashMap::<VertexKey, u32>::new();
    for (face_index, face) in faces.iter().enumerate() {
        let generated = if face.vertices.iter().any(|vertex| vertex.normal.is_none()) {
            face_normal(face, &positions)?
        } else {
            [0.0; 3]
        };
        for triangle_corner in 1..face.vertices.len() - 1 {
            for corner in [0, triangle_corner, triangle_corner + 1] {
                let item = face.vertices[corner];
                let key = VertexKey {
                    position: item.position,
                    uv: item.uv,
                    normal: item.normal,
                    generated_normal_face: item.normal.is_none().then_some(face_index),
                };
                let index = if let Some(index) = lookup.get(&key) {
                    *index
                } else {
                    let index = u32::try_from(vertices.len()).map_err(|_| {
                        ImportError::InvalidData("OBJ has more than u32::MAX vertices".into())
                    })?;
                    vertices.push(Vertex {
                        position: positions[item.position],
                        normal: item.normal.map_or(generated, |index| normals[index]),
                        uv: item.uv.map_or([0.0; 2], |index| uvs[index]),
                        color: colors[item.position],
                    });
                    lookup.insert(key, index);
                    index
                };
                primitives[face.primitive].indices.push(index);
            }
        }
    }
    primitives.retain(|primitive| !primitive.indices.is_empty());
    for primitive in &primitives {
        if let Some(name) = &primitive.material
            && !materials.contains_key(name)
            && load_material.is_some()
        {
            warnings.push(format!("material `{name}` is used but was not defined"));
        }
    }
    let name = source_name
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(source_name)
        .strip_suffix(".obj")
        .unwrap_or_else(|| {
            source_name
                .rsplit(['/', '\\'])
                .next()
                .unwrap_or(source_name)
        })
        .to_owned();
    Ok(MeshAsset {
        name,
        vertices,
        primitives,
        materials,
        textures: BTreeMap::new(),
        vertex_scalar_fields: BTreeMap::new(),
        nodes: Vec::new(),
        animations: Vec::new(),
        source: SourceInfo {
            format: "obj".into(),
            path: Some(source_name.into()),
            ..SourceInfo::default()
        },
        warnings,
    })
}

/// Parses a Wavefront material library.
pub fn import_mtl(
    bytes: &[u8],
    source_name: &str,
) -> Result<BTreeMap<String, Material>, ImportError> {
    let text = std::str::from_utf8(bytes).map_err(|error| ImportError::Parse {
        format: "MTL",
        location: source_name.into(),
        message: format!("input is not UTF-8: {error}"),
    })?;
    let mut output = BTreeMap::new();
    let mut current = None::<Material>;
    for (line_index, raw_line) in text.lines().enumerate() {
        let line_number = line_index + 1;
        let line = raw_line.split('#').next().unwrap_or("").trim();
        if line.is_empty() {
            continue;
        }
        let mut words = line.split_whitespace();
        let statement = words.next().unwrap_or("");
        let args: Vec<_> = words.collect();
        if statement == "newmtl" {
            if let Some(material) = current.take() {
                output.insert(material.name.clone(), material);
            }
            if args.is_empty() {
                return parse_error("MTL", line_number, "newmtl needs a name");
            }
            current = Some(Material {
                name: args.join(" "),
                base_color: [1.0; 4],
                emissive: [0.0; 3],
                specular: [0.0; 3],
                shininess: 0.0,
                opacity: 1.0,
                base_color_texture: None,
                transmission: None,
                ior: None,
            });
            continue;
        }
        let Some(material) = current.as_mut() else {
            // Common exporters put comments and global options before newmtl.
            continue;
        };
        match statement {
            "Kd" | "Ke" | "Ks" => {
                require_count(&args, 3, "MTL", line_number, statement)?;
                let value = [
                    number(args[0], "MTL", line_number)?,
                    number(args[1], "MTL", line_number)?,
                    number(args[2], "MTL", line_number)?,
                ];
                match statement {
                    "Kd" => material.base_color[..3].copy_from_slice(&value),
                    "Ke" => material.emissive = value,
                    "Ks" => material.specular = value,
                    _ => unreachable!(),
                }
            }
            "Ns" => {
                require_count(&args, 1, "MTL", line_number, "Ns")?;
                material.shininess = number(args[0], "MTL", line_number)?;
            }
            "d" | "Tr" => {
                require_count(&args, 1, "MTL", line_number, statement)?;
                let value = number(args[0], "MTL", line_number)?;
                material.opacity = if statement == "Tr" {
                    1.0 - value
                } else {
                    value
                };
                material.base_color[3] = material.opacity;
            }
            "map_Kd" => {
                if args.is_empty() {
                    return parse_error("MTL", line_number, "map_Kd needs a texture path");
                }
                // Preserve spaces. Import options are intentionally left in the
                // string; decoding them belongs to a later, richer MTL codec.
                material.base_color_texture = Some(args.join(" "));
            }
            _ => {}
        }
    }
    if let Some(material) = current {
        output.insert(material.name.clone(), material);
    }
    Ok(output)
}

fn find_or_create_primitive(
    primitives: &mut Vec<Primitive>,
    name: &str,
    material: Option<&str>,
) -> usize {
    if let Some(index) = primitives
        .iter()
        .position(|primitive| primitive.name == name && primitive.material.as_deref() == material)
    {
        return index;
    }
    primitives.push(Primitive {
        name: name.into(),
        material: material.map(str::to_owned),
        indices: Vec::new(),
    });
    primitives.len() - 1
}

fn parse_face_vertex(
    value: &str,
    position_count: usize,
    uv_count: usize,
    normal_count: usize,
    line: usize,
) -> Result<FaceVertex, ImportError> {
    let fields: Vec<_> = value.split('/').collect();
    if fields.is_empty() || fields.len() > 3 || fields[0].is_empty() {
        return parse_error("OBJ", line, format!("invalid face vertex `{value}`"));
    }
    Ok(FaceVertex {
        position: obj_index(fields[0], position_count, line, "position")?,
        uv: fields
            .get(1)
            .filter(|field| !field.is_empty())
            .map(|field| obj_index(field, uv_count, line, "texture coordinate"))
            .transpose()?,
        normal: fields
            .get(2)
            .filter(|field| !field.is_empty())
            .map(|field| obj_index(field, normal_count, line, "normal"))
            .transpose()?,
    })
}

fn obj_index(value: &str, count: usize, line: usize, kind: &str) -> Result<usize, ImportError> {
    let parsed = value.parse::<i64>().map_err(|_| ImportError::Parse {
        format: "OBJ",
        location: format!("line {line}"),
        message: format!("invalid {kind} index `{value}`"),
    })?;
    if parsed == 0 {
        return parse_error("OBJ", line, format!("{kind} index cannot be zero"));
    }
    let resolved = if parsed > 0 {
        parsed - 1
    } else {
        count as i64 + parsed
    };
    if resolved < 0 || resolved >= count as i64 {
        return parse_error(
            "OBJ",
            line,
            format!("{kind} index `{value}` is outside the {count} values defined so far"),
        );
    }
    Ok(resolved as usize)
}

fn face_normal(face: &Face, positions: &[[f32; 3]]) -> Result<[f32; 3], ImportError> {
    let a = positions[face.vertices[0].position];
    let b = positions[face.vertices[1].position];
    let c = positions[face.vertices[2].position];
    let ab = [b[0] - a[0], b[1] - a[1], b[2] - a[2]];
    let ac = [c[0] - a[0], c[1] - a[1], c[2] - a[2]];
    let cross = [
        ab[1] * ac[2] - ab[2] * ac[1],
        ab[2] * ac[0] - ab[0] * ac[2],
        ab[0] * ac[1] - ab[1] * ac[0],
    ];
    let length_squared = cross.iter().map(|value| value * value).sum::<f32>();
    if length_squared <= f32::EPSILON {
        return parse_error(
            "OBJ",
            face.line,
            "face is degenerate and has no usable normal",
        );
    }
    Ok(normalize(cross))
}

fn normalize(value: [f32; 3]) -> [f32; 3] {
    let length = value.iter().map(|item| item * item).sum::<f32>().sqrt();
    if length > f32::EPSILON {
        [value[0] / length, value[1] / length, value[2] / length]
    } else {
        value
    }
}

fn number(value: &str, format: &'static str, line: usize) -> Result<f32, ImportError> {
    parse_f32(value, format, format!("line {line}"))
}

fn require_count(
    args: &[&str],
    count: usize,
    format: &'static str,
    line: usize,
    kind: &str,
) -> Result<(), ImportError> {
    if args.len() < count {
        parse_error(
            format,
            line,
            format!("{kind} needs at least {count} values, found {}", args.len()),
        )
    } else {
        Ok(())
    }
}

fn parse_error<T>(
    format: &'static str,
    line: usize,
    message: impl Into<String>,
) -> Result<T, ImportError> {
    Err(ImportError::Parse {
        format,
        location: format!("line {line}"),
        message: message.into(),
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_all_face_tuple_forms_and_deduplicates() {
        let obj = b"
            v 0 0 0
            v 1 0 0
            v 0 1 0
            vt 0 0
            vt 1 0
            vt 0 1
            vn 0 0 2
            f 1/1/1 2/2/1 3/3/1
            f 1//1 2//1 3//1
        ";
        let asset = import_obj(obj, "forms.obj").unwrap();
        assert_eq!(asset.triangle_count(), 2);
        assert_eq!(asset.vertices.len(), 6);
        assert_eq!(asset.vertices[0].normal, [0.0, 0.0, 1.0]);
        assert_eq!(asset.vertices[0].uv, [0.0, 0.0]);
    }

    #[test]
    fn triangulates_polygon_and_resolves_negative_indices() {
        let obj = b"
            v 0 0 0
            v 1 0 0
            v 1 1 0
            v 0 1 0
            f -4 -3 -2 -1
        ";
        let asset = import_obj(obj, "quad.obj").unwrap();
        assert_eq!(asset.triangle_count(), 2);
        assert_eq!(asset.vertices.len(), 4);
        assert!(asset.vertices.iter().all(|v| v.normal == [0.0, 0.0, 1.0]));
    }

    #[test]
    fn splits_primitives_by_group_and_material() {
        let obj = b"
            mtllib colors.mtl
            v 0 0 0
            v 1 0 0
            v 0 1 0
            g front panel
            usemtl red
            f 1 2 3
            usemtl blue
            f 1 2 3
        ";
        let asset = import_obj_with_materials(obj, "groups.obj", |name| {
            assert_eq!(name, "colors.mtl");
            Ok(b"newmtl red\nKd 1 0 0\nnewmtl blue\nKd 0 0 1".to_vec())
        })
        .unwrap();
        assert_eq!(asset.primitives.len(), 2);
        assert_eq!(asset.primitives[0].name, "front panel");
        assert_eq!(asset.primitives[0].material.as_deref(), Some("red"));
        assert_eq!(asset.primitives[1].material.as_deref(), Some("blue"));
        assert_eq!(asset.materials.len(), 2);
    }

    #[test]
    fn parses_basic_mtl_properties() {
        let materials = import_mtl(
            b"
                newmtl painted metal
                Kd .2 .3 .4
                Ke .01 .02 .03
                Ks .8 .7 .6
                Ns 96
                Tr .25
                map_Kd textures/paint color.png
            ",
            "paint.mtl",
        )
        .unwrap();
        let material = &materials["painted metal"];
        assert_eq!(material.base_color, [0.2, 0.3, 0.4, 0.75]);
        assert_eq!(material.emissive, [0.01, 0.02, 0.03]);
        assert_eq!(material.specular, [0.8, 0.7, 0.6]);
        assert_eq!(material.shininess, 96.0);
        assert_eq!(
            material.base_color_texture.as_deref(),
            Some("textures/paint color.png")
        );
    }

    #[test]
    fn reports_line_for_invalid_index() {
        let error = import_obj(b"v 0 0 0\nf 1 2 3\n", "bad.obj").unwrap_err();
        let message = error.to_string();
        assert!(message.contains("line 2"), "{message}");
        assert!(message.contains("outside"), "{message}");
    }

    #[test]
    fn preserves_vertex_colors_and_generates_normals() {
        let asset = import_obj(
            b"v 0 0 0 1 0 0\nv 1 0 0 0 1 0\nv 0 1 0 0 0 1\nf 1 2 3",
            "c.obj",
        )
        .unwrap();
        assert_eq!(asset.vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(asset.vertices[1].color, [0.0, 1.0, 0.0, 1.0]);
        assert_eq!(asset.vertices[2].normal, [0.0, 0.0, 1.0]);
    }
}
