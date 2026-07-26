//! glTF 2.0 / GLB import.
//!
//! The first implementation imports the selected/default scene as one
//! canonical mesh while preserving node transforms, UV0, vertex colors,
//! material base color, and decoded base-color images.

use crate::{
    AxisConvention, Handedness, ImportError, Material, MeshAsset, Primitive, SourceInfo,
    TextureAsset, Vertex, VertexScalarField,
};
use serde_json::Value;
use std::path::Path;

type Matrix = [[f32; 4]; 4];

pub fn import_gltf(path: &Path) -> Result<MeshAsset, ImportError> {
    let (document, buffers, images) =
        ::gltf::import(path).map_err(|error| ImportError::Gltf(error.to_string()))?;
    let mut asset = MeshAsset {
        name: path
            .file_stem()
            .and_then(|name| name.to_str())
            .unwrap_or("glTF scene")
            .to_owned(),
        source: SourceInfo {
            format: match path
                .extension()
                .and_then(|extension| extension.to_str())
                .unwrap_or_default()
                .to_ascii_lowercase()
                .as_str()
            {
                "glb" => "GLB 2.0",
                _ => "glTF 2.0",
            }
            .into(),
            path: Some(path.to_string_lossy().into_owned()),
            up_axis: AxisConvention::YUp,
            handedness: Handedness::Right,
            unit_scale_meters: Some(1.0),
        },
        ..MeshAsset::default()
    };

    for (index, image) in images.iter().enumerate() {
        let name = format!("image-{index}");
        asset
            .textures
            .insert(name.clone(), decoded_image(name, image)?);
    }

    for material in document.materials() {
        let index = material.index().unwrap_or(usize::MAX);
        let name = material
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("material-{index}"));
        let pbr = material.pbr_metallic_roughness();
        let base_color = pbr.base_color_factor();
        let emissive = material.emissive_factor();
        let base_color_texture = pbr
            .base_color_texture()
            .map(|info| format!("image-{}", info.texture().source().index()));
        asset.materials.insert(
            name.clone(),
            Material {
                name,
                base_color,
                emissive,
                specular: [pbr.metallic_factor(); 3],
                shininess: 1.0 - pbr.roughness_factor(),
                opacity: base_color[3],
                base_color_texture,
                transmission: material
                    .transmission()
                    .map(|value| value.transmission_factor()),
                ior: material.ior(),
            },
        );
    }

    let scenes: Vec<_> = if let Some(scene) = document.default_scene() {
        vec![scene]
    } else {
        document.scenes().collect()
    };
    for scene in scenes {
        for node in scene.nodes() {
            visit_node(node, identity(), &document, &buffers, &mut asset)?;
        }
    }
    for field in asset.vertex_scalar_fields.values_mut() {
        field.values.resize(asset.vertices.len(), 0.0);
    }
    if asset.vertices.is_empty() {
        return Err(ImportError::InvalidData(
            "glTF contains no triangle geometry in its imported scene".into(),
        ));
    }
    Ok(asset)
}

fn visit_node(
    node: ::gltf::Node<'_>,
    parent: Matrix,
    document: &::gltf::Document,
    buffers: &[::gltf::buffer::Data],
    asset: &mut MeshAsset,
) -> Result<(), ImportError> {
    let world = multiply(parent, node.transform().matrix());
    let vertex_start = asset.vertices.len();
    if let Some(mesh) = node.mesh() {
        for (primitive_index, primitive) in mesh.primitives().enumerate() {
            if primitive.mode() != ::gltf::mesh::Mode::Triangles {
                asset.warnings.push(format!(
                    "mesh `{}` primitive {primitive_index} is not TRIANGLES and was skipped",
                    mesh.name().unwrap_or("<unnamed>")
                ));
                continue;
            }
            let reader = primitive.reader(|buffer| Some(&buffers[buffer.index()].0));
            let positions: Vec<_> = reader
                .read_positions()
                .ok_or_else(|| {
                    ImportError::InvalidData(format!(
                        "mesh `{}` primitive {primitive_index} has no POSITION accessor",
                        mesh.name().unwrap_or("<unnamed>")
                    ))
                })?
                .collect();
            let normals = reader
                .read_normals()
                .map(|values| values.collect::<Vec<_>>());
            let uvs = reader
                .read_tex_coords(0)
                .map(|values| values.into_f32().collect::<Vec<_>>());
            let colors = reader
                .read_colors(0)
                .map(|values| values.into_rgba_f32().collect::<Vec<_>>());
            let vertex_offset = u32::try_from(asset.vertices.len()).map_err(|_| {
                ImportError::InvalidData("glTF exceeds the u32 vertex index limit".into())
            })?;
            for (index, position) in positions.iter().copied().enumerate() {
                let position = z_up(transform_point(world, position));
                let normal = normals
                    .as_ref()
                    .and_then(|values| values.get(index).copied())
                    .map(|normal| normalize(z_up(transform_normal(world, normal))))
                    .unwrap_or([0.0; 3]);
                asset.vertices.push(Vertex {
                    position,
                    normal,
                    uv: uvs
                        .as_ref()
                        .and_then(|values| values.get(index).copied())
                        .unwrap_or([0.0; 2]),
                    color: colors
                        .as_ref()
                        .and_then(|values| values.get(index).copied())
                        .unwrap_or([1.0; 4]),
                });
            }
            let indices = reader
                .read_indices()
                .map(|values| values.into_u32().collect::<Vec<_>>())
                .unwrap_or_else(|| (0..positions.len() as u32).collect());
            let indices = indices
                .into_iter()
                .map(|index| index + vertex_offset)
                .collect();
            let material = primitive.material().index().map(|index| {
                primitive
                    .material()
                    .name()
                    .map(str::to_owned)
                    .unwrap_or_else(|| format!("material-{index}"))
            });
            asset.primitives.push(Primitive {
                name: format!("{}:{primitive_index}", mesh.name().unwrap_or("glTF mesh")),
                material,
                indices,
            });
        }
    }
    if let Some(accessor_index) = zerofps_mobility_accessor(&node, asset) {
        let values = read_scalar_f32_accessor(document.accessors().nth(accessor_index), buffers)
            .ok_or_else(|| {
                ImportError::InvalidData(format!(
                    "node `{}` references invalid mobility accessor {accessor_index}",
                    node.name().unwrap_or("<unnamed>")
                ))
            })?;
        let added_vertices = asset.vertices.len() - vertex_start;
        if values.len() != added_vertices {
            return Err(ImportError::InvalidData(format!(
                "node `{}` mobility accessor has {} values for {added_vertices} imported vertices",
                node.name().unwrap_or("<unnamed>"),
                values.len()
            )));
        }
        let field = asset
            .vertex_scalar_fields
            .entry("dynamics.mobility".into())
            .or_insert_with(|| VertexScalarField {
                values: vec![0.0; vertex_start],
            });
        field.values.resize(vertex_start, 0.0);
        field.values.extend(values);
    }
    for child in node.children() {
        visit_node(child, world, document, buffers, asset)?;
    }
    Ok(())
}

fn zerofps_mobility_accessor(node: &::gltf::Node<'_>, asset: &mut MeshAsset) -> Option<usize> {
    let raw = node.extras().as_ref()?;
    let extras: Value = match serde_json::from_str(raw.get()) {
        Ok(value) => value,
        Err(error) => {
            asset.warnings.push(format!(
                "node `{}` has invalid extras JSON: {error}",
                node.name().unwrap_or("<unnamed>")
            ));
            return None;
        }
    };
    extras
        .get("zerofpsCloth")?
        .get("mobilityAccessor")?
        .as_u64()
        .and_then(|index| usize::try_from(index).ok())
}

fn read_scalar_f32_accessor(
    accessor: Option<::gltf::Accessor<'_>>,
    buffers: &[::gltf::buffer::Data],
) -> Option<Vec<f32>> {
    let accessor = accessor?;
    if accessor.dimensions() != ::gltf::accessor::Dimensions::Scalar
        || accessor.data_type() != ::gltf::accessor::DataType::F32
    {
        return None;
    }
    ::gltf::accessor::Iter::<f32>::new(accessor, |buffer| Some(&buffers[buffer.index()].0))
        .map(Iterator::collect)
}

fn decoded_image(name: String, image: &::gltf::image::Data) -> Result<TextureAsset, ImportError> {
    use ::gltf::image::Format;
    let pixel_count = image.width as usize * image.height as usize;
    let mut pixels = Vec::with_capacity(pixel_count * 4);
    match image.format {
        Format::R8 => {
            for value in &image.pixels {
                pixels.extend_from_slice(&[*value, *value, *value, 255]);
            }
        }
        Format::R8G8 => {
            for value in image.pixels.chunks_exact(2) {
                pixels.extend_from_slice(&[value[0], value[0], value[0], value[1]]);
            }
        }
        Format::R8G8B8 => {
            for value in image.pixels.chunks_exact(3) {
                pixels.extend_from_slice(&[value[0], value[1], value[2], 255]);
            }
        }
        Format::R8G8B8A8 => pixels.extend_from_slice(&image.pixels),
        other => {
            return Err(ImportError::InvalidData(format!(
                "glTF image `{name}` uses unsupported decoded format {other:?}"
            )));
        }
    }
    if pixels.len() != pixel_count * 4 {
        return Err(ImportError::InvalidData(format!(
            "glTF image `{name}` decoded to an unexpected byte count"
        )));
    }
    Ok(TextureAsset {
        name,
        width: image.width,
        height: image.height,
        pixels,
    })
}

fn identity() -> Matrix {
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.0, 1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0],
    ]
}

/// glTF matrices are column-major: result column `c`, row `r`.
fn multiply(left: Matrix, right: Matrix) -> Matrix {
    let mut result = [[0.0; 4]; 4];
    for column in 0..4 {
        for row in 0..4 {
            result[column][row] = (0..4)
                .map(|index| left[index][row] * right[column][index])
                .sum();
        }
    }
    result
}

fn transform_point(matrix: Matrix, point: [f32; 3]) -> [f32; 3] {
    [
        matrix[0][0] * point[0] + matrix[1][0] * point[1] + matrix[2][0] * point[2] + matrix[3][0],
        matrix[0][1] * point[0] + matrix[1][1] * point[1] + matrix[2][1] * point[2] + matrix[3][1],
        matrix[0][2] * point[0] + matrix[1][2] * point[1] + matrix[2][2] * point[2] + matrix[3][2],
    ]
}

/// Transform a normal by the inverse transpose of a glTF node's linear
/// transform. Using the ordinary direction matrix here is only correct for
/// rotation and uniform scale and makes lighting disagree between objects.
fn transform_normal(matrix: Matrix, normal: [f32; 3]) -> [f32; 3] {
    let x = [matrix[0][0], matrix[0][1], matrix[0][2]];
    let y = [matrix[1][0], matrix[1][1], matrix[1][2]];
    let z = [matrix[2][0], matrix[2][1], matrix[2][2]];
    let cofactor_x = cross(y, z);
    let cofactor_y = cross(z, x);
    let cofactor_z = cross(x, y);
    let determinant = dot(x, cofactor_x);
    if determinant.abs() <= f32::EPSILON {
        return [0.0; 3];
    }
    [
        (cofactor_x[0] * normal[0] + cofactor_y[0] * normal[1] + cofactor_z[0] * normal[2])
            / determinant,
        (cofactor_x[1] * normal[0] + cofactor_y[1] * normal[1] + cofactor_z[1] * normal[2])
            / determinant,
        (cofactor_x[2] * normal[0] + cofactor_y[2] * normal[1] + cofactor_z[2] * normal[2])
            / determinant,
    ]
}

fn cross(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [
        left[1] * right[2] - left[2] * right[1],
        left[2] * right[0] - left[0] * right[2],
        left[0] * right[1] - left[1] * right[0],
    ]
}

fn dot(left: [f32; 3], right: [f32; 3]) -> f32 {
    left[0] * right[0] + left[1] * right[1] + left[2] * right[2]
}

fn z_up(value: [f32; 3]) -> [f32; 3] {
    [value[0], -value[2], value[1]]
}

fn normalize(value: [f32; 3]) -> [f32; 3] {
    let length = value.iter().map(|value| value * value).sum::<f32>().sqrt();
    if length <= f32::EPSILON {
        [0.0; 3]
    } else {
        value.map(|value| value / length)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn z_up_conversion_preserves_right_handed_axes() {
        assert_eq!(z_up([1.0, 0.0, 0.0]), [1.0, -0.0, 0.0]);
        assert_eq!(z_up([0.0, 1.0, 0.0]), [0.0, -0.0, 1.0]);
        assert_eq!(z_up([0.0, 0.0, 1.0]), [0.0, -1.0, 0.0]);
    }

    #[test]
    fn matrix_hierarchy_applies_parent_before_child() {
        let mut parent = identity();
        parent[3][0] = 2.0;
        let mut child = identity();
        child[3][1] = 3.0;
        assert_eq!(
            transform_point(multiply(parent, child), [0.0; 3]),
            [2.0, 3.0, 0.0]
        );
    }

    #[test]
    fn node_normal_uses_inverse_transpose_under_non_uniform_scale() {
        let mut scale = identity();
        scale[0][0] = 2.0;
        scale[1][1] = 3.0;
        scale[2][2] = 4.0;
        let normal = transform_normal(scale, [1.0, 1.0, 0.0]);
        let transformed_tangent = [2.0, -3.0, 0.0];

        assert!(
            dot(normal, transformed_tangent).abs() < 1.0e-6,
            "an imported normal must remain perpendicular to transformed geometry"
        );
    }

    #[test]
    fn imports_zerofps_cloth_mobility_as_vertex_field() {
        let fixture =
            Path::new(env!("CARGO_MANIFEST_DIR")).join("../../appdata/models/Tripo/flag_model.glb");
        let asset = import_gltf(&fixture).expect("generated flag fixture should import");
        let mobility = asset
            .vertex_scalar_fields
            .get("dynamics.mobility")
            .expect("flag cloth mobility field");

        assert_eq!(mobility.values.len(), asset.vertices.len());
        assert!(mobility.values.iter().all(|value| value.is_finite()));
        assert!(mobility.values.iter().any(|value| *value == 0.0));
        assert!(mobility.values.iter().any(|value| *value == 1.0));
        asset
            .validate()
            .expect("imported field should be canonical");
    }
}
