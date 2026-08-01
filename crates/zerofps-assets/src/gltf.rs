//! glTF 2.0 / GLB import.
//!
//! The selected/default scene is imported as canonical primitive data plus its
//! original node hierarchy. Vertex data remains node-local so the engine's
//! geometry tree can compose parent and child transforms.

use crate::{
    AssetAnimationChannel, AssetAnimationClip, AssetNode, AssetTransform, AxisConvention,
    Handedness, ImportError, Material, MeshAsset, Primitive, SourceInfo, TextureAsset, Vertex,
    VertexScalarField,
};
use serde_json::Value;
use std::path::Path;

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
            up_axis: AxisConvention::ZUp,
            handedness: Handedness::Right,
            unit_scale_meters: Some(1.0),
        },
        ..MeshAsset::default()
    };

    for extension in document.extensions_used() {
        match extension {
            "KHR_materials_volume" => asset.warnings.push(
                "KHR_materials_volume was parsed, but thickness and attenuation are not yet represented by ZeroFPS materials".into(),
            ),
            "KHR_materials_unlit" => asset.warnings.push(
                "KHR_materials_unlit was parsed, but the canonical material does not yet preserve an unlit shader flag".into(),
            ),
            "KHR_materials_variants" => asset.warnings.push(
                "KHR_materials_variants was parsed; the default primitive material was imported and variant mappings were skipped".into(),
            ),
            "KHR_texture_transform" => asset.warnings.push(
                "KHR_texture_transform was parsed, but UV offset/rotation/scale are not yet applied".into(),
            ),
            "KHR_lights_punctual" => asset.warnings.push(
                "KHR_lights_punctual was parsed, but glTF lights are not yet instantiated as ZeroFPS scene lights".into(),
            ),
            _ => {}
        }
    }

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
        let specular_glossiness = material.pbr_specular_glossiness();
        let base_color = specular_glossiness
            .as_ref()
            .map(|value| value.diffuse_factor())
            .unwrap_or_else(|| pbr.base_color_factor());
        let emissive_strength = material.emissive_strength().unwrap_or(1.0);
        let emissive = material
            .emissive_factor()
            .map(|channel| channel * emissive_strength);
        let base_color_texture = specular_glossiness
            .as_ref()
            .and_then(|value| value.diffuse_texture())
            .or_else(|| pbr.base_color_texture())
            .map(|info| format!("image-{}", info.texture().source().index()));
        let (specular, shininess) = if let Some(value) = specular_glossiness.as_ref() {
            (value.specular_factor(), value.glossiness_factor())
        } else if let Some(value) = material.specular() {
            let factor = value.specular_factor();
            (
                value
                    .specular_color_factor()
                    .map(|channel| channel * factor),
                1.0 - pbr.roughness_factor(),
            )
        } else {
            ([pbr.metallic_factor(); 3], 1.0 - pbr.roughness_factor())
        };
        asset.materials.insert(
            name.clone(),
            Material {
                name,
                base_color,
                emissive,
                specular,
                shininess,
                opacity: base_color[3],
                base_color_texture,
                transmission: material
                    .transmission()
                    .map(|value| value.transmission_factor()),
                ior: material.ior(),
            },
        );
    }

    asset.animations = import_animations(&document, &buffers, &mut asset.warnings);
    let scenes: Vec<_> = if let Some(scene) = document.default_scene() {
        vec![scene]
    } else {
        document.scenes().collect()
    };
    for scene in scenes {
        for node in scene.nodes() {
            visit_node(node, None, &document, &buffers, &mut asset)?;
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

fn import_animations(
    document: &::gltf::Document,
    buffers: &[::gltf::buffer::Data],
    warnings: &mut Vec<String>,
) -> Vec<AssetAnimationClip> {
    use ::gltf::animation::util::ReadOutputs;
    document.animations().enumerate().map(|(animation_index, animation)| {
        let mut channels = Vec::new();
        for channel in animation.channels() {
            let reader = channel.reader(|buffer| Some(&buffers[buffer.index()].0));
            let Some(times) = reader.read_inputs().map(Iterator::collect::<Vec<_>>) else { continue; };
            let node = channel.target().node().index();
            let cubic = channel.sampler().interpolation()
                == ::gltf::animation::Interpolation::CubicSpline;
            let output = match reader.read_outputs() {
                Some(ReadOutputs::Translations(values)) => { let values = values.map(z_up).collect::<Vec<_>>(); AssetAnimationChannel::Translation { node, times, values: animation_key_values(values, cubic) } },
                Some(ReadOutputs::Rotations(values)) => { let values = values.into_f32().map(|q| [q[3], q[0], -q[2], q[1]]).collect::<Vec<_>>(); AssetAnimationChannel::Rotation { node, times, values: animation_key_values(values, cubic) } },
                Some(ReadOutputs::Scales(values)) => { let values = values.map(|s| [s[0], s[2], s[1]]).collect::<Vec<_>>(); AssetAnimationChannel::Scale { node, times, values: animation_key_values(values, cubic) } },
                Some(ReadOutputs::MorphTargetWeights(_)) | None => { warnings.push("glTF morph-weight animation import is not available in the quick animation build".into()); continue; }
            };
            channels.push(output);
        }
        AssetAnimationClip { name: animation.name().map(str::to_owned).unwrap_or_else(|| format!("Animation {animation_index}")), channels }
    }).collect()
}

fn animation_key_values<T>(values: Vec<T>, cubic: bool) -> Vec<T> {
    if cubic {
        values.into_iter().skip(1).step_by(3).collect()
    } else {
        values
    }
}

fn visit_node(
    node: ::gltf::Node<'_>,
    parent: Option<usize>,
    document: &::gltf::Document,
    buffers: &[::gltf::buffer::Data],
    asset: &mut MeshAsset,
) -> Result<(), ImportError> {
    let node_index = asset.nodes.len();
    let (translation, rotation, scale) = node.transform().decomposed();
    asset.nodes.push(AssetNode {
        name: node
            .name()
            .map(str::to_owned)
            .unwrap_or_else(|| format!("Node {node_index}")),
        parent,
        children: Vec::new(),
        local: AssetTransform {
            translation: z_up(translation),
            rotation: [rotation[3], rotation[0], -rotation[2], rotation[1]],
            scale: [scale[0], scale[2], scale[1]],
        },
        primitives: Vec::new(),
    });
    if let Some(parent) = parent {
        asset.nodes[parent].children.push(node_index);
    }
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
                let position = z_up(position);
                let normal = normals
                    .as_ref()
                    .and_then(|values| values.get(index).copied())
                    .map(|normal| normalize(z_up(normal)))
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
            let canonical_primitive = asset.primitives.len();
            asset.primitives.push(Primitive {
                name: format!("{}:{primitive_index}", mesh.name().unwrap_or("glTF mesh")),
                material,
                indices,
            });
            asset.nodes[node_index].primitives.push(canonical_primitive);
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
        visit_node(child, Some(node_index), document, buffers, asset)?;
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
        cached_mips: Vec::new(),
    })
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
    fn quaternion_basis_conversion_keeps_identity() {
        let gltf_xyzw = [0.0, 0.0, 0.0, 1.0];
        let engine_wxyz = [gltf_xyzw[3], gltf_xyzw[0], -gltf_xyzw[2], gltf_xyzw[1]];
        assert_eq!(engine_wxyz, [1.0, 0.0, -0.0, 0.0]);
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

    #[test]
    fn imports_car_as_a_transform_hierarchy() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../appdata/models/Fast-Driver/low_poly_car.glb");
        if !fixture.exists() {
            return;
        }
        let asset = import_gltf(&fixture).expect("generated car fixture should import");
        let car = asset
            .nodes
            .iter()
            .position(|node| node.name == "Car")
            .expect("Car root node");
        assert_eq!(asset.nodes[car].parent, None);
        for name in ["Chassis", "Wheel_FL", "Wheel_FR", "Wheel_RL", "Wheel_RR"] {
            let index = asset
                .nodes
                .iter()
                .position(|node| node.name == name)
                .unwrap_or_else(|| panic!("{name} node"));
            assert_eq!(asset.nodes[index].parent, Some(car), "{name} parent");
            assert!(asset.nodes[car].children.contains(&index));
        }
        assert!(
            asset.nodes.iter().any(|node| !node.primitives.is_empty()),
            "mesh primitives must remain owned by their glTF nodes"
        );
        asset.validate().expect("hierarchy must be canonical");
    }

    #[test]
    fn imports_required_specular_glossiness_bridge_fixture() {
        let fixture = Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("../../appdata/models/gltf_glb/bridge_over_a_river.glb");
        let asset = crate::import_file(&fixture)
            .expect("required KHR_materials_pbrSpecularGlossiness should import");
        assert!(!asset.vertices.is_empty());
        assert!(!asset.primitives.is_empty());
        assert!(!asset.materials.is_empty());
        let clip = asset.animations.first().expect("bridge animation clip");
        assert!(clip.channels.iter().any(|channel| matches!(
            channel,
            AssetAnimationChannel::Rotation { node: 4, times, values }
                if !times.is_empty() && !values.is_empty()
        )));
        let root = asset
            .nodes
            .iter()
            .find(|node| node.name == "Sketchfab_model")
            .expect("bridge root node");
        assert_eq!(root.parent, None);
        assert!((root.local.rotation[0] - std::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-6);
        assert!((root.local.rotation[1] + std::f32::consts::FRAC_1_SQRT_2).abs() < 1.0e-6);
        assert!(root.local.rotation[2].abs() < 1.0e-6);
        assert!(root.local.rotation[3].abs() < 1.0e-6);
        assert!(asset.materials.values().all(|material| {
            material.base_color.iter().all(|value| value.is_finite())
                && material.specular.iter().all(|value| value.is_finite())
                && material.shininess.is_finite()
        }));
    }
}
