//! Interoperable mesh import for ZeroFPS.
//!
//! Codecs translate external files into this canonical representation. Runtime
//! and scene code never depend on source-format details.

pub mod obj;
pub mod ply;
pub mod stl;

use std::{collections::BTreeMap, path::Path};
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

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum MeshFormat {
    Obj,
    Ply,
    Stl,
}

impl MeshFormat {
    pub fn from_path(path: &Path) -> Option<Self> {
        match path.extension()?.to_str()?.to_ascii_lowercase().as_str() {
            "obj" => Some(Self::Obj),
            "ply" => Some(Self::Ply),
            "stl" => Some(Self::Stl),
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
}
