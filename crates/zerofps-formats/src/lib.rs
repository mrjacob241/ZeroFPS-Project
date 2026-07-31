//! Human-readable, versioned ZeroFPS project persistence.

use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs::{self, File},
    io::{Read, Write},
    path::{Component as PathComponent, Path, PathBuf},
    time::{SystemTime, UNIX_EPOCH},
};
use thiserror::Error;
use zerofps_core::{
    Attribute, AttributeDeclaration, AttributeKey, Component, GeometryTree, NodeId, Transform, Vec3,
};

pub const FORMAT_NAME: &str = "zerofps-project";
pub const FORMAT_VERSION: u32 = 1;
pub const ARCHIVE_MANIFEST: &str = "scene.json";

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct BundleAsset {
    pub source: PathBuf,
    pub archive_path: String,
}

#[derive(Debug)]
pub struct LoadedBundle {
    pub project: ProjectFile,
    /// Archive-relative names mapped to safely extracted local files.
    pub extracted_files: BTreeMap<String, PathBuf>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct ProjectFile {
    /// Allows tools to reject unrelated JSON before interpreting it.
    pub format: String,
    pub version: u32,
    pub project: ProjectMetadata,
    pub scene: SceneFile,
}

#[derive(Clone, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub struct ProjectMetadata {
    pub name: String,
    #[serde(default)]
    pub entry_scene: String,
    #[serde(default)]
    pub properties: BTreeMap<String, String>,
}

#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct SceneFile {
    pub name: String,
    pub geometry: GeometryTree,
}

impl ProjectFile {
    pub fn new(
        name: impl Into<String>,
        scene_name: impl Into<String>,
        geometry: GeometryTree,
    ) -> Self {
        let scene_name = scene_name.into();
        Self {
            format: FORMAT_NAME.into(),
            version: FORMAT_VERSION,
            project: ProjectMetadata {
                name: name.into(),
                entry_scene: scene_name.clone(),
                properties: BTreeMap::new(),
            },
            scene: SceneFile {
                name: scene_name,
                geometry,
            },
        }
    }

    pub fn validate(&self) -> Result<(), FormatError> {
        if self.format != FORMAT_NAME {
            return Err(FormatError::WrongFormat(self.format.clone()));
        }
        if self.version != FORMAT_VERSION {
            return Err(FormatError::UnsupportedVersion(self.version));
        }
        validate_name("project", &self.project.name)?;
        validate_name("scene", &self.scene.name)?;
        validate_tree(&self.scene.geometry)
    }

    /// Canonical pretty JSON. BTreeMaps and geometry slots keep output stable.
    pub fn to_json(&self) -> Result<String, FormatError> {
        self.validate()?;
        let mut output = serde_json::to_string_pretty(self)?;
        output.push('\n');
        Ok(output)
    }

    pub fn from_json(input: &str) -> Result<Self, FormatError> {
        let mut project: Self = serde_json::from_str(input)?;
        project.validate()?;
        project.scene.geometry.propagate_all()?;
        Ok(project)
    }

    pub fn save(&self, path: impl AsRef<Path>) -> Result<(), FormatError> {
        fs::write(path, self.to_json()?)?;
        Ok(())
    }

    pub fn load(path: impl AsRef<Path>) -> Result<Self, FormatError> {
        Self::from_json(&fs::read_to_string(path)?)
    }
}

/// Saves a ZIP-compatible ZeroFPS project container with a `.zfp` extension.
pub fn save_zfp(
    path: impl AsRef<Path>,
    project: &ProjectFile,
    files: &[BundleAsset],
) -> Result<(), FormatError> {
    let path = path.as_ref();
    let manifest = project.to_json()?;
    let mut names = BTreeSet::new();
    for file in files {
        validate_archive_path(&file.archive_path)?;
        if !names.insert(file.archive_path.clone()) {
            return Err(FormatError::InvalidArchive(format!(
                "duplicate archive path `{}`",
                file.archive_path
            )));
        }
    }
    for file in files {
        File::open(&file.source)?;
    }
    let temporary = temporary_archive_path(path);
    let result = (|| {
        let output = File::options()
            .write(true)
            .create_new(true)
            .open(&temporary)?;
        let mut archive = zip::ZipWriter::new(output);
        let options = zip::write::SimpleFileOptions::default()
            .compression_method(zip::CompressionMethod::Deflated)
            .unix_permissions(0o644);
        archive.start_file(ARCHIVE_MANIFEST, options)?;
        archive.write_all(manifest.as_bytes())?;
        for file in files {
            archive.start_file(&file.archive_path, options)?;
            let mut source = File::open(&file.source)?;
            std::io::copy(&mut source, &mut archive)?;
        }
        let output = archive.finish()?;
        output.sync_all()?;
        verify_zfp_archive(&temporary)?;
        fs::rename(&temporary, path)?;
        Ok(())
    })();
    if result.is_err() {
        let _ = fs::remove_file(&temporary);
    }
    result
}

fn temporary_archive_path(path: &Path) -> PathBuf {
    let filename = path
        .file_name()
        .and_then(|name| name.to_str())
        .unwrap_or("project.zfp");
    let nonce = SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .map(|duration| duration.as_nanos())
        .unwrap_or_default();
    path.with_file_name(format!(".{filename}.{}.{nonce}.tmp", std::process::id()))
}

fn verify_zfp_archive(path: &Path) -> Result<(), FormatError> {
    let mut archive = zip::ZipArchive::new(File::open(path)?)?;
    let mut manifest = String::new();
    {
        archive
            .by_name(ARCHIVE_MANIFEST)
            .map_err(|_| {
                FormatError::InvalidArchive("temporary archive is missing scene.json".into())
            })?
            .read_to_string(&mut manifest)?;
    }
    ProjectFile::from_json(&manifest)?;
    let mut buffer = [0_u8; 64 * 1024];
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        while entry.read(&mut buffer)? != 0 {}
    }
    Ok(())
}

/// Opens a `.zfp` and extracts only validated relative files beneath `cache_root`.
pub fn load_zfp(
    path: impl AsRef<Path>,
    cache_root: impl AsRef<Path>,
) -> Result<LoadedBundle, FormatError> {
    let mut archive = zip::ZipArchive::new(File::open(path)?)?;
    let mut manifest = String::new();
    archive
        .by_name(ARCHIVE_MANIFEST)
        .map_err(|_| FormatError::InvalidArchive("missing scene.json".into()))?
        .read_to_string(&mut manifest)?;
    let project = ProjectFile::from_json(&manifest)?;
    let cache_root = cache_root.as_ref();
    fs::create_dir_all(cache_root)?;
    let mut extracted_files = BTreeMap::new();
    for index in 0..archive.len() {
        let mut entry = archive.by_index(index)?;
        let name = entry.name().to_owned();
        if name == ARCHIVE_MANIFEST || entry.is_dir() {
            continue;
        }
        validate_archive_path(&name)?;
        let destination = cache_root.join(&name);
        if let Some(parent) = destination.parent() {
            fs::create_dir_all(parent)?;
        }
        let mut output = File::create(&destination)?;
        std::io::copy(&mut entry, &mut output)?;
        extracted_files.insert(name, destination);
    }
    Ok(LoadedBundle {
        project,
        extracted_files,
    })
}

fn validate_archive_path(path: &str) -> Result<(), FormatError> {
    let parsed = Path::new(path);
    if path.is_empty()
        || parsed.is_absolute()
        || parsed.components().any(|component| {
            matches!(
                component,
                PathComponent::ParentDir | PathComponent::RootDir | PathComponent::Prefix(_)
            )
        })
        || !(path.starts_with("assets/") || path.starts_with("fields/"))
    {
        return Err(FormatError::InvalidArchive(format!(
            "unsafe or unsupported archive path `{path}`"
        )));
    }
    Ok(())
}

fn validate_name(kind: &'static str, name: &str) -> Result<(), FormatError> {
    if name.trim().is_empty() {
        Err(FormatError::EmptyName(kind))
    } else {
        Ok(())
    }
}

fn validate_tree(tree: &GeometryTree) -> Result<(), FormatError> {
    let ids: BTreeSet<_> = tree.iter().map(|(id, _)| id).collect();
    let roots: BTreeSet<_> = tree.roots().iter().copied().collect();
    if roots.len() != tree.roots().len() {
        return Err(FormatError::InvalidTree("duplicate root".into()));
    }
    for root in &roots {
        if !ids.contains(root) || tree.node(*root)?.parent().is_some() {
            return Err(FormatError::InvalidTree(format!("invalid root {root:?}")));
        }
    }
    for (id, node) in tree.iter() {
        validate_name("object", &node.name)?;
        match node.parent() {
            Some(parent) => {
                if !ids.contains(&parent) || !tree.node(parent)?.children().contains(&id) {
                    return Err(FormatError::InvalidTree(format!(
                        "node {id:?} has inconsistent parent {parent:?}"
                    )));
                }
            }
            None if !roots.contains(&id) => {
                return Err(FormatError::InvalidTree(format!("unlisted root {id:?}")));
            }
            None => {}
        }
        let mut unique = BTreeSet::new();
        for child in node.children() {
            if !unique.insert(*child)
                || !ids.contains(child)
                || tree.node(*child)?.parent() != Some(id)
            {
                return Err(FormatError::InvalidTree(format!(
                    "node {id:?} has invalid child {child:?}"
                )));
            }
        }
    }

    let mut visited = BTreeSet::new();
    let mut active = BTreeSet::new();
    for root in tree.roots() {
        visit(tree, *root, &mut visited, &mut active)?;
    }
    if visited != ids {
        return Err(FormatError::InvalidTree(
            "tree contains unreachable nodes or a parent cycle".into(),
        ));
    }
    Ok(())
}

fn visit(
    tree: &GeometryTree,
    id: NodeId,
    visited: &mut BTreeSet<NodeId>,
    active: &mut BTreeSet<NodeId>,
) -> Result<(), FormatError> {
    if !active.insert(id) {
        return Err(FormatError::InvalidTree(format!("cycle at {id:?}")));
    }
    if visited.insert(id) {
        for child in tree.node(id)?.children() {
            visit(tree, *child, visited, active)?;
        }
    }
    active.remove(&id);
    Ok(())
}

/// A small but meaningful project used by the editor's first-run experience.
pub fn sample_project() -> ProjectFile {
    let mut tree = GeometryTree::new();
    let root = tree.create("Sample scene", None).expect("new tree");
    let model = tree.create("Welcome cube", Some(root)).expect("valid root");
    tree.set_local_transform(
        model,
        Transform {
            translation: Vec3::new(0.0, 0.5, 0.0),
            ..Transform::IDENTITY
        },
    )
    .expect("valid model");
    tree.add_component(
        model,
        Component::Model {
            asset: "builtin:cube".into(),
            primitive: None,
        },
    )
    .expect("valid model");
    tree.set_attribute(
        root,
        AttributeKey::Tint,
        AttributeDeclaration::Value(Attribute::Color([0.18, 0.55, 0.9, 1.0])),
    )
    .expect("valid root");
    ProjectFile::new("My ZeroFPS Project", "Main", tree)
}

#[derive(Debug, Error)]
pub enum FormatError {
    #[error("I/O error: {0}")]
    Io(#[from] std::io::Error),
    #[error("invalid JSON: {0}")]
    Json(#[from] serde_json::Error),
    #[error("ZIP error: {0}")]
    Zip(#[from] zip::result::ZipError),
    #[error("scene error: {0}")]
    Scene(#[from] zerofps_core::SceneError),
    #[error("expected format `{FORMAT_NAME}`, found `{0}`")]
    WrongFormat(String),
    #[error("project format version {0} is not supported")]
    UnsupportedVersion(u32),
    #[error("{0} name cannot be empty")]
    EmptyName(&'static str),
    #[error("invalid geometry tree: {0}")]
    InvalidTree(String),
    #[error("invalid ZFP archive: {0}")]
    InvalidArchive(String),
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_round_trip_is_canonical() {
        let first = sample_project().to_json().unwrap();
        let loaded = ProjectFile::from_json(&first).unwrap();
        assert_eq!(loaded.to_json().unwrap(), first);
        assert!(first.ends_with('\n'));
        assert!(first.contains("\"Welcome cube\""));
    }

    #[test]
    fn collider_settings_round_trip_in_readable_scene_json() {
        let mut project = sample_project();
        let root = project.scene.geometry.roots()[0];
        project
            .scene
            .geometry
            .add_component(
                root,
                Component::Collider {
                    shape: "cylinder".into(),
                    center: Vec3::new(1.0, 2.0, 3.0),
                    half_extents: Vec3::new(0.75, 0.75, 1.5),
                    radius: 0.75,
                    height: 3.0,
                    coupling_stiffness: 0.5,
                    coupling_damping: 1.0,
                    elasticity_stiffness: 0.5,
                    elasticity_damping: 1.0,
                    restitution: 0.0,
                    force_cutoff: 0.01,
                    collision_force_cutoff: 0.01,
                    density: 1_000.0,
                    mass: 5_301.4375,
                    automatic_mass: true,
                    friction: Vec3::new(0.05, 0.8, 0.8),
                    friction_margin_percent: 5.0,
                    joint: "engine".into(),
                },
            )
            .unwrap();
        let json = project.to_json().unwrap();
        let loaded = ProjectFile::from_json(&json).unwrap();
        assert_eq!(loaded.to_json().unwrap(), json);
        assert!(json.contains("\"shape\": \"cylinder\""));
        assert!(json.contains("\"radius\": 0.75"));
        assert!(json.contains("\"joint\": \"engine\""));

        let legacy: Component = serde_json::from_str(r#"{"Collider":{"shape":"box"}}"#).unwrap();
        assert!(matches!(
            legacy,
            Component::Collider {
                half_extents,
                radius: 0.5,
                height: 1.0,
                restitution: 0.0,
                friction_margin_percent: 5.0,
                ..
            } if half_extents == Vec3::new(0.5, 0.5, 0.5)
        ));
    }

    #[test]
    fn rejects_wrong_version_and_empty_names() {
        let mut project = sample_project();
        project.version = 99;
        assert!(matches!(
            project.validate(),
            Err(FormatError::UnsupportedVersion(99))
        ));
        project.version = FORMAT_VERSION;
        project.project.name.clear();
        assert!(matches!(
            project.validate(),
            Err(FormatError::EmptyName("project"))
        ));
    }

    #[test]
    fn output_is_deterministic() {
        let mut project = sample_project();
        project
            .project
            .properties
            .insert("z-last".into(), "z".into());
        project
            .project
            .properties
            .insert("a-first".into(), "a".into());
        let one = project.to_json().unwrap();
        let two = project.to_json().unwrap();
        assert_eq!(one, two);
        assert!(one.find("a-first").unwrap() < one.find("z-last").unwrap());
    }

    #[test]
    fn filesystem_round_trip() {
        let path = std::env::temp_dir().join(format!(
            "zerofps-format-test-{}-{}.json",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        let project = sample_project();
        project.save(&path).unwrap();
        let loaded = ProjectFile::load(&path).unwrap();
        assert_eq!(loaded.to_json().unwrap(), project.to_json().unwrap());
        let _ = fs::remove_file(path);
    }

    #[test]
    fn zfp_round_trip_contains_manifest_assets_and_fields() {
        let root = std::env::temp_dir().join(format!(
            "zerofps-zfp-test-{}-{}",
            std::process::id(),
            std::thread::current().name().unwrap_or("test")
        ));
        fs::create_dir_all(&root).unwrap();
        let model = root.join("model.glb");
        let field = root.join("temperature.bin");
        fs::write(&model, b"model bytes").unwrap();
        fs::write(&field, b"field bytes").unwrap();
        let archive = root.join("project.zfp");
        let project = sample_project();
        save_zfp(
            &archive,
            &project,
            &[
                BundleAsset {
                    source: model,
                    archive_path: "assets/0000-model.glb".into(),
                },
                BundleAsset {
                    source: field,
                    archive_path: "fields/temperature.bin".into(),
                },
            ],
        )
        .unwrap();

        let loaded = load_zfp(&archive, root.join("cache")).unwrap();
        assert_eq!(
            loaded.project.to_json().unwrap(),
            project.to_json().unwrap()
        );
        assert_eq!(
            fs::read(&loaded.extracted_files["assets/0000-model.glb"]).unwrap(),
            b"model bytes"
        );
        assert_eq!(
            fs::read(&loaded.extracted_files["fields/temperature.bin"]).unwrap(),
            b"field bytes"
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn zfp_round_trip_supports_imported_model_custom_attributes() {
        let root = std::env::temp_dir().join(format!(
            "zerofps-zfp-custom-attribute-{}",
            std::process::id()
        ));
        fs::create_dir_all(&root).unwrap();
        let archive = root.join("model-only.zfp");
        let mut project = sample_project();
        let model = project.scene.geometry.roots()[0];
        project
            .scene
            .geometry
            .set_attribute(
                model,
                AttributeKey::Custom("mesh.autofix".into()),
                AttributeDeclaration::Value(Attribute::Bool(true)),
            )
            .unwrap();

        save_zfp(&archive, &project, &[]).unwrap();
        let loaded = load_zfp(&archive, root.join("cache")).unwrap();
        assert_eq!(
            loaded.project.to_json().unwrap(),
            project.to_json().unwrap()
        );
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn zfp_rejects_escaping_paths() {
        let file = BundleAsset {
            source: PathBuf::from("unused"),
            archive_path: "assets/../../escape".into(),
        };
        let path = std::env::temp_dir().join("zerofps-invalid.zfp");
        assert!(matches!(
            save_zfp(path, &sample_project(), &[file]),
            Err(FormatError::InvalidArchive(_))
        ));
    }

    #[test]
    fn zfp_rejects_duplicate_entry_names_before_reading_sources() {
        let duplicate = || BundleAsset {
            source: PathBuf::from("unused"),
            archive_path: "assets/0000/model.glb".into(),
        };
        let path = std::env::temp_dir().join("zerofps-duplicate.zfp");
        assert!(matches!(
            save_zfp(path, &sample_project(), &[duplicate(), duplicate()]),
            Err(FormatError::InvalidArchive(_))
        ));
    }

    #[test]
    fn failed_zfp_save_preserves_existing_destination() {
        let root = std::env::temp_dir().join(format!("zerofps-atomic-save-{}", std::process::id()));
        fs::create_dir_all(&root).unwrap();
        let destination = root.join("existing.zfp");
        fs::write(&destination, b"previous project bytes").unwrap();
        let missing = BundleAsset {
            source: root.join("missing.glb"),
            archive_path: "assets/missing.glb".into(),
        };
        assert!(save_zfp(&destination, &sample_project(), &[missing]).is_err());
        assert_eq!(fs::read(&destination).unwrap(), b"previous project bytes");
        assert!(fs::read_dir(&root).unwrap().all(|entry| {
            !entry
                .unwrap()
                .file_name()
                .to_string_lossy()
                .ends_with(".tmp")
        }));
        let _ = fs::remove_dir_all(root);
    }
}
