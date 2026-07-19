//! Human-readable, versioned ZeroFPS project persistence.

use serde::{Deserialize, Serialize};
use std::{
    collections::{BTreeMap, BTreeSet},
    fs,
    path::Path,
};
use thiserror::Error;
use zerofps_core::{
    Attribute, AttributeDeclaration, AttributeKey, Component, GeometryTree, NodeId, Transform, Vec3,
};

pub const FORMAT_NAME: &str = "zerofps-project";
pub const FORMAT_VERSION: u32 = 1;

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
}
