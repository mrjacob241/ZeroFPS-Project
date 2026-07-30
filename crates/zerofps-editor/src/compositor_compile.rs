//! Translation from the mutable editor node canvas to the executor-neutral IR.
//!
//! The compiler deliberately resolves files and object textures here.  Neither
//! the CPU nor Vulkan executor is allowed to walk editor state or perform I/O.

use std::collections::{BTreeMap, BTreeSet};
use std::path::PathBuf;
use std::sync::Arc;

use zerofps_assets::TextureAsset;

use super::compositor_graph::{
    CompiledGraph, GraphCompileError, GraphNode, GraphNodeId, GraphOperation, GraphSource,
};
use super::{EditorApp, NodeSettings, resize_texture_for_lod};

type EditorSocket = (usize, usize);

pub(crate) struct CompiledCompositorOutput {
    pub graph: CompiledGraph,
    pub object_index: usize,
    pub channel: usize,
}

struct Compiler<'a> {
    app: &'a mut EditorApp,
    links: Vec<(usize, usize, usize, usize)>,
    generation: u64,
    lod: u32,
    sources: Vec<GraphSource>,
    nodes: Vec<GraphNode>,
    compiled: BTreeMap<EditorSocket, GraphNodeId>,
    visiting: BTreeSet<EditorSocket>,
}

impl EditorApp {
    /// Compile one Output node into dependency-first executor IR.
    ///
    /// A socket, rather than merely an editor node, is the identity during
    /// compilation. This matters for Color Decoder: its four outputs produce
    /// four different IR values and can safely feed the same downstream node.
    pub(crate) fn compile_compositor_graph(
        &mut self,
        output_node_id: usize,
        generation: u64,
        lod: u32,
    ) -> Result<CompiledCompositorOutput, GraphCompileError> {
        let output = self
            .compositor_nodes
            .iter()
            .find(|node| node.id == output_node_id)
            .ok_or(GraphCompileError::MissingNode(output_node_id))?;
        let (object_index, channel) = match output.settings {
            NodeSettings::Output {
                object_index,
                channel,
            } => (object_index, channel),
            _ => return Err(GraphCompileError::NotOutputNode(output_node_id)),
        };
        if channel != 0 {
            return Err(GraphCompileError::UnsupportedOutputChannel(channel));
        }
        let links = self.compositor_links.clone();
        let source_socket = input_source(&links, output_node_id, 0)?;
        let mut compiler = Compiler {
            app: self,
            links,
            generation,
            lod,
            sources: Vec::new(),
            nodes: Vec::new(),
            compiled: BTreeMap::new(),
            visiting: BTreeSet::new(),
        };
        let source_output = compiler.compile_socket(source_socket)?;
        let graph_output = compiler.nodes.len();
        compiler.nodes.push(GraphNode {
            id: graph_output,
            operation: GraphOperation::ClampColor,
            inputs: [Some(source_output), None, None, None],
        });
        Ok(CompiledCompositorOutput {
            graph: CompiledGraph {
                generation: compiler.generation,
                sources: compiler.sources,
                nodes: compiler.nodes,
                output: graph_output,
            },
            object_index,
            channel,
        })
    }

    pub(crate) fn compile_compositor_preview(
        &mut self,
        node_id: usize,
        output: usize,
        generation: u64,
        lod: u32,
    ) -> Result<CompiledGraph, GraphCompileError> {
        let links = self.compositor_links.clone();
        let mut compiler = Compiler {
            app: self,
            links,
            generation,
            lod,
            sources: Vec::new(),
            nodes: Vec::new(),
            compiled: BTreeMap::new(),
            visiting: BTreeSet::new(),
        };
        let graph_output = compiler.compile_socket((node_id, output))?;
        Ok(CompiledGraph {
            generation: compiler.generation,
            sources: compiler.sources,
            nodes: compiler.nodes,
            output: graph_output,
        })
    }
}

impl Compiler<'_> {
    fn compile_socket(&mut self, socket: EditorSocket) -> Result<GraphNodeId, GraphCompileError> {
        if let Some(&compiled) = self.compiled.get(&socket) {
            return Ok(compiled);
        }
        if !self.visiting.insert(socket) {
            return Err(GraphCompileError::Cycle(socket.0));
        }

        let settings = self
            .app
            .compositor_nodes
            .iter()
            .find(|node| node.id == socket.0)
            .map(|node| node.settings.clone())
            .ok_or(GraphCompileError::MissingNode(socket.0))?;
        let output_count = if matches!(&settings, NodeSettings::ColorDecoder) {
            4
        } else if matches!(
            &settings,
            NodeSettings::Output { .. }
                | NodeSettings::Debug
                | NodeSettings::ObjectTransform { .. }
                | NodeSettings::ObjectMesh { .. }
        ) {
            0
        } else {
            1
        };
        if socket.1 >= output_count {
            return Err(GraphCompileError::InvalidOutputSocket {
                node: socket.0,
                output: socket.1,
            });
        }

        let mut inputs = [None; 4];
        let operation = match settings {
            NodeSettings::ObjectTexture {
                object_index,
                channel,
            } => {
                if channel != 0 {
                    return Err(GraphCompileError::UnsupportedSourceChannel {
                        node: socket.0,
                        channel,
                    });
                }
                let path = self
                    .app
                    .object_asset_path(object_index)
                    .ok_or_else(|| source_error(socket.0, "selected object has no model asset"))?;
                let mesh = self
                    .app
                    .imported_assets
                    .iter()
                    .find(|asset| asset.path == path)
                    .map(|asset| &asset.mesh)
                    .ok_or_else(|| source_error(socket.0, "selected model asset is unavailable"))?;
                let texture_name = mesh
                    .primitives
                    .iter()
                    .filter_map(|primitive| primitive.material.as_ref())
                    .filter_map(|name| mesh.materials.get(name))
                    .find_map(|material| material.base_color_texture.as_ref())
                    .ok_or_else(|| source_error(socket.0, "model has no base-color texture"))?;
                let source_key = format!("{path}\0{texture_name}");
                let texture =
                    if let Some(texture) = self.app.compositor_source_cache.get(&source_key) {
                        Arc::clone(texture)
                    } else {
                        let texture = mesh
                            .textures
                            .get(texture_name)
                            .cloned()
                            .map(Arc::new)
                            .ok_or_else(|| {
                                source_error(socket.0, "base-color texture is unavailable")
                            })?;
                        self.app
                            .compositor_source_cache
                            .insert(source_key, Arc::clone(&texture));
                        texture
                    };
                self.add_source(GraphSource::Texture(resize_texture_for_lod(
                    &texture, self.lod,
                )))
            }
            NodeSettings::PaintedMask { object_index } => {
                let object = self
                    .app
                    .object_node_id(object_index)
                    .ok_or_else(|| source_error(socket.0, "painted object is unavailable"))?;
                let texture = self
                    .app
                    .painted_masks
                    .get(&object)
                    .map(|mask| Arc::new(mask.texture(false)))
                    .ok_or_else(|| source_error(socket.0, "object has no painted texture"))?;
                self.add_source(GraphSource::Texture(resize_texture_for_lod(
                    &texture, self.lod,
                )))
            }
            NodeSettings::ImageAsset { path } => {
                if path.is_empty() {
                    return Err(source_error(socket.0, "no image was selected"));
                }
                let texture = if let Some(texture) = self.app.compositor_image_cache.get(&path) {
                    Arc::clone(texture)
                } else {
                    let image = image::open(&path)
                        .map_err(|error| {
                            source_error(socket.0, format!("could not decode `{path}`: {error}"))
                        })?
                        .into_rgba8();
                    let texture = Arc::new(TextureAsset {
                        name: PathBuf::from(&path)
                            .file_name()
                            .and_then(|name| name.to_str())
                            .unwrap_or("compositor-image")
                            .into(),
                        width: image.width(),
                        height: image.height(),
                        pixels: image.into_raw(),
                        cached_mips: Vec::new(),
                    });
                    self.app
                        .compositor_image_cache
                        .insert(path.clone(), Arc::clone(&texture));
                    texture
                };
                self.add_source(GraphSource::Texture(resize_texture_for_lod(
                    &texture, self.lod,
                )))
            }
            NodeSettings::ConstantValue { color, .. } => {
                self.add_source(GraphSource::Constant([color[0], color[1], color[2], 1.0]))
            }
            NodeSettings::ObjectHandle { value, .. } => {
                self.add_source(GraphSource::Constant([value, value, value, 1.0]))
            }
            NodeSettings::Time { scale, modulus, .. } => {
                let value = super::scaled_modulated_time(
                    self.app.compositor_clock_started.elapsed().as_secs_f32(),
                    scale,
                    modulus,
                );
                self.add_source(GraphSource::Constant([value, value, value, 1.0]))
            }
            // Debug is a sink and therefore has no connectable graph output.
            // This branch only satisfies exhaustive loading of legacy or
            // hand-edited project data; socket validation rejects it first.
            NodeSettings::Debug => self.add_source(GraphSource::Constant([0.0, 0.0, 0.0, 1.0])),
            NodeSettings::Remap { points, mode, .. } => {
                inputs[0] = Some(self.required_input(socket.0, 0)?);
                GraphOperation::Remap {
                    points,
                    bezier: mode == 1,
                }
            }
            NodeSettings::TextureMath {
                operation,
                constant,
            } => {
                inputs[0] = Some(self.required_input(socket.0, 0)?);
                GraphOperation::Math {
                    operation,
                    constant,
                }
            }
            NodeSettings::Algebra { expression } => {
                for (index, target) in inputs.iter_mut().take(3).enumerate() {
                    *target = self.optional_input(socket.0, index)?;
                }
                let program = super::compositor_graph::compile_algebra_expression(&expression)
                    .map_err(|message| source_error(socket.0, message))?;
                GraphOperation::Algebra { program }
            }
            NodeSettings::SharpThreshold { threshold } => {
                inputs[0] = Some(self.required_input(socket.0, 0)?);
                GraphOperation::SharpThreshold { threshold }
            }
            NodeSettings::SmoothThreshold { threshold, width } => {
                inputs[0] = Some(self.required_input(socket.0, 0)?);
                GraphOperation::SmoothThreshold { threshold, width }
            }
            NodeSettings::ImageFilter { filter, radius } => {
                inputs[0] = Some(self.required_input(socket.0, 0)?);
                GraphOperation::ImageFilter { filter, radius }
            }
            NodeSettings::TextureCombine {
                mode,
                operation,
                alpha,
            } => {
                inputs[0] = Some(self.required_input(socket.0, 0)?);
                inputs[1] = Some(self.required_input(socket.0, 1)?);
                if mode == 1 {
                    inputs[2] = self.optional_input(socket.0, 2)?;
                }
                GraphOperation::Combine {
                    mode,
                    operation,
                    alpha,
                }
            }
            NodeSettings::ColorSpaceConvert { from, to } => {
                inputs[0] = Some(self.required_input(socket.0, 0)?);
                GraphOperation::ColorSpace { from, to }
            }
            NodeSettings::ColorDecoder => {
                inputs[0] = Some(self.required_input(socket.0, 0)?);
                GraphOperation::ExtractChannel { channel: socket.1 }
            }
            NodeSettings::Grayscale { mode } => {
                inputs[0] = Some(self.required_input(socket.0, 0)?);
                GraphOperation::Grayscale { mode }
            }
            NodeSettings::ColorEncoder => {
                // Missing RGB inputs are black; a missing alpha input is opaque.
                for channel in 0..4 {
                    inputs[channel] = Some(match self.optional_input(socket.0, channel)? {
                        Some(input) => input,
                        None => self.add_constant(if channel == 3 { 1.0 } else { 0.0 }),
                    });
                }
                GraphOperation::JoinChannels
            }
            NodeSettings::Output { .. }
            | NodeSettings::ObjectTransform { .. }
            | NodeSettings::ObjectMesh { .. }
            | NodeSettings::Position { .. }
            | NodeSettings::Rotation { .. }
            | NodeSettings::MassDensity { .. }
            | NodeSettings::SpringMesh { .. }
            | NodeSettings::ForceField { .. }
            | NodeSettings::VelocityField { .. }
            | NodeSettings::Simulator { .. }
            | NodeSettings::ObjectSimulator { .. }
            | NodeSettings::ForceOutput { .. }
            | NodeSettings::Engine { .. } => {
                return Err(GraphCompileError::NotOutputNode(socket.0));
            }
        };

        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            id,
            operation,
            inputs,
        });
        self.visiting.remove(&socket);
        self.compiled.insert(socket, id);
        Ok(id)
    }

    fn required_input(
        &mut self,
        node: usize,
        input: usize,
    ) -> Result<GraphNodeId, GraphCompileError> {
        let socket = input_source(&self.links, node, input)?;
        self.compile_socket(socket)
    }

    fn optional_input(
        &mut self,
        node: usize,
        input: usize,
    ) -> Result<Option<GraphNodeId>, GraphCompileError> {
        let Some(socket) = optional_input_source(&self.links, node, input)? else {
            return Ok(None);
        };
        self.compile_socket(socket).map(Some)
    }

    fn add_source(&mut self, source: GraphSource) -> GraphOperation {
        let index = self.sources.len();
        self.sources.push(source);
        GraphOperation::Source(index)
    }

    fn add_constant(&mut self, value: f32) -> GraphNodeId {
        let operation = self.add_source(GraphSource::Constant([value, value, value, value]));
        let id = self.nodes.len();
        self.nodes.push(GraphNode {
            id,
            operation,
            inputs: [None; 4],
        });
        id
    }
}

fn input_source(
    links: &[(usize, usize, usize, usize)],
    node: usize,
    input: usize,
) -> Result<EditorSocket, GraphCompileError> {
    optional_input_source(links, node, input)?
        .ok_or(GraphCompileError::MissingInput { node, input })
}

fn optional_input_source(
    links: &[(usize, usize, usize, usize)],
    node: usize,
    input: usize,
) -> Result<Option<EditorSocket>, GraphCompileError> {
    let mut matches = links
        .iter()
        .filter(|link| link.2 == node && link.3 == input)
        .map(|link| (link.0, link.1));
    let result = matches.next();
    if matches.next().is_some() {
        return Err(GraphCompileError::MultipleInputs { node, input });
    }
    Ok(result)
}

fn source_error(node: usize, message: impl Into<String>) -> GraphCompileError {
    GraphCompileError::SourceUnavailable {
        node,
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn input_lookup_rejects_missing_and_ambiguous_links() {
        assert_eq!(
            input_source(&[], 7, 2),
            Err(GraphCompileError::MissingInput { node: 7, input: 2 })
        );
        let links = [(1, 0, 7, 2), (2, 0, 7, 2)];
        assert_eq!(
            input_source(&links, 7, 2),
            Err(GraphCompileError::MultipleInputs { node: 7, input: 2 })
        );
    }

    #[test]
    fn input_lookup_preserves_output_socket() {
        let links = [(11, 3, 20, 0)];
        assert_eq!(input_source(&links, 20, 0), Ok((11, 3)));
    }
}
