use std::{collections::HashMap, fmt, sync::Arc};

use zerofps_assets::TextureAsset;

pub type GraphNodeId = usize;
pub const MAX_ALGEBRA_INSTRUCTIONS: usize = 32;

#[derive(Clone, Copy, Debug, PartialEq)]
pub enum AlgebraInstruction {
    Variable(u8),
    Constant(f32),
    Add,
    Subtract,
    Multiply,
    Divide,
    Power,
    Negate,
    Sin,
    Cos,
    Abs,
    Sign,
    Sqrt,
}

pub fn compile_algebra_expression(expression: &str) -> Result<Vec<AlgebraInstruction>, String> {
    let mut parser = AlgebraParser {
        input: expression.as_bytes(),
        cursor: 0,
        program: Vec::new(),
    };
    parser.expression()?;
    parser.whitespace();
    if parser.cursor != parser.input.len() {
        return Err("unexpected trailing expression".into());
    }
    if parser.program.is_empty() {
        return Err("expression is empty".into());
    }
    if parser.program.len() > MAX_ALGEBRA_INSTRUCTIONS {
        return Err(format!(
            "expression needs {} operations; maximum is {MAX_ALGEBRA_INSTRUCTIONS}",
            parser.program.len()
        ));
    }
    Ok(parser.program)
}

pub fn evaluate_algebra_program(
    program: &[AlgebraInstruction],
    variables: [f32; 3],
) -> Result<f32, String> {
    let mut stack = Vec::with_capacity(program.len());
    for instruction in program {
        match *instruction {
            AlgebraInstruction::Variable(index) => stack.push(variables[index as usize]),
            AlgebraInstruction::Constant(value) => stack.push(value),
            AlgebraInstruction::Negate
            | AlgebraInstruction::Sin
            | AlgebraInstruction::Cos
            | AlgebraInstruction::Abs
            | AlgebraInstruction::Sign
            | AlgebraInstruction::Sqrt => {
                let value = stack.pop().ok_or("invalid unary expression")?;
                stack.push(match instruction {
                    AlgebraInstruction::Negate => -value,
                    AlgebraInstruction::Sin => value.sin(),
                    AlgebraInstruction::Cos => value.cos(),
                    AlgebraInstruction::Abs => value.abs(),
                    AlgebraInstruction::Sign => {
                        if value > 0.0 {
                            1.0
                        } else if value < 0.0 {
                            -1.0
                        } else {
                            0.0
                        }
                    }
                    AlgebraInstruction::Sqrt => value.max(0.0).sqrt(),
                    _ => unreachable!(),
                });
            }
            operation => {
                let right = stack.pop().ok_or("invalid binary expression")?;
                let left = stack.pop().ok_or("invalid binary expression")?;
                stack.push(match operation {
                    AlgebraInstruction::Add => left + right,
                    AlgebraInstruction::Subtract => left - right,
                    AlgebraInstruction::Multiply => left * right,
                    AlgebraInstruction::Divide => {
                        if right.abs() <= 1.0e-6 {
                            0.0
                        } else {
                            left / right
                        }
                    }
                    AlgebraInstruction::Power => left.max(0.0).powf(right),
                    _ => unreachable!(),
                });
            }
        }
    }
    let value = stack.pop().ok_or("expression produced no value")?;
    if !stack.is_empty() || !value.is_finite() {
        return Err("expression produced an invalid value".into());
    }
    Ok(value)
}

struct AlgebraParser<'a> {
    input: &'a [u8],
    cursor: usize,
    program: Vec<AlgebraInstruction>,
}

impl AlgebraParser<'_> {
    fn expression(&mut self) -> Result<(), String> {
        self.term()?;
        loop {
            self.whitespace();
            if self.consume(b'+') {
                self.term()?;
                self.program.push(AlgebraInstruction::Add);
            } else if self.consume(b'-') {
                self.term()?;
                self.program.push(AlgebraInstruction::Subtract);
            } else {
                return Ok(());
            }
        }
    }

    fn term(&mut self) -> Result<(), String> {
        self.power()?;
        loop {
            self.whitespace();
            if self.consume(b'*') {
                self.power()?;
                self.program.push(AlgebraInstruction::Multiply);
            } else if self.consume(b'/') {
                self.power()?;
                self.program.push(AlgebraInstruction::Divide);
            } else {
                return Ok(());
            }
        }
    }

    fn power(&mut self) -> Result<(), String> {
        self.unary()?;
        self.whitespace();
        if self.consume(b'^') {
            self.power()?;
            self.program.push(AlgebraInstruction::Power);
        }
        Ok(())
    }

    fn unary(&mut self) -> Result<(), String> {
        self.whitespace();
        if self.consume(b'+') {
            self.unary()
        } else if self.consume(b'-') {
            self.unary()?;
            self.program.push(AlgebraInstruction::Negate);
            Ok(())
        } else {
            self.primary()
        }
    }

    fn primary(&mut self) -> Result<(), String> {
        self.whitespace();
        if self.consume(b'(') {
            self.expression()?;
            self.whitespace();
            return self
                .consume(b')')
                .then_some(())
                .ok_or_else(|| "missing closing parenthesis".into());
        }
        if self
            .input
            .get(self.cursor)
            .is_some_and(|byte| byte.is_ascii_digit() || *byte == b'.')
        {
            let value = self.number()?;
            self.program.push(AlgebraInstruction::Constant(value));
            return Ok(());
        }
        let identifier = self.identifier()?;
        match identifier.as_str() {
            "x" => self.program.push(AlgebraInstruction::Variable(0)),
            "y" => self.program.push(AlgebraInstruction::Variable(1)),
            "z" => self.program.push(AlgebraInstruction::Variable(2)),
            "pi" => self
                .program
                .push(AlgebraInstruction::Constant(std::f32::consts::PI)),
            "e" => self
                .program
                .push(AlgebraInstruction::Constant(std::f32::consts::E)),
            function => {
                self.whitespace();
                if !self.consume(b'(') {
                    return Err(format!("unknown variable `{function}`"));
                }
                self.expression()?;
                self.whitespace();
                if !self.consume(b')') {
                    return Err("missing function parenthesis".into());
                }
                self.program.push(match function {
                    "sin" => AlgebraInstruction::Sin,
                    "cos" => AlgebraInstruction::Cos,
                    "abs" => AlgebraInstruction::Abs,
                    "sign" => AlgebraInstruction::Sign,
                    "sqrt" => AlgebraInstruction::Sqrt,
                    _ => return Err(format!("unknown function `{function}`")),
                });
            }
        }
        Ok(())
    }

    fn number(&mut self) -> Result<f32, String> {
        let start = self.cursor;
        while self.input.get(self.cursor).is_some_and(|byte| {
            byte.is_ascii_digit() || matches!(*byte, b'.' | b'e' | b'E' | b'+' | b'-')
        }) {
            if self.cursor > start
                && matches!(self.input[self.cursor], b'+' | b'-')
                && !matches!(self.input[self.cursor - 1], b'e' | b'E')
            {
                break;
            }
            self.cursor += 1;
        }
        std::str::from_utf8(&self.input[start..self.cursor])
            .ok()
            .and_then(|value| value.parse().ok())
            .ok_or_else(|| "invalid number".into())
    }

    fn identifier(&mut self) -> Result<String, String> {
        let start = self.cursor;
        while self
            .input
            .get(self.cursor)
            .is_some_and(|byte| byte.is_ascii_alphabetic() || *byte == b'_')
        {
            self.cursor += 1;
        }
        (self.cursor > start)
            .then(|| String::from_utf8_lossy(&self.input[start..self.cursor]).into_owned())
            .ok_or_else(|| "expected a value".into())
    }

    fn whitespace(&mut self) {
        while self
            .input
            .get(self.cursor)
            .is_some_and(|byte| byte.is_ascii_whitespace())
        {
            self.cursor += 1;
        }
    }

    fn consume(&mut self, expected: u8) -> bool {
        if self.input.get(self.cursor) == Some(&expected) {
            self.cursor += 1;
            true
        } else {
            false
        }
    }
}

#[derive(Clone)]
pub enum GraphSource {
    Texture(Arc<TextureAsset>),
    Constant([f32; 4]),
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphOperation {
    Source(usize),
    Remap {
        points: Vec<[f32; 2]>,
        bezier: bool,
    },
    Math {
        operation: usize,
        constant: f32,
    },
    Algebra {
        program: Vec<AlgebraInstruction>,
    },
    SharpThreshold {
        threshold: f32,
    },
    SmoothThreshold {
        threshold: f32,
        width: f32,
    },
    ImageFilter {
        filter: usize,
        radius: f32,
    },
    Combine {
        mode: usize,
        operation: usize,
        alpha: f32,
    },
    ColorSpace {
        from: usize,
        to: usize,
    },
    ExtractChannel {
        channel: usize,
    },
    Grayscale {
        mode: usize,
    },
    JoinChannels,
    ClampColor,
}

#[derive(Clone, Debug, PartialEq)]
pub struct GraphNode {
    pub id: GraphNodeId,
    pub operation: GraphOperation,
    pub inputs: [Option<GraphNodeId>; 4],
}

#[derive(Clone)]
pub struct CompiledGraph {
    pub generation: u64,
    pub sources: Vec<GraphSource>,
    /// Dependency-first order; executors never recurse through editor state.
    pub nodes: Vec<GraphNode>,
    pub output: GraphNodeId,
}

#[derive(Clone, Debug, PartialEq)]
pub enum GraphCompileError {
    MissingNode(GraphNodeId),
    MissingInput { node: GraphNodeId, input: usize },
    MultipleInputs { node: GraphNodeId, input: usize },
    Cycle(GraphNodeId),
    InvalidOutputSocket { node: GraphNodeId, output: usize },
    NotOutputNode(GraphNodeId),
    SourceUnavailable { node: GraphNodeId, message: String },
    UnsupportedSourceChannel { node: GraphNodeId, channel: usize },
    UnsupportedOutputChannel(usize),
}

impl std::fmt::Display for GraphCompileError {
    fn fmt(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::MissingNode(node) => write!(formatter, "compositor node {node} does not exist"),
            Self::MissingInput { node, input } => {
                write!(
                    formatter,
                    "input {} on compositor node {node} is not connected",
                    input + 1
                )
            }
            Self::MultipleInputs { node, input } => {
                write!(
                    formatter,
                    "input {} on compositor node {node} has multiple links",
                    input + 1
                )
            }
            Self::Cycle(node) => write!(
                formatter,
                "compositor graph contains a cycle at node {node}"
            ),
            Self::InvalidOutputSocket { node, output } => {
                write!(
                    formatter,
                    "output {} does not exist on compositor node {node}",
                    output + 1
                )
            }
            Self::NotOutputNode(node) => {
                write!(formatter, "compositor node {node} is not an Output node")
            }
            Self::SourceUnavailable { node, message } => {
                write!(
                    formatter,
                    "source on compositor node {node} is unavailable: {message}"
                )
            }
            Self::UnsupportedSourceChannel { node, channel } => {
                write!(
                    formatter,
                    "source channel {channel} on compositor node {node} is not implemented"
                )
            }
            Self::UnsupportedOutputChannel(channel) => {
                write!(formatter, "output channel {channel} is not implemented")
            }
        }
    }
}

impl std::error::Error for GraphCompileError {}

pub trait GraphExecutor {
    type Image;
    type Error;

    fn execute(&mut self, graph: &CompiledGraph) -> Result<Self::Image, Self::Error>;
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub enum GraphExecutionError {
    InvalidSource(usize),
    InvalidTexture {
        source: usize,
        expected_bytes: usize,
        actual_bytes: usize,
    },
    MissingInput {
        node: GraphNodeId,
        input: usize,
    },
    MissingDependency {
        node: GraphNodeId,
        dependency: GraphNodeId,
    },
    DuplicateNode(GraphNodeId),
    MissingOutput(GraphNodeId),
}

impl fmt::Display for GraphExecutionError {
    fn fmt(&self, formatter: &mut fmt::Formatter<'_>) -> fmt::Result {
        match self {
            Self::InvalidSource(index) => write!(formatter, "graph source {index} does not exist"),
            Self::InvalidTexture {
                source,
                expected_bytes,
                actual_bytes,
            } => write!(
                formatter,
                "graph source {source} has {actual_bytes} bytes; expected {expected_bytes}"
            ),
            Self::MissingInput { node, input } => {
                write!(formatter, "graph node {node} is missing input {input}")
            }
            Self::MissingDependency { node, dependency } => write!(
                formatter,
                "graph node {node} depends on unavailable node {dependency}"
            ),
            Self::DuplicateNode(node) => write!(formatter, "graph node {node} appears twice"),
            Self::MissingOutput(node) => {
                write!(formatter, "graph output node {node} is unavailable")
            }
        }
    }
}

impl std::error::Error for GraphExecutionError {}

/// Deterministic reference executor used by the CPU device and GPU parity tests.
///
#[derive(Clone, Debug, PartialEq)]
pub struct FloatImage {
    pub name: String,
    pub width: u32,
    pub height: u32,
    pub pixels: Vec<f32>,
}

impl FloatImage {
    pub fn to_texture_asset_clamped(&self) -> TextureAsset {
        TextureAsset {
            name: self.name.clone(),
            width: self.width,
            height: self.height,
            pixels: self.pixels.iter().copied().map(to_u8).collect(),
            cached_mips: Vec::new(),
        }
    }
}

/// Deterministic floating-point reference executor. Quantization occurs only
/// when a `ClampColor` boundary is converted into a renderable TextureAsset.
#[derive(Default)]
pub struct CpuGraphExecutor;

impl GraphExecutor for CpuGraphExecutor {
    type Image = Arc<FloatImage>;
    type Error = GraphExecutionError;

    fn execute(&mut self, graph: &CompiledGraph) -> Result<Self::Image, Self::Error> {
        let mut images = HashMap::<GraphNodeId, Arc<FloatImage>>::new();
        for node in &graph.nodes {
            if images.contains_key(&node.id) {
                return Err(GraphExecutionError::DuplicateNode(node.id));
            }
            let image = execute_node(graph, node, &images)?;
            images.insert(node.id, Arc::new(image));
        }
        images
            .remove(&graph.output)
            .ok_or(GraphExecutionError::MissingOutput(graph.output))
    }
}

fn execute_node(
    graph: &CompiledGraph,
    node: &GraphNode,
    images: &HashMap<GraphNodeId, Arc<FloatImage>>,
) -> Result<FloatImage, GraphExecutionError> {
    let required = |index: usize| -> Result<&FloatImage, GraphExecutionError> {
        let dependency = node.inputs[index].ok_or(GraphExecutionError::MissingInput {
            node: node.id,
            input: index,
        })?;
        images
            .get(&dependency)
            .map(AsRef::as_ref)
            .ok_or(GraphExecutionError::MissingDependency {
                node: node.id,
                dependency,
            })
    };
    let optional = |index: usize| -> Result<Option<&FloatImage>, GraphExecutionError> {
        node.inputs[index]
            .map(|dependency| {
                images.get(&dependency).map(AsRef::as_ref).ok_or(
                    GraphExecutionError::MissingDependency {
                        node: node.id,
                        dependency,
                    },
                )
            })
            .transpose()
    };

    match &node.operation {
        GraphOperation::Source(source) => source_texture(graph, *source),
        GraphOperation::Remap { points, bezier } => {
            Ok(map_rgb(required(0)?, "compositor-remap", |value| {
                remap_value(value, points, *bezier)
            }))
        }
        GraphOperation::Math {
            operation,
            constant,
        } => Ok(map_rgb(required(0)?, "compositor-math", |value| {
            arithmetic(value, *constant, *operation)
        })),
        GraphOperation::Algebra { program } => {
            let inputs = [optional(0)?, optional(1)?, optional(2)?];
            let reference = inputs.iter().flatten().next().copied();
            let width = reference.map_or(1, |image| image.width);
            let height = reference.map_or(1, |image| image.height);
            let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
            for y in 0..height {
                for x in 0..width {
                    for channel in 0..4 {
                        let variables = std::array::from_fn(|index| {
                            inputs[index]
                                .map(|image| sample(image, x, y, width, height)[channel])
                                .unwrap_or(0.0)
                        });
                        pixels.push(evaluate_algebra_program(program, variables).unwrap_or(0.0));
                    }
                }
            }
            Ok(FloatImage {
                name: "compositor-algebra".into(),
                width,
                height,
                pixels,
            })
        }
        GraphOperation::SharpThreshold { threshold } => Ok(map_rgb(
            required(0)?,
            "compositor-sharp-threshold",
            |value| if value < *threshold { 0.0 } else { 1.0 },
        )),
        GraphOperation::SmoothThreshold { threshold, width } => {
            let half_width = width.max(0.0) * 0.5;
            Ok(map_rgb(
                required(0)?,
                "compositor-smooth-threshold",
                |value| smoothstep(threshold - half_width, threshold + half_width, value),
            ))
        }
        GraphOperation::ImageFilter { filter, radius } => {
            Ok(filter_image(required(0)?, *filter, *radius))
        }
        GraphOperation::Combine {
            mode,
            operation,
            alpha,
        } => Ok(combine(
            required(0)?,
            required(1)?,
            optional(2)?,
            *mode,
            *operation,
            *alpha,
        )),
        GraphOperation::ColorSpace { from, to } => Ok(color_space(required(0)?, *from, *to)),
        GraphOperation::ExtractChannel { channel } => Ok(extract_channel(required(0)?, *channel)),
        GraphOperation::Grayscale { mode } => Ok(grayscale(required(0)?, *mode)),
        GraphOperation::JoinChannels => Ok(join_channels([
            optional(0)?,
            optional(1)?,
            optional(2)?,
            optional(3)?,
        ])),
        GraphOperation::ClampColor => {
            Ok(map_rgba(required(0)?, "compositor-color-output", |value| {
                value.clamp(0.0, 1.0)
            }))
        }
    }
}

fn source_texture(
    graph: &CompiledGraph,
    source_index: usize,
) -> Result<FloatImage, GraphExecutionError> {
    match graph
        .sources
        .get(source_index)
        .ok_or(GraphExecutionError::InvalidSource(source_index))?
    {
        GraphSource::Texture(texture) => {
            let expected = texture.width as usize * texture.height as usize * 4;
            if texture.pixels.len() != expected {
                return Err(GraphExecutionError::InvalidTexture {
                    source: source_index,
                    expected_bytes: expected,
                    actual_bytes: texture.pixels.len(),
                });
            }
            Ok(FloatImage {
                name: texture.name.clone(),
                width: texture.width,
                height: texture.height,
                pixels: texture
                    .pixels
                    .iter()
                    .map(|value| *value as f32 / 255.0)
                    .collect(),
            })
        }
        GraphSource::Constant(color) => Ok(FloatImage {
            name: "compositor-constant".into(),
            width: 1,
            height: 1,
            pixels: color.to_vec(),
        }),
    }
}

fn map_rgb(source: &FloatImage, name: &str, operation: impl Fn(f32) -> f32) -> FloatImage {
    let mut result = source.clone();
    result.name = name.into();
    for pixel in result.pixels.chunks_exact_mut(4) {
        for value in &mut pixel[..3] {
            *value = operation(*value);
        }
    }
    result
}

fn map_rgba(source: &FloatImage, name: &str, operation: impl Fn(f32) -> f32) -> FloatImage {
    let mut result = source.clone();
    result.name = name.into();
    result
        .pixels
        .iter_mut()
        .for_each(|value| *value = operation(*value));
    result
}

fn arithmetic(left: f32, right: f32, operation: usize) -> f32 {
    match operation {
        0 => left + right,
        1 => left - right,
        2 => left * right,
        3 if right.abs() > f32::EPSILON => left / right,
        3 => 0.0,
        4 => left.max(0.0).powf(right),
        5 => left.min(right),
        6 => left.max(right),
        _ => (left - right).abs(),
    }
}

pub(crate) fn remap_value(value: f32, points: &[[f32; 2]], bezier: bool) -> f32 {
    if points.is_empty() {
        return value;
    }
    if bezier && points.len() == 4 {
        let x = value.clamp(0.0, 1.0);
        let mut low = 0.0;
        let mut high = 1.0;
        for _ in 0..16 {
            let middle = (low + high) * 0.5;
            if cubic(points, middle)[0] < x {
                low = middle;
            } else {
                high = middle;
            }
        }
        return cubic(points, (low + high) * 0.5)[1];
    }
    let mut sorted = points.to_vec();
    sorted.sort_by(|a, b| a[0].total_cmp(&b[0]));
    if value <= sorted[0][0] {
        return sorted[0][1];
    }
    for segment in sorted.windows(2) {
        if value <= segment[1][0] {
            let width = segment[1][0] - segment[0][0];
            let t = if width.abs() <= f32::EPSILON {
                1.0
            } else {
                (value - segment[0][0]) / width
            };
            return segment[0][1] + t * (segment[1][1] - segment[0][1]);
        }
    }
    sorted.last().map_or(value, |point| point[1])
}

fn cubic(points: &[[f32; 2]], t: f32) -> [f32; 2] {
    let u = 1.0 - t;
    let weights = [u * u * u, 3.0 * u * u * t, 3.0 * u * t * t, t * t * t];
    let mut result = [0.0; 2];
    for (point, weight) in points.iter().zip(weights) {
        result[0] += point[0] * weight;
        result[1] += point[1] * weight;
    }
    result
}

fn smoothstep(edge0: f32, edge1: f32, value: f32) -> f32 {
    if (edge1 - edge0).abs() <= f32::EPSILON {
        return if value < edge0 { 0.0 } else { 1.0 };
    }
    let t = ((value - edge0) / (edge1 - edge0)).clamp(0.0, 1.0);
    t * t * (3.0 - 2.0 * t)
}

fn combine(
    left: &FloatImage,
    right: &FloatImage,
    alpha_texture: Option<&FloatImage>,
    mode: usize,
    operation: usize,
    fallback_alpha: f32,
) -> FloatImage {
    let width = left.width.max(right.width).max(1);
    let height = left.height.max(right.height).max(1);
    image_from_fn("compositor-combine", width, height, |x, y| {
        let a = sample(left, x, y, width, height);
        let b = sample(right, x, y, width, height);
        let alpha = alpha_texture
            .map(|texture| sample(texture, x, y, width, height)[0])
            .unwrap_or(fallback_alpha)
            .clamp(0.0, 1.0);
        std::array::from_fn(|channel| {
            if mode == 1 {
                alpha * a[channel] + (1.0 - alpha) * b[channel]
            } else {
                arithmetic(a[channel], b[channel], operation)
            }
        })
    })
}

fn color_space(source: &FloatImage, from: usize, to: usize) -> FloatImage {
    if from == to {
        let mut result = source.clone();
        result.name = "compositor-color-space".into();
        return result;
    }
    map_rgb(source, "compositor-color-space", |value| {
        if from == 0 {
            if value <= 0.04045 {
                value / 12.92
            } else {
                ((value + 0.055) / 1.055).powf(2.4)
            }
        } else if value <= 0.003_130_8 {
            value * 12.92
        } else {
            1.055 * value.powf(1.0 / 2.4) - 0.055
        }
    })
}

fn extract_channel(source: &FloatImage, channel: usize) -> FloatImage {
    let mut result = source.clone();
    result.name = "compositor-channel".into();
    for pixel in result.pixels.chunks_exact_mut(4) {
        let value = pixel[channel.min(3)];
        pixel.copy_from_slice(&[value, value, value, 1.0]);
    }
    result
}

fn grayscale(source: &FloatImage, mode: usize) -> FloatImage {
    let mut result = source.clone();
    result.name = "compositor-grayscale".into();
    for pixel in result.pixels.chunks_exact_mut(4) {
        let rgb = [pixel[0], pixel[1], pixel[2]];
        let gray = match mode {
            1 => (rgb[0] + rgb[1] + rgb[2]) / 3.0,
            2 => (rgb[0].max(rgb[1]).max(rgb[2]) + rgb[0].min(rgb[1]).min(rgb[2])) * 0.5,
            _ => 0.2126 * rgb[0] + 0.7152 * rgb[1] + 0.0722 * rgb[2],
        };
        pixel[..3].fill(gray);
    }
    result
}

fn join_channels(channels: [Option<&FloatImage>; 4]) -> FloatImage {
    let width = channels
        .iter()
        .flatten()
        .map(|texture| texture.width)
        .max()
        .unwrap_or(1)
        .max(1);
    let height = channels
        .iter()
        .flatten()
        .map(|texture| texture.height)
        .max()
        .unwrap_or(1)
        .max(1);
    image_from_fn("compositor-channel-join", width, height, |x, y| {
        std::array::from_fn(|channel| {
            channels[channel]
                .map(|texture| sample(texture, x, y, width, height)[0])
                .unwrap_or(if channel == 3 { 1.0 } else { 0.0 })
        })
    })
}

fn filter_image(source: &FloatImage, filter: usize, radius: f32) -> FloatImage {
    let radius = radius.round().clamp(0.0, 128.0) as i32;
    if radius == 0 {
        let mut result = source.clone();
        result.name = "compositor-filter".into();
        return result;
    }
    image_from_fn("compositor-filter", source.width, source.height, |x, y| {
        let original = texel(source, x as i32, y as i32);
        let mut output = match filter {
            0 => neighborhood_weighted(source, x, y, radius, true),
            1 => neighborhood_weighted(source, x, y, radius, false),
            2 => {
                let blur = neighborhood_weighted(source, x, y, radius, true);
                std::array::from_fn(|channel| {
                    if channel == 3 {
                        original[3]
                    } else {
                        original[channel] * 2.0 - blur[channel]
                    }
                })
            }
            3 => sobel(source, x, y),
            4 => neighborhood_order(source, x, y, radius, 0),
            5 => neighborhood_order(source, x, y, radius, 1),
            _ => neighborhood_order(source, x, y, radius, 2),
        };
        output[3] = original[3];
        output
    })
}

fn neighborhood_weighted(
    source: &FloatImage,
    x: u32,
    y: u32,
    radius: i32,
    gaussian: bool,
) -> [f32; 4] {
    let sigma = (radius as f32 * 0.5).max(0.5);
    let mut sum = [0.0; 4];
    let mut total = 0.0;
    for dy in -radius..=radius {
        for dx in -radius..=radius {
            let weight = if gaussian {
                (-(dx * dx + dy * dy) as f32 / (2.0 * sigma * sigma)).exp()
            } else {
                1.0
            };
            let pixel = texel(source, x as i32 + dx, y as i32 + dy);
            for channel in 0..4 {
                sum[channel] += pixel[channel] * weight;
            }
            total += weight;
        }
    }
    sum.map(|value| value / total.max(f32::EPSILON))
}

fn neighborhood_order(source: &FloatImage, x: u32, y: u32, radius: i32, mode: usize) -> [f32; 4] {
    std::array::from_fn(|channel| {
        if channel == 3 {
            return texel(source, x as i32, y as i32)[3];
        }
        let mut values = Vec::with_capacity(((2 * radius + 1) * (2 * radius + 1)) as usize);
        for dy in -radius..=radius {
            for dx in -radius..=radius {
                values.push(texel(source, x as i32 + dx, y as i32 + dy)[channel]);
            }
        }
        match mode {
            1 => values.into_iter().fold(f32::NEG_INFINITY, f32::max),
            2 => values.into_iter().fold(f32::INFINITY, f32::min),
            _ => {
                values.sort_by(f32::total_cmp);
                values[values.len() / 2]
            }
        }
    })
}

fn sobel(source: &FloatImage, x: u32, y: u32) -> [f32; 4] {
    const X: [[f32; 3]; 3] = [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]];
    const Y: [[f32; 3]; 3] = [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]];
    let mut gx = 0.0;
    let mut gy = 0.0;
    for row in 0..3 {
        for column in 0..3 {
            let pixel = texel(
                source,
                x as i32 + column as i32 - 1,
                y as i32 + row as i32 - 1,
            );
            let luminance = 0.2126 * pixel[0] + 0.7152 * pixel[1] + 0.0722 * pixel[2];
            gx += luminance * X[row][column];
            gy += luminance * Y[row][column];
        }
    }
    let edge = gx.hypot(gy);
    [edge, edge, edge, texel(source, x as i32, y as i32)[3]]
}

fn image_from_fn(
    name: &str,
    width: u32,
    height: u32,
    mut function: impl FnMut(u32, u32) -> [f32; 4],
) -> FloatImage {
    let mut pixels = Vec::with_capacity(width as usize * height as usize * 4);
    for y in 0..height {
        for x in 0..width {
            pixels.extend(function(x, y));
        }
    }
    FloatImage {
        name: name.into(),
        width,
        height,
        pixels,
    }
}

fn sample(source: &FloatImage, x: u32, y: u32, width: u32, height: u32) -> [f32; 4] {
    if source.width == 0 || source.height == 0 {
        return [0.0; 4];
    }
    let source_x = (((x as f32 + 0.5) * source.width as f32 / width as f32).floor() as u32)
        .min(source.width - 1);
    let source_y = (((y as f32 + 0.5) * source.height as f32 / height as f32).floor() as u32)
        .min(source.height - 1);
    texel(source, source_x as i32, source_y as i32)
}

fn texel(source: &FloatImage, x: i32, y: i32) -> [f32; 4] {
    if source.width == 0 || source.height == 0 {
        return [0.0; 4];
    }
    let x = x.clamp(0, source.width as i32 - 1) as usize;
    let y = y.clamp(0, source.height as i32 - 1) as usize;
    let offset = (y * source.width as usize + x) * 4;
    std::array::from_fn(|channel| source.pixels[offset + channel])
}

fn to_u8(value: f32) -> u8 {
    (value.clamp(0.0, 1.0) * 255.0).round() as u8
}

#[cfg(test)]
mod tests {
    use super::*;

    fn texture(width: u32, height: u32, pixels: &[[u8; 4]]) -> Arc<TextureAsset> {
        Arc::new(TextureAsset {
            name: "fixture".into(),
            width,
            height,
            pixels: pixels.iter().flatten().copied().collect(),
            cached_mips: Vec::new(),
        })
    }

    fn execute(
        sources: Vec<GraphSource>,
        nodes: Vec<GraphNode>,
        output: usize,
    ) -> Arc<TextureAsset> {
        let image = CpuGraphExecutor
            .execute(&CompiledGraph {
                generation: 1,
                sources,
                nodes,
                output,
            })
            .unwrap();
        Arc::new(image.to_texture_asset_clamped())
    }

    fn node(id: usize, operation: GraphOperation, inputs: &[usize]) -> GraphNode {
        let mut slots = [None; 4];
        for (slot, dependency) in inputs.iter().copied().enumerate() {
            slots[slot] = Some(dependency);
        }
        GraphNode {
            id,
            operation,
            inputs: slots,
        }
    }

    #[test]
    fn executes_dependency_first_math_threshold_and_remap() {
        let result = execute(
            vec![GraphSource::Texture(texture(1, 1, &[[64, 128, 192, 77]]))],
            vec![
                node(4, GraphOperation::Source(0), &[]),
                node(
                    8,
                    GraphOperation::Math {
                        operation: 2,
                        constant: 0.5,
                    },
                    &[4],
                ),
                node(9, GraphOperation::SharpThreshold { threshold: 0.3 }, &[8]),
            ],
            9,
        );
        assert_eq!(result.pixels, [0, 0, 255, 77]);

        let remapped = execute(
            vec![GraphSource::Constant([0.25, 0.5, 0.75, 1.0])],
            vec![
                node(0, GraphOperation::Source(0), &[]),
                node(
                    1,
                    GraphOperation::Remap {
                        points: vec![[0.0, 1.0], [1.0, 0.0]],
                        bezier: false,
                    },
                    &[0],
                ),
            ],
            1,
        );
        // Constants remain exact floats until the final display conversion.
        assert_eq!(remapped.pixels, [191, 128, 64, 255]);
    }

    #[test]
    fn math_preserves_unbounded_floats_until_color_boundary() {
        let graph = CompiledGraph {
            generation: 1,
            sources: vec![GraphSource::Constant([0.25, -3.0, 3.0, 1.0])],
            nodes: vec![
                node(0, GraphOperation::Source(0), &[]),
                node(
                    1,
                    GraphOperation::Math {
                        operation: 0,
                        constant: 2.0,
                    },
                    &[0],
                ),
                node(2, GraphOperation::ClampColor, &[1]),
            ],
            output: 1,
        };
        let unbounded = CpuGraphExecutor.execute(&graph).unwrap();
        assert_eq!(unbounded.pixels, [2.25, -1.0, 5.0, 1.0]);

        let bounded = CpuGraphExecutor
            .execute(&CompiledGraph { output: 2, ..graph })
            .unwrap();
        assert_eq!(bounded.pixels, [1.0, 0.0, 1.0, 1.0]);
    }

    #[test]
    fn algebra_compiles_precedence_functions_and_three_variables() {
        let program =
            compile_algebra_expression("x + y * z - abs(-2)").expect("valid algebra expression");
        let value = evaluate_algebra_program(&program, [1.0, 3.0, 4.0])
            .expect("compiled program evaluates");
        assert!((value - 11.0).abs() < 1.0e-6);
        assert!(compile_algebra_expression("x + unknown").is_err());
        assert!(compile_algebra_expression("(x + y").is_err());
    }

    #[test]
    fn algebra_sign_returns_negative_zero_and_positive_signs() {
        let program = compile_algebra_expression("sign(x)").expect("valid sign expression");
        assert_eq!(
            evaluate_algebra_program(&program, [-2.5, 0.0, 0.0]),
            Ok(-1.0)
        );
        assert_eq!(evaluate_algebra_program(&program, [0.0, 0.0, 0.0]), Ok(0.0));
        assert_eq!(evaluate_algebra_program(&program, [8.0, 0.0, 0.0]), Ok(1.0));
    }

    #[test]
    fn algebra_graph_resamples_inputs_and_defaults_missing_values_to_zero() {
        let graph = CompiledGraph {
            generation: 1,
            nodes: vec![
                node(0, GraphOperation::Source(0), &[]),
                GraphNode {
                    id: 1,
                    operation: GraphOperation::Algebra {
                        program: compile_algebra_expression("x + y + z").unwrap(),
                    },
                    inputs: [Some(0), None, None, None],
                },
            ],
            sources: vec![GraphSource::Constant([0.25, 0.5, 0.75, 1.0])],
            output: 1,
        };
        let image = CpuGraphExecutor.execute(&graph).expect("graph executes");
        assert_eq!(image.pixels, vec![0.25, 0.5, 0.75, 1.0]);
    }

    #[test]
    fn combines_mismatched_images_and_optional_alpha() {
        let result = execute(
            vec![
                GraphSource::Texture(texture(2, 1, &[[255, 0, 0, 255], [0, 0, 255, 255]])),
                GraphSource::Constant([0.0, 1.0, 0.0, 1.0]),
                GraphSource::Constant([0.25, 0.25, 0.25, 1.0]),
            ],
            vec![
                node(0, GraphOperation::Source(0), &[]),
                node(1, GraphOperation::Source(1), &[]),
                node(2, GraphOperation::Source(2), &[]),
                node(
                    3,
                    GraphOperation::Combine {
                        mode: 1,
                        operation: 0,
                        alpha: 1.0,
                    },
                    &[0, 1, 2],
                ),
            ],
            3,
        );
        assert_eq!(result.pixels, [64, 191, 0, 255, 0, 191, 64, 255]);
    }

    #[test]
    fn color_channels_grayscale_and_join_are_rgba_complete() {
        let result = execute(
            vec![GraphSource::Texture(texture(1, 1, &[[255, 128, 0, 64]]))],
            vec![
                node(0, GraphOperation::Source(0), &[]),
                node(1, GraphOperation::ExtractChannel { channel: 0 }, &[0]),
                node(2, GraphOperation::ExtractChannel { channel: 1 }, &[0]),
                node(3, GraphOperation::ExtractChannel { channel: 2 }, &[0]),
                node(4, GraphOperation::ExtractChannel { channel: 3 }, &[0]),
                node(5, GraphOperation::JoinChannels, &[1, 2, 3, 4]),
                node(6, GraphOperation::Grayscale { mode: 1 }, &[5]),
                node(7, GraphOperation::ColorSpace { from: 0, to: 1 }, &[6]),
            ],
            7,
        );
        assert_eq!(&result.pixels[3..4], &[64]);
        assert!(result.pixels[0] < 128);
        assert_eq!(result.pixels[0], result.pixels[1]);
        assert_eq!(result.pixels[1], result.pixels[2]);
    }

    #[test]
    fn all_image_filters_are_deterministic_and_preserve_alpha() {
        let source = texture(
            3,
            3,
            &[
                [0, 0, 0, 1],
                [0, 0, 0, 2],
                [0, 0, 0, 3],
                [0, 0, 0, 4],
                [255, 128, 64, 5],
                [0, 0, 0, 6],
                [0, 0, 0, 7],
                [0, 0, 0, 8],
                [0, 0, 0, 9],
            ],
        );
        for filter in 0..7 {
            let first = execute(
                vec![GraphSource::Texture(Arc::clone(&source))],
                vec![
                    node(0, GraphOperation::Source(0), &[]),
                    node(
                        1,
                        GraphOperation::ImageFilter {
                            filter,
                            radius: 1.0,
                        },
                        &[0],
                    ),
                ],
                1,
            );
            let second = execute(
                vec![GraphSource::Texture(Arc::clone(&source))],
                vec![
                    node(0, GraphOperation::Source(0), &[]),
                    node(
                        1,
                        GraphOperation::ImageFilter {
                            filter,
                            radius: 1.0,
                        },
                        &[0],
                    ),
                ],
                1,
            );
            assert_eq!(first, second, "filter {filter}");
            assert_eq!(first.pixels[4 * 4 + 3], 5, "filter {filter}");
        }
    }

    #[test]
    fn reports_structural_and_source_errors_without_panicking() {
        let missing = CpuGraphExecutor.execute(&CompiledGraph {
            generation: 0,
            sources: vec![],
            nodes: vec![node(
                2,
                GraphOperation::Math {
                    operation: 0,
                    constant: 1.0,
                },
                &[],
            )],
            output: 2,
        });
        assert_eq!(
            missing.unwrap_err(),
            GraphExecutionError::MissingInput { node: 2, input: 0 }
        );

        let invalid = CpuGraphExecutor.execute(&CompiledGraph {
            generation: 0,
            sources: vec![],
            nodes: vec![node(0, GraphOperation::Source(7), &[])],
            output: 0,
        });
        assert_eq!(invalid.unwrap_err(), GraphExecutionError::InvalidSource(7));
    }
}
