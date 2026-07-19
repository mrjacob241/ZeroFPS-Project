//! Polygon File Format (PLY) import.
//!
//! The importer deliberately follows the schema in the header instead of
//! assuming a fixed record layout. Unknown elements and properties are
//! consumed and ignored, which makes files written by scanners and modelling
//! tools usable without teaching the engine about all their metadata.

use crate::{ImportError, MeshAsset, Primitive, SourceInfo, Vertex};

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Encoding {
    Ascii,
    Little,
    Big,
}

#[derive(Clone, Copy, Debug)]
enum ScalarType {
    I8,
    U8,
    I16,
    U16,
    I32,
    U32,
    F32,
    F64,
}

impl ScalarType {
    fn parse(word: &str, line: usize) -> Result<Self, ImportError> {
        match word {
            "char" | "int8" => Ok(Self::I8),
            "uchar" | "uint8" => Ok(Self::U8),
            "short" | "int16" => Ok(Self::I16),
            "ushort" | "uint16" => Ok(Self::U16),
            "int" | "int32" => Ok(Self::I32),
            "uint" | "uint32" => Ok(Self::U32),
            "float" | "float32" => Ok(Self::F32),
            "double" | "float64" => Ok(Self::F64),
            _ => parse_error(
                format!("header line {line}"),
                format!("unknown scalar type `{word}`"),
            ),
        }
    }
}

#[derive(Clone, Debug)]
enum PropertyKind {
    Scalar(ScalarType),
    List(ScalarType, ScalarType),
}

#[derive(Clone, Debug)]
struct Property {
    name: String,
    kind: PropertyKind,
}

#[derive(Clone, Debug)]
struct Element {
    name: String,
    count: usize,
    properties: Vec<Property>,
}

struct Header {
    encoding: Encoding,
    elements: Vec<Element>,
    body_offset: usize,
}

#[derive(Clone, Copy, Debug)]
enum Number {
    Signed(i64),
    Unsigned(u64),
    Float(f64),
}

impl Number {
    fn finite_f32(self, location: &str) -> Result<f32, ImportError> {
        let value = match self {
            Self::Signed(value) => value as f64,
            Self::Unsigned(value) => value as f64,
            Self::Float(value) => value,
        };
        if !value.is_finite() || value < f32::MIN as f64 || value > f32::MAX as f64 {
            return parse_error(location, "number is not a finite 32-bit float");
        }
        Ok(value as f32)
    }

    fn usize(self, location: &str) -> Result<usize, ImportError> {
        let value = match self {
            Self::Signed(value) if value >= 0 => value as u64,
            Self::Unsigned(value) => value,
            _ => return parse_error(location, "expected a non-negative integer"),
        };
        usize::try_from(value)
            .map_err(|_| ply_error(location, "integer is too large for this platform"))
    }
}

/// Imports ASCII, binary little-endian, or binary big-endian PLY data.
pub fn import_ply(bytes: &[u8], source_name: &str) -> Result<MeshAsset, ImportError> {
    let header = parse_header(bytes)?;
    let mut reader = BodyReader::new(header.encoding, &bytes[header.body_offset..]);
    let mut vertices = Vec::new();
    let mut polygons = Vec::<Vec<u32>>::new();

    for element in &header.elements {
        for row in 0..element.count {
            let location = format!("element `{}` row {}", element.name, row + 1);
            match element.name.as_str() {
                "vertex" => vertices.push(read_vertex(&mut reader, element, &location)?),
                "face" => {
                    if let Some(indices) = read_face(&mut reader, element, &location)? {
                        polygons.push(indices);
                    }
                }
                _ => skip_record(&mut reader, element, &location)?,
            }
        }
    }

    if vertices.is_empty() {
        return Err(ImportError::InvalidData(
            "PLY contains no vertex records".into(),
        ));
    }

    let mut indices = Vec::new();
    for (polygon_index, polygon) in polygons.iter().enumerate() {
        if polygon.len() < 3 {
            return parse_error(
                format!("face {}", polygon_index + 1),
                "face must contain at least three vertices",
            );
        }
        for &index in polygon {
            if index as usize >= vertices.len() {
                return parse_error(
                    format!("face {}", polygon_index + 1),
                    format!(
                        "vertex index {index} is outside the {} imported vertices",
                        vertices.len()
                    ),
                );
            }
        }
        for corner in 1..polygon.len() - 1 {
            indices.extend_from_slice(&[polygon[0], polygon[corner], polygon[corner + 1]]);
        }
    }

    let asset = MeshAsset {
        name: source_name.to_owned(),
        vertices,
        primitives: vec![Primitive {
            name: "faces".into(),
            material: None,
            indices,
        }],
        source: SourceInfo {
            format: "ply".into(),
            path: None,
            ..SourceInfo::default()
        },
        ..MeshAsset::default()
    };
    asset.validate()?;
    Ok(asset)
}

fn parse_header(bytes: &[u8]) -> Result<Header, ImportError> {
    let mut offset = 0;
    let mut lines = Vec::new();
    let mut body_offset = None;
    while offset < bytes.len() {
        let end = bytes[offset..]
            .iter()
            .position(|byte| *byte == b'\n')
            .map_or(bytes.len(), |index| offset + index);
        let raw = bytes[offset..end]
            .strip_suffix(b"\r")
            .unwrap_or(&bytes[offset..end]);
        let line = std::str::from_utf8(raw)
            .map_err(|_| ply_error("header", "header is not valid UTF-8/ASCII"))?;
        lines.push(line);
        offset = if end < bytes.len() { end + 1 } else { end };
        if line.trim() == "end_header" {
            body_offset = Some(offset);
            break;
        }
        if lines.len() > 100_000 {
            return parse_error("header", "unreasonably long header");
        }
    }
    let body_offset =
        body_offset.ok_or_else(|| ply_error("header", "missing `end_header` declaration"))?;
    if lines.first().map(|line| line.trim()) != Some("ply") {
        return parse_error("header line 1", "expected `ply` magic");
    }

    let mut encoding = None;
    let mut elements: Vec<Element> = Vec::new();
    for (index, line) in lines.iter().enumerate().skip(1) {
        let line_number = index + 1;
        let words: Vec<_> = line.split_whitespace().collect();
        if words.is_empty() || matches!(words[0], "comment" | "obj_info") {
            continue;
        }
        match words[0] {
            "format" => {
                if words.len() != 3 || words[2] != "1.0" {
                    return parse_error(
                        format!("header line {line_number}"),
                        "expected `format <encoding> 1.0`",
                    );
                }
                if encoding.is_some() {
                    return parse_error(
                        format!("header line {line_number}"),
                        "duplicate format declaration",
                    );
                }
                encoding = Some(match words[1] {
                    "ascii" => Encoding::Ascii,
                    "binary_little_endian" => Encoding::Little,
                    "binary_big_endian" => Encoding::Big,
                    other => {
                        return parse_error(
                            format!("header line {line_number}"),
                            format!("unsupported encoding `{other}`"),
                        );
                    }
                });
            }
            "element" => {
                if words.len() != 3 {
                    return parse_error(
                        format!("header line {line_number}"),
                        "expected `element <name> <count>`",
                    );
                }
                let count = words[2].parse::<usize>().map_err(|_| {
                    ply_error(
                        format!("header line {line_number}"),
                        "invalid element count",
                    )
                })?;
                elements.push(Element {
                    name: words[1].into(),
                    count,
                    properties: Vec::new(),
                });
            }
            "property" => {
                let element = elements.last_mut().ok_or_else(|| {
                    ply_error(
                        format!("header line {line_number}"),
                        "property appears before any element",
                    )
                })?;
                let property = if words.get(1) == Some(&"list") {
                    if words.len() != 5 {
                        return parse_error(
                            format!("header line {line_number}"),
                            "expected `property list <count-type> <item-type> <name>`",
                        );
                    }
                    Property {
                        name: words[4].into(),
                        kind: PropertyKind::List(
                            ScalarType::parse(words[2], line_number)?,
                            ScalarType::parse(words[3], line_number)?,
                        ),
                    }
                } else {
                    if words.len() != 3 {
                        return parse_error(
                            format!("header line {line_number}"),
                            "expected `property <type> <name>`",
                        );
                    }
                    Property {
                        name: words[2].into(),
                        kind: PropertyKind::Scalar(ScalarType::parse(words[1], line_number)?),
                    }
                };
                if element
                    .properties
                    .iter()
                    .any(|old| old.name == property.name)
                {
                    return parse_error(
                        format!("header line {line_number}"),
                        format!("duplicate property `{}`", property.name),
                    );
                }
                element.properties.push(property);
            }
            "end_header" => break,
            directive => {
                return parse_error(
                    format!("header line {line_number}"),
                    format!("unknown header directive `{directive}`"),
                );
            }
        }
    }

    let encoding = encoding.ok_or_else(|| ply_error("header", "missing format declaration"))?;
    let vertex = elements
        .iter()
        .find(|element| element.name == "vertex")
        .ok_or_else(|| ply_error("header", "missing vertex element"))?;
    for required in ["x", "y", "z"] {
        match vertex
            .properties
            .iter()
            .find(|property| property.name == required)
        {
            Some(Property {
                kind: PropertyKind::Scalar(_),
                ..
            }) => {}
            Some(_) => {
                return parse_error(
                    "header",
                    format!("vertex property `{required}` must be scalar"),
                );
            }
            None => {
                return parse_error(
                    "header",
                    format!("vertex element is missing required `{required}` property"),
                );
            }
        }
    }

    Ok(Header {
        encoding,
        elements,
        body_offset,
    })
}

struct BodyReader<'a> {
    encoding: Encoding,
    bytes: &'a [u8],
    offset: usize,
}

impl<'a> BodyReader<'a> {
    fn new(encoding: Encoding, bytes: &'a [u8]) -> Self {
        Self {
            encoding,
            bytes,
            offset: 0,
        }
    }

    fn scalar(&mut self, ty: ScalarType, location: &str) -> Result<Number, ImportError> {
        if self.encoding == Encoding::Ascii {
            while self
                .bytes
                .get(self.offset)
                .is_some_and(u8::is_ascii_whitespace)
            {
                self.offset += 1;
            }
            let start = self.offset;
            while self
                .bytes
                .get(self.offset)
                .is_some_and(|byte| !byte.is_ascii_whitespace())
            {
                self.offset += 1;
            }
            if start == self.offset {
                return parse_error(location, "unexpected end of ASCII body");
            }
            let token = std::str::from_utf8(&self.bytes[start..self.offset])
                .map_err(|_| ply_error(location, "body token is not ASCII"))?;
            return parse_ascii_number(token, ty, location);
        }

        macro_rules! bytes {
            ($n:expr) => {{
                let end = self
                    .offset
                    .checked_add($n)
                    .ok_or_else(|| ply_error(location, "binary offset overflow"))?;
                let slice = self
                    .bytes
                    .get(self.offset..end)
                    .ok_or_else(|| ply_error(location, "unexpected end of binary body"))?;
                self.offset = end;
                slice
            }};
        }
        let little = self.encoding == Encoding::Little;
        Ok(match ty {
            ScalarType::I8 => Number::Signed(i8::from_ne_bytes([bytes!(1)[0]]) as i64),
            ScalarType::U8 => Number::Unsigned(bytes!(1)[0] as u64),
            ScalarType::I16 => {
                let data: [u8; 2] = bytes!(2).try_into().unwrap();
                Number::Signed(if little {
                    i16::from_le_bytes(data)
                } else {
                    i16::from_be_bytes(data)
                } as i64)
            }
            ScalarType::U16 => {
                let data: [u8; 2] = bytes!(2).try_into().unwrap();
                Number::Unsigned(if little {
                    u16::from_le_bytes(data)
                } else {
                    u16::from_be_bytes(data)
                } as u64)
            }
            ScalarType::I32 => {
                let data: [u8; 4] = bytes!(4).try_into().unwrap();
                Number::Signed(if little {
                    i32::from_le_bytes(data)
                } else {
                    i32::from_be_bytes(data)
                } as i64)
            }
            ScalarType::U32 => {
                let data: [u8; 4] = bytes!(4).try_into().unwrap();
                Number::Unsigned(if little {
                    u32::from_le_bytes(data)
                } else {
                    u32::from_be_bytes(data)
                } as u64)
            }
            ScalarType::F32 => {
                let data: [u8; 4] = bytes!(4).try_into().unwrap();
                Number::Float(if little {
                    f32::from_le_bytes(data)
                } else {
                    f32::from_be_bytes(data)
                } as f64)
            }
            ScalarType::F64 => {
                let data: [u8; 8] = bytes!(8).try_into().unwrap();
                Number::Float(if little {
                    f64::from_le_bytes(data)
                } else {
                    f64::from_be_bytes(data)
                })
            }
        })
    }

    fn property(
        &mut self,
        property: &Property,
        location: &str,
    ) -> Result<Vec<Number>, ImportError> {
        match property.kind {
            PropertyKind::Scalar(ty) => Ok(vec![self.scalar(ty, location)?]),
            PropertyKind::List(count_ty, item_ty) => {
                let count = self.scalar(count_ty, location)?.usize(location)?;
                // Prevent hostile counts from allocating huge vectors or looping
                // long after a truncated input has already become impossible.
                if count > 100_000_000 || count > self.bytes.len().saturating_add(1) {
                    return parse_error(location, format!("unreasonable list length {count}"));
                }
                let mut values = Vec::with_capacity(count);
                for _ in 0..count {
                    values.push(self.scalar(item_ty, location)?);
                }
                Ok(values)
            }
        }
    }
}

fn parse_ascii_number(token: &str, ty: ScalarType, location: &str) -> Result<Number, ImportError> {
    let bad = || {
        ply_error(
            location,
            format!("`{token}` does not match declared scalar type"),
        )
    };
    match ty {
        ScalarType::I8 => token
            .parse::<i8>()
            .map(|value| Number::Signed(value as i64))
            .map_err(|_| bad()),
        ScalarType::U8 => token
            .parse::<u8>()
            .map(|value| Number::Unsigned(value as u64))
            .map_err(|_| bad()),
        ScalarType::I16 => token
            .parse::<i16>()
            .map(|value| Number::Signed(value as i64))
            .map_err(|_| bad()),
        ScalarType::U16 => token
            .parse::<u16>()
            .map(|value| Number::Unsigned(value as u64))
            .map_err(|_| bad()),
        ScalarType::I32 => token
            .parse::<i32>()
            .map(|value| Number::Signed(value as i64))
            .map_err(|_| bad()),
        ScalarType::U32 => token
            .parse::<u32>()
            .map(|value| Number::Unsigned(value as u64))
            .map_err(|_| bad()),
        ScalarType::F32 => token
            .parse::<f32>()
            .map(|value| Number::Float(value as f64))
            .map_err(|_| bad()),
        ScalarType::F64 => token.parse::<f64>().map(Number::Float).map_err(|_| bad()),
    }
}

fn read_vertex(
    reader: &mut BodyReader<'_>,
    element: &Element,
    location: &str,
) -> Result<Vertex, ImportError> {
    let mut vertex = Vertex {
        color: [1.0; 4],
        ..Vertex::default()
    };
    for property in &element.properties {
        let values = reader.property(property, location)?;
        let value = values.first().copied();
        let scalar = || {
            value.ok_or_else(|| ply_error(location, format!("`{}` must be scalar", property.name)))
        };
        match property.name.as_str() {
            "x" => vertex.position[0] = scalar()?.finite_f32(location)?,
            "y" => vertex.position[1] = scalar()?.finite_f32(location)?,
            "z" => vertex.position[2] = scalar()?.finite_f32(location)?,
            "nx" => vertex.normal[0] = scalar()?.finite_f32(location)?,
            "ny" => vertex.normal[1] = scalar()?.finite_f32(location)?,
            "nz" => vertex.normal[2] = scalar()?.finite_f32(location)?,
            "u" | "s" | "texture_u" => vertex.uv[0] = scalar()?.finite_f32(location)?,
            "v" | "t" | "texture_v" => vertex.uv[1] = scalar()?.finite_f32(location)?,
            "red" | "r" => vertex.color[0] = color_value(scalar()?, property, location)?,
            "green" | "g" => vertex.color[1] = color_value(scalar()?, property, location)?,
            "blue" | "b" => vertex.color[2] = color_value(scalar()?, property, location)?,
            "alpha" | "a" => vertex.color[3] = color_value(scalar()?, property, location)?,
            _ => {}
        }
    }
    Ok(vertex)
}

fn color_value(number: Number, property: &Property, location: &str) -> Result<f32, ImportError> {
    let value = number.finite_f32(location)?;
    let normalized = match property.kind {
        PropertyKind::Scalar(ScalarType::U8) => value / 255.0,
        PropertyKind::Scalar(ScalarType::U16) => value / 65535.0,
        _ => value,
    };
    if !(0.0..=1.0).contains(&normalized) {
        return parse_error(
            location,
            format!(
                "color property `{}` is outside its valid range",
                property.name
            ),
        );
    }
    Ok(normalized)
}

fn read_face(
    reader: &mut BodyReader<'_>,
    element: &Element,
    location: &str,
) -> Result<Option<Vec<u32>>, ImportError> {
    let mut indices = None;
    for property in &element.properties {
        let values = reader.property(property, location)?;
        if matches!(property.name.as_str(), "vertex_indices" | "vertex_index") {
            if !matches!(property.kind, PropertyKind::List(_, _)) {
                return parse_error(location, "face vertex indices must be a list property");
            }
            let mut converted = Vec::with_capacity(values.len());
            for value in values {
                let index = value.usize(location)?;
                converted.push(u32::try_from(index).map_err(|_| {
                    ply_error(location, "vertex index exceeds the supported u32 range")
                })?);
            }
            indices = Some(converted);
        }
    }
    Ok(indices)
}

fn skip_record(
    reader: &mut BodyReader<'_>,
    element: &Element,
    location: &str,
) -> Result<(), ImportError> {
    for property in &element.properties {
        reader.property(property, location)?;
    }
    Ok(())
}

fn parse_error<T>(
    location: impl Into<String>,
    message: impl Into<String>,
) -> Result<T, ImportError> {
    Err(ply_error(location, message))
}

fn ply_error(location: impl Into<String>, message: impl Into<String>) -> ImportError {
    ImportError::Parse {
        format: "PLY",
        location: location.into(),
        message: message.into(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn imports_ascii_attributes_and_triangulates_polygon() {
        let data = b"ply
format ascii 1.0
comment a useful comment
element vertex 4
property float x
property float y
property float z
property float nx
property float ny
property float nz
property float texture_u
property float texture_v
property uchar red
property uchar green
property uchar blue
element face 1
property list uchar int vertex_indices
end_header
0 0 0 0 0 1 0 0 255 0 0
1 0 0 0 0 1 1 0 0 255 0
1 1 0 0 0 1 1 1 0 0 255
0 1 0 0 0 1 0 1 255 255 255
4 0 1 2 3
";
        let asset = import_ply(data, "quad.ply").unwrap();
        assert_eq!(asset.vertices.len(), 4);
        assert_eq!(asset.primitives[0].indices, [0, 1, 2, 0, 2, 3]);
        assert_eq!(asset.vertices[0].normal, [0.0, 0.0, 1.0]);
        assert_eq!(asset.vertices[0].color, [1.0, 0.0, 0.0, 1.0]);
        assert_eq!(asset.vertices[2].uv, [1.0, 1.0]);
    }

    #[test]
    fn imports_little_endian_and_skips_unknown_element() {
        let header = b"ply
format binary_little_endian 1.0
element vertex 3
property float x
property float y
property float z
element edge 1
property int vertex1
property int vertex2
element face 1
property list uchar uint vertex_indices
end_header
";
        let mut data = header.to_vec();
        for value in [0.0_f32, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            data.extend(value.to_le_bytes());
        }
        data.extend(0_i32.to_le_bytes());
        data.extend(1_i32.to_le_bytes());
        data.push(3);
        for index in [0_u32, 1, 2] {
            data.extend(index.to_le_bytes());
        }
        let asset = import_ply(&data, "binary.ply").unwrap();
        assert_eq!(asset.triangle_count(), 1);
        assert_eq!(asset.vertices[1].position, [1.0, 0.0, 0.0]);
    }

    #[test]
    fn imports_big_endian() {
        let header = b"ply
format binary_big_endian 1.0
element vertex 3
property double x
property double y
property double z
element face 1
property list uchar ushort vertex_indices
end_header
";
        let mut data = header.to_vec();
        for value in [0.0_f64, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 0.0] {
            data.extend(value.to_be_bytes());
        }
        data.push(3);
        for index in [0_u16, 1, 2] {
            data.extend(index.to_be_bytes());
        }
        assert_eq!(import_ply(&data, "be.ply").unwrap().triangle_count(), 1);
    }

    #[test]
    fn rejects_malformed_headers_and_out_of_bounds_faces() {
        assert!(import_ply(b"not ply\n", "bad.ply").is_err());
        assert!(import_ply(
            b"ply\nformat ascii 1.0\nelement vertex 1\nproperty float x\nproperty float y\nend_header\n0 0\n",
            "missing-z.ply"
        )
        .is_err());
        assert!(import_ply(
            b"ply\nformat ascii 1.0\nelement vertex 3\nproperty float x\nproperty float y\nproperty float z\nelement face 1\nproperty list uchar int vertex_indices\nend_header\n0 0 0\n1 0 0\n0 1 0\n3 0 1 99\n",
            "bounds.ply"
        )
        .is_err());
    }

    #[test]
    fn rejects_truncated_binary_and_absurd_list_counts() {
        let truncated = b"ply
format binary_little_endian 1.0
element vertex 1
property float x
property float y
property float z
end_header
\0\0";
        assert!(import_ply(truncated, "short.ply").is_err());

        let absurd = b"ply
format ascii 1.0
element vertex 3
property float x
property float y
property float z
element face 1
property list uint int vertex_indices
end_header
0 0 0
1 0 0
0 1 0
4294967295
";
        assert!(import_ply(absurd, "absurd.ply").is_err());
    }
}
