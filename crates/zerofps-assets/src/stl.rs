//! Stereolithography (STL) import.
//!
//! STL has no magic number: an ASCII file starts with `solid`, but a binary
//! header is allowed to contain the same word. We consequently recognize a
//! binary file by validating its facet count against the byte length.

use std::collections::BTreeMap;

use crate::{ImportError, MeshAsset, Primitive, SourceInfo, Vertex, parse_f32};

const BINARY_HEADER_LEN: usize = 84;
const BINARY_FACET_LEN: usize = 50;

pub fn import_stl(bytes: &[u8], source_name: &str) -> Result<MeshAsset, ImportError> {
    if let Some((count, expected)) = binary_layout(bytes)? {
        // An exact binary layout is authoritative, including when its header
        // begins with "solid". Non-UTF-8 data is binary even with trailing
        // vendor bytes.
        if expected == bytes.len() || std::str::from_utf8(bytes).is_err() {
            return import_binary(bytes, source_name, count, expected);
        }
    }

    let ascii_text = std::str::from_utf8(bytes)
        .ok()
        .filter(|text| text.trim_start().to_ascii_lowercase().starts_with("solid"));
    if let Some(text) = ascii_text {
        return import_ascii(text, source_name);
    }

    if let Some((count, expected)) = binary_layout(bytes)? {
        return import_binary(bytes, source_name, count, expected);
    }

    Err(parse_error(
        "header",
        "not a complete binary STL or an ASCII STL beginning with `solid`",
    ))
}

/// Returns a plausible complete binary layout. An impossible declared size is
/// reported as truncation rather than falling through to a confusing ASCII
/// error (unless this is recognizably ASCII).
fn binary_layout(bytes: &[u8]) -> Result<Option<(usize, usize)>, ImportError> {
    if bytes.len() < BINARY_HEADER_LEN {
        return Ok(None);
    }
    let count = u32::from_le_bytes(bytes[80..84].try_into().expect("fixed slice")) as usize;
    let payload = count
        .checked_mul(BINARY_FACET_LEN)
        .ok_or_else(|| invalid("binary STL facet count overflows the address space"))?;
    let expected = BINARY_HEADER_LEN
        .checked_add(payload)
        .ok_or_else(|| invalid("binary STL byte length overflows the address space"))?;
    if expected > bytes.len() {
        let ascii = std::str::from_utf8(bytes)
            .ok()
            .is_some_and(|text| text.trim_start().to_ascii_lowercase().starts_with("solid"));
        if ascii {
            Ok(None)
        } else {
            Err(parse_error(
                "facet data",
                format!(
                    "truncated binary STL: {count} facets require {expected} bytes, found {}",
                    bytes.len()
                ),
            ))
        }
    } else {
        Ok(Some((count, expected)))
    }
}

fn import_binary(
    bytes: &[u8],
    source_name: &str,
    count: usize,
    expected: usize,
) -> Result<MeshAsset, ImportError> {
    if count == 0 {
        return Err(invalid("STL contains no facets"));
    }
    let mut builder = Builder::new(source_name, count)?;
    for facet in 0..count {
        let start = BINARY_HEADER_LEN + facet * BINARY_FACET_LEN;
        let values = &bytes[start..start + 48];
        let mut floats = [0.0; 12];
        for (index, value) in floats.iter_mut().enumerate() {
            let offset = index * 4;
            *value = f32::from_le_bytes(
                values[offset..offset + 4]
                    .try_into()
                    .expect("four-byte float"),
            );
            if !value.is_finite() {
                return Err(invalid(format!(
                    "binary STL facet {facet} contains a non-finite float"
                )));
            }
        }
        builder.push_facet(
            [floats[0], floats[1], floats[2]],
            [
                [floats[3], floats[4], floats[5]],
                [floats[6], floats[7], floats[8]],
                [floats[9], floats[10], floats[11]],
            ],
            format!("binary facet {facet}"),
        )?;
    }
    let mut asset = builder.finish()?;
    if expected < bytes.len() {
        asset.warnings.push(format!(
            "ignored {} trailing bytes after binary STL facets",
            bytes.len() - expected
        ));
    }
    Ok(asset)
}

fn import_ascii(text: &str, source_name: &str) -> Result<MeshAsset, ImportError> {
    let lines: Vec<(usize, &str)> = text
        .lines()
        .enumerate()
        .filter_map(|(number, line)| {
            let line = line.trim();
            (!line.is_empty()).then_some((number + 1, line))
        })
        .collect();
    if lines.is_empty() || !keyword(lines[0].1, "solid") {
        return Err(parse_error("line 1", "expected `solid`"));
    }

    let mut cursor = 1;
    let mut facets = Vec::new();
    let mut ended = false;
    while cursor < lines.len() {
        let (line_no, line) = lines[cursor];
        if keyword(line, "endsolid") {
            cursor += 1;
            ended = true;
            break;
        }
        let normal = parse_prefixed_vec3(line, "facet normal", line_no)?;
        cursor += 1;
        expect_line(&lines, cursor, "outer loop")?;
        cursor += 1;
        let mut vertices = [[0.0; 3]; 3];
        for vertex in &mut vertices {
            let (number, line) = lines
                .get(cursor)
                .copied()
                .ok_or_else(|| parse_error("end of file", "expected three `vertex` lines"))?;
            *vertex = parse_prefixed_vec3(line, "vertex", number)?;
            cursor += 1;
        }
        expect_line(&lines, cursor, "endloop")?;
        cursor += 1;
        expect_line(&lines, cursor, "endfacet")?;
        cursor += 1;
        facets.push((normal, vertices, line_no));
    }
    if !ended {
        return Err(parse_error("end of file", "expected `endsolid`"));
    }
    if cursor < lines.len() {
        return Err(parse_error(
            format!("line {}", lines[cursor].0),
            "unexpected content after `endsolid`",
        ));
    }
    if facets.is_empty() {
        return Err(invalid("STL contains no facets"));
    }
    let mut builder = Builder::new(source_name, facets.len())?;
    for (normal, vertices, line) in facets {
        builder.push_facet(normal, vertices, format!("line {line}"))?;
    }
    builder.finish()
}

fn expect_line(
    lines: &[(usize, &str)],
    cursor: usize,
    expected: &'static str,
) -> Result<(), ImportError> {
    let (number, line) = lines
        .get(cursor)
        .copied()
        .ok_or_else(|| parse_error("end of file", format!("expected `{expected}`")))?;
    if line.eq_ignore_ascii_case(expected) {
        Ok(())
    } else {
        Err(parse_error(
            format!("line {number}"),
            format!("expected `{expected}`, found `{line}`"),
        ))
    }
}

fn parse_prefixed_vec3(
    line: &str,
    prefix: &'static str,
    line_no: usize,
) -> Result<[f32; 3], ImportError> {
    let words: Vec<&str> = line.split_whitespace().collect();
    let prefix_words: Vec<&str> = prefix.split_whitespace().collect();
    if words.len() != prefix_words.len() + 3
        || !words[..prefix_words.len()]
            .iter()
            .zip(prefix_words)
            .all(|(actual, expected)| actual.eq_ignore_ascii_case(expected))
    {
        return Err(parse_error(
            format!("line {line_no}"),
            format!("expected `{prefix} x y z`"),
        ));
    }
    Ok([
        parse_f32(words[words.len() - 3], "STL", format!("line {line_no}"))?,
        parse_f32(words[words.len() - 2], "STL", format!("line {line_no}"))?,
        parse_f32(words[words.len() - 1], "STL", format!("line {line_no}"))?,
    ])
}

fn keyword(line: &str, expected: &str) -> bool {
    line.split_whitespace()
        .next()
        .is_some_and(|word| word.eq_ignore_ascii_case(expected))
}

struct Builder {
    asset: MeshAsset,
    indices: Vec<u32>,
    vertices: BTreeMap<[u32; 6], u32>,
}

impl Builder {
    fn new(source_name: &str, facets: usize) -> Result<Self, ImportError> {
        let index_capacity = facets
            .checked_mul(3)
            .ok_or_else(|| invalid("STL vertex count overflows the address space"))?;
        if index_capacity > u32::MAX as usize {
            return Err(invalid("STL has more vertices than 32-bit indices support"));
        }
        Ok(Self {
            asset: MeshAsset {
                name: source_name.into(),
                source: SourceInfo {
                    format: "stl".into(),
                    path: None,
                    ..SourceInfo::default()
                },
                ..MeshAsset::default()
            },
            indices: Vec::with_capacity(index_capacity),
            vertices: BTreeMap::new(),
        })
    }

    fn push_facet(
        &mut self,
        supplied_normal: [f32; 3],
        positions: [[f32; 3]; 3],
        location: String,
    ) -> Result<(), ImportError> {
        if supplied_normal
            .iter()
            .chain(positions.iter().flatten())
            .any(|v| !v.is_finite())
        {
            return Err(invalid(format!("{location} contains a non-finite value")));
        }
        let geometric = cross(
            sub(positions[1], positions[0]),
            sub(positions[2], positions[0]),
        );
        let normal = normalize(supplied_normal)
            .or_else(|| normalize(geometric))
            .ok_or_else(|| invalid(format!("{location} is degenerate and has no valid normal")))?;

        for position in positions {
            let key = [
                position[0].to_bits(),
                position[1].to_bits(),
                position[2].to_bits(),
                normal[0].to_bits(),
                normal[1].to_bits(),
                normal[2].to_bits(),
            ];
            let index = if let Some(index) = self.vertices.get(&key) {
                *index
            } else {
                let index = u32::try_from(self.asset.vertices.len())
                    .map_err(|_| invalid("STL has more vertices than 32-bit indices support"))?;
                self.asset.vertices.push(Vertex {
                    position,
                    normal,
                    uv: [0.0; 2],
                    color: [1.0; 4],
                });
                self.vertices.insert(key, index);
                index
            };
            self.indices.push(index);
        }
        Ok(())
    }

    fn finish(mut self) -> Result<MeshAsset, ImportError> {
        self.asset.primitives.push(Primitive {
            name: "facets".into(),
            material: None,
            indices: self.indices,
        });
        self.asset.validate()?;
        Ok(self.asset)
    }
}

fn sub(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [a[0] - b[0], a[1] - b[1], a[2] - b[2]]
}

fn cross(a: [f32; 3], b: [f32; 3]) -> [f32; 3] {
    [
        a[1] * b[2] - a[2] * b[1],
        a[2] * b[0] - a[0] * b[2],
        a[0] * b[1] - a[1] * b[0],
    ]
}

fn normalize(value: [f32; 3]) -> Option<[f32; 3]> {
    let length_squared = value.iter().map(|v| v * v).sum::<f32>();
    if !length_squared.is_finite() || length_squared <= f32::EPSILON {
        return None;
    }
    let inverse = length_squared.sqrt().recip();
    Some([value[0] * inverse, value[1] * inverse, value[2] * inverse])
}

fn parse_error(location: impl Into<String>, message: impl Into<String>) -> ImportError {
    ImportError::Parse {
        format: "STL",
        location: location.into(),
        message: message.into(),
    }
}

fn invalid(message: impl Into<String>) -> ImportError {
    ImportError::InvalidData(message.into())
}

#[cfg(test)]
mod tests {
    use super::*;

    const ASCII: &str = "solid test
facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet
endsolid test
";

    fn binary(header: &[u8]) -> Vec<u8> {
        let mut bytes = vec![0; 84 + 50];
        bytes[..header.len()].copy_from_slice(header);
        bytes[80..84].copy_from_slice(&1_u32.to_le_bytes());
        let values = [0., 0., 1., 0., 0., 0., 1., 0., 0., 0., 1., 0.];
        for (index, value) in values.iter().enumerate() {
            let start = 84 + index * 4;
            bytes[start..start + 4].copy_from_slice(&f32::to_le_bytes(*value));
        }
        bytes
    }

    #[test]
    fn imports_ascii_stl() {
        let mesh = import_stl(ASCII.as_bytes(), "triangle.stl").unwrap();
        assert_eq!(mesh.vertices.len(), 3);
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.vertices[0].normal, [0.0, 0.0, 1.0]);
    }

    #[test]
    fn imports_binary_stl() {
        let mesh = import_stl(&binary(b"binary triangle"), "triangle.stl").unwrap();
        assert_eq!(mesh.triangle_count(), 1);
        assert_eq!(mesh.vertices.len(), 3);
    }

    #[test]
    fn binary_layout_wins_when_header_starts_with_solid() {
        let mesh = import_stl(&binary(b"solid definitely binary"), "ambiguous.stl").unwrap();
        assert_eq!(mesh.triangle_count(), 1);
    }

    #[test]
    fn rejects_truncated_binary() {
        let mut bytes = binary(b"binary truncated");
        bytes.truncate(bytes.len() - 4);
        assert!(import_stl(&bytes, "bad.stl").is_err());
    }

    #[test]
    fn rejects_malformed_ascii_and_non_finite_values() {
        let malformed = ASCII.replace("endloop", "not_endloop");
        assert!(import_stl(malformed.as_bytes(), "bad.stl").is_err());
        let non_finite = ASCII.replace("vertex 0 0 0", "vertex NaN 0 0");
        assert!(import_stl(non_finite.as_bytes(), "bad.stl").is_err());
    }

    #[test]
    fn deduplicates_only_when_position_and_facet_normal_match() {
        let facet = "facet normal 0 0 1
 outer loop
  vertex 0 0 0
  vertex 1 0 0
  vertex 0 1 0
 endloop
endfacet";
        let two = format!("solid test\n{facet}\n{facet}\nendsolid test\n");
        let mesh = import_stl(two.as_bytes(), "two.stl").unwrap();
        assert_eq!(mesh.triangle_count(), 2);
        assert_eq!(mesh.vertices.len(), 3);
    }
}
