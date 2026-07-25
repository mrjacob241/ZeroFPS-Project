use std::path::Path;

use zerofps_assets::{autofix_mesh, import_file};

fn main() {
    for path in std::env::args().skip(1) {
        match inspect(Path::new(&path)) {
            Ok(()) => {}
            Err(error) => eprintln!("{path}: {error}"),
        }
    }
}

fn inspect(path: &Path) -> Result<(), Box<dyn std::error::Error>> {
    let source = import_file(path)?;
    let (mesh, _) = autofix_mesh(&source);
    let mut triangles = 0usize;
    let mut opposed = 0usize;
    let mut zero_normals = 0usize;
    let mut inward_faces = 0usize;
    let mut signed_volume = 0.0_f64;
    let center = mesh.vertices.iter().fold([0.0; 3], |mut sum, vertex| {
        for axis in 0..3 {
            sum[axis] += vertex.position[axis] / mesh.vertices.len().max(1) as f32;
        }
        sum
    });
    let mut minimum_dot = 1.0_f32;
    for primitive in &mesh.primitives {
        for triangle in primitive.indices.chunks_exact(3) {
            let vertices = [
                mesh.vertices[triangle[0] as usize],
                mesh.vertices[triangle[1] as usize],
                mesh.vertices[triangle[2] as usize],
            ];
            let edge_a = subtract(vertices[1].position, vertices[0].position);
            let edge_b = subtract(vertices[2].position, vertices[0].position);
            let face = normalize(cross(edge_a, edge_b));
            let average = normalize([
                vertices.iter().map(|vertex| vertex.normal[0]).sum(),
                vertices.iter().map(|vertex| vertex.normal[1]).sum(),
                vertices.iter().map(|vertex| vertex.normal[2]).sum(),
            ]);
            let alignment = dot(face, average);
            let face_center = [
                vertices
                    .iter()
                    .map(|vertex| vertex.position[0])
                    .sum::<f32>()
                    / 3.0,
                vertices
                    .iter()
                    .map(|vertex| vertex.position[1])
                    .sum::<f32>()
                    / 3.0,
                vertices
                    .iter()
                    .map(|vertex| vertex.position[2])
                    .sum::<f32>()
                    / 3.0,
            ];
            triangles += 1;
            zero_normals += usize::from(dot(average, average) < 0.5);
            opposed += usize::from(alignment < 0.0);
            inward_faces += usize::from(dot(face, subtract(face_center, center)) < 0.0);
            signed_volume += dot(
                vertices[0].position,
                cross(vertices[1].position, vertices[2].position),
            ) as f64
                / 6.0;
            minimum_dot = minimum_dot.min(alignment);
        }
    }
    println!(
        "{} (autofixed): {triangles} triangles, {opposed} opposed face/vertex normals, \
         {zero_normals} zero-normal faces, {inward_faces} centroid-inward faces, \
         signed volume {signed_volume:.4}, minimum alignment {minimum_dot:.4}",
        path.display()
    );
    Ok(())
}

fn subtract(left: [f32; 3], right: [f32; 3]) -> [f32; 3] {
    [left[0] - right[0], left[1] - right[1], left[2] - right[2]]
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

fn normalize(value: [f32; 3]) -> [f32; 3] {
    let length = dot(value, value).sqrt();
    if length <= f32::EPSILON {
        [0.0; 3]
    } else {
        value.map(|component| component / length)
    }
}
