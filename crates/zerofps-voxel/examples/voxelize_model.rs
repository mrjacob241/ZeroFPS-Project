use std::path::PathBuf;

use zerofps_assets::import_file;
use zerofps_voxel::{Triangle, VoxelizeOptions, voxelize};

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let path = std::env::args_os()
        .nth(1)
        .map(PathBuf::from)
        .ok_or("usage: voxelize_model <mesh> [longest-axis-cells]")?;
    let resolution = std::env::args()
        .nth(2)
        .map(|value| value.parse())
        .transpose()?
        .unwrap_or(64);
    let asset = import_file(&path)?;
    let triangles: Vec<Triangle> = asset
        .primitives
        .iter()
        .flat_map(|primitive| primitive.indices.chunks_exact(3))
        .map(|indices| {
            [
                asset.vertices[indices[0] as usize].position,
                asset.vertices[indices[1] as usize].position,
                asset.vertices[indices[2] as usize].position,
            ]
        })
        .collect();
    let grid = voxelize(
        &triangles,
        VoxelizeOptions {
            longest_axis_cells: resolution,
            padding: 2,
        },
    )?;
    let counts = grid.counts();
    println!("source: {}", path.display());
    println!("triangles: {}", triangles.len());
    println!("dimensions: {:?}", grid.dimensions);
    println!("cell size: {:.6}", grid.cell_size);
    println!("exterior cells: {}", counts.exterior);
    println!("boundary cells: {}", counts.boundary);
    println!("interior cells: {}", counts.interior);
    println!("occupied volume: {:.6}", grid.occupied_volume());
    Ok(())
}
