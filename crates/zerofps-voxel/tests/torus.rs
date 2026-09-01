use std::path::PathBuf;

use zerofps_assets::import_file;
use zerofps_voxel::{Cell, Triangle, VoxelizeOptions, voxelize};

#[test]
fn downloaded_torus_preserves_central_hole_and_fills_tube() {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("../../appdata/models/stl/Torus.stl");
    if !path.is_file() {
        eprintln!(
            "skipping local fixture absent from this checkout: {}",
            path.display()
        );
        return;
    }
    let asset = import_file(&path).expect("import Torus.stl");
    let triangles = asset_triangles(&asset);
    let grid = voxelize(
        &triangles,
        VoxelizeOptions {
            longest_axis_cells: 64,
            padding: 2,
        },
    )
    .expect("voxelize Torus.stl");

    let counts = grid.counts();
    assert!(counts.interior > 0, "torus tube should contain solid cells");
    assert_eq!(
        grid.sample([0.0, 0.0, 0.0]),
        Some(Cell::Exterior),
        "the central torus hole must remain connected to the exterior"
    );
    assert_eq!(
        grid.sample([0.67, 0.0, 0.0]),
        Some(Cell::Interior),
        "a point near the center of the torus tube must be solid"
    );
    let analytic_volume = std::f64::consts::TAU * std::f64::consts::PI * 0.67 * 0.31_f64.powi(2);
    let relative_error = (grid.occupied_volume() - analytic_volume).abs() / analytic_volume;
    assert!(
        relative_error < 0.35,
        "voxel volume error {:.1}%",
        relative_error * 100.0
    );
}

fn asset_triangles(asset: &zerofps_assets::MeshAsset) -> Vec<Triangle> {
    asset
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
        .collect()
}
