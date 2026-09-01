# zerofps-voxel

`zerofps-voxel` is a format- and engine-independent triangle mesh voxelizer.
Its public API consumes `[[[f32; 3]; 3]]` and produces a uniform grid whose
cells are classified as exterior, boundary, or interior.

The first CPU backend performs conservative triangle/AABB rasterization using
the separating axis theorem, followed by six-connected exterior flood fill.
Closed, watertight input is currently required for reliable solid filling.

```rust
use zerofps_voxel::{VoxelizeOptions, voxelize};

# let triangles = vec![[[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]]];
let grid = voxelize(&triangles, VoxelizeOptions {
    longest_axis_cells: 64,
    padding: 2,
})?;
println!("{:?}", grid.counts());
# Ok::<(), zerofps_voxel::VoxelizeError>(())
```

Run the mesh diagnostic example from the workspace root:

```text
cargo run -p zerofps-voxel --example voxelize_model -- path/to/model.stl 64
```

The test suite contains a procedural torus convergence test and, when the
local ignored fixture is present, an end-to-end test using
`appdata/models/stl/Torus.stl`.
