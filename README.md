# ⚠️ ZeroFPS-Project ⚠️ (WARNING: Vibe Coded Project)
## Rendering at the speed of static images.

This project is born to create a game engine from scratch in Rust, but in the process has created a nice static postcard simulator...

![ZeroFPS Project editor viewport](panels/clip.gif)

Long term objective of the ZeroFPS Project is to add a dynamic texture modification system based on nodes compositing and a Rust-based scripting language.

![ZeroFPS Project editor compositing](panels/clip_globe.gif)

## Goals

- [x] Build a responsive Rust engine/editor with isolated input, rendering, import, and dialog work.
- [x] Provide a 3D scene editor backed by a shared hierarchical transform and cascading-attribute model.
- [x] Import GLB/glTF, OBJ, PLY, and STL, with reversible small-hole mesh repair.
- [x] Render textured meshes using depth buffering, diffuse/toon materials, smooth normals, and perspective/orthographic cameras.
- [x] Save portable `.zfp` projects containing readable scene settings and copied assets.
- [x] Author node-based texture graphs and apply color/channel operations to object textures.
- [ ] Complete compositor evaluation and add typed physical fields, normalization, units, and UV/object/world-space mapping.
- [ ] Simulate heat diffusion, thermal emission, IR reflection, and night-vision sensors.
- [ ] Support skeletal animation, ragdolls, and modular physics.
- [ ] Build a real-time NVIDIA-accelerated renderer with replaceable cameras, lights, and post-processing.
- [ ] Create a concise Rust-based scripting language and compile scripts into games.
- [ ] Provide an in-app custom HUD creator.
- [ ] Compile, launch, and debug standalone native Rust games with editor telemetry.
- [ ] Make multiplayer native through concise authoritative server and scripting APIs.
- [ ] Stabilize replaceable subsystem contracts and document the complete game-creation workflow.

## Run

This Rust game engine rewritten from scratch can be compiled and run with:

```bash
cargo run -p zerofps-editor
```
