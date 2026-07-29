struct Camera {
    yaw: f32,
    pitch: f32,
    zoom: f32,
    grid_spacing: f32,
    camera_target: vec4<f32>,
    viewport: vec2<f32>,
    projection: u32,
    _padding: u32,
    global_light_enabled: u32,
    point_light_count: u32,
    _lighting_padding: vec2<u32>,
    point_positions: array<vec4<f32>, 8>,
    point_colors: array<vec4<f32>, 8>,
};

struct VertexInput {
    @location(0) position: vec4<f32>,
    @location(1) normal: vec4<f32>,
    @location(2) uv_color_rg: vec4<f32>,
    @location(3) color_ba_base_rg: vec4<f32>,
    @location(4) base_ba_material: vec4<f32>,
    @location(5) object_translation: vec4<f32>,
    @location(6) object_rotation: vec4<f32>,
    @location(7) object_scale: vec4<f32>,
};

struct VertexOutput {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) normal: vec3<f32>,
    @location(1) uv: vec2<f32>,
    @location(2) color: vec4<f32>,
    @location(3) base_color: vec4<f32>,
    @location(4) @interpolate(flat) material: vec3<f32>,
    @location(5) world_position: vec3<f32>,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(1) @binding(0) var model_texture: texture_2d<f32>;
@group(1) @binding(1) var model_sampler: sampler;

fn quaternion_rotate(q: vec4<f32>, value: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, value);
    return value + q.w * t + cross(q.xyz, t);
}

fn srgb_to_linear(value: vec3<f32>) -> vec3<f32> {
    return select(
        pow((value + vec3<f32>(0.055)) / 1.055, vec3<f32>(2.4)),
        value / 12.92,
        value <= vec3<f32>(0.04045)
    );
}

@vertex
fn vs_main(input: VertexInput) -> VertexOutput {
    let world_position = quaternion_rotate(
        input.object_rotation,
        input.position.xyz * input.object_scale.xyz
    ) + input.object_translation.xyz;
    let point = world_position - camera.camera_target.xyz;
    let sy = sin(camera.yaw);
    let cy = cos(camera.yaw);
    let sp = sin(camera.pitch);
    let cp = cos(camera.pitch);
    let view_x = point.x * cy - point.y * sy;
    let forward = point.x * sy + point.y * cy;
    let view_y = point.z * cp - forward * sp;
    let view_depth = point.z * sp + forward * cp;
    let camera_distance = 20.0 * camera.grid_spacing;
    let depth = camera_distance + view_depth;
    let scale = min(camera.viewport.x, camera.viewport.y) * 0.18 * camera.zoom;
    let center_offset = vec2<f32>(0.0, -50.0 / camera.viewport.y);

    var out: VertexOutput;
    if camera.projection == 0u {
        let near = 0.001 * camera.grid_spacing;
        let far = camera_distance * 1000.0;
        let safe_depth = max(depth, near);
        let normalized_depth = clamp((safe_depth - near) / (far - near), 0.0, 1.0);
        out.clip_position = vec4<f32>(
            center_offset.x * safe_depth + 2.0 * view_x * scale * camera_distance / camera.viewport.x,
            center_offset.y * safe_depth + 2.0 * view_y * scale * camera_distance / camera.viewport.y,
            normalized_depth * safe_depth,
            safe_depth
        );
    } else {
        out.clip_position = vec4<f32>(
            center_offset.x + 2.0 * view_x * scale / camera.viewport.x,
            center_offset.y + 2.0 * view_y * scale / camera.viewport.y,
            clamp(depth / (camera_distance * 1000.0), 0.0, 1.0),
            1.0
        );
    }
    // Inverse-transpose for diagonal object scale, followed by the object's
    // world rotation. Preserve the sign of mirrored axes: abs(scale) would
    // silently turn the normal back into local-looking illumination.
    let safe_scale = select(
        input.object_scale.xyz,
        vec3<f32>(1.0),
        abs(input.object_scale.xyz) <= vec3<f32>(0.000001)
    );
    out.normal = normalize(quaternion_rotate(
        normalize(input.object_rotation),
        input.normal.xyz / safe_scale
    ));
    out.uv = input.uv_color_rg.xy;
    out.color = vec4<f32>(
        input.uv_color_rg.zw,
        input.color_ba_base_rg.xy
    );
    out.base_color = vec4<f32>(
        input.color_ba_base_rg.zw,
        input.base_ba_material.xy
    );
    out.material = vec3<f32>(input.base_ba_material.zw, input.object_scale.w);
    out.world_position = world_position;
    return out;
}

@fragment
fn fs_main(input: VertexOutput) -> @location(0) vec4<f32> {
    let sampled = textureSample(model_texture, model_sampler, input.uv);
    let sampled_rgb = select(
        sampled.rgb,
        srgb_to_linear(sampled.rgb),
        input.material.z > 0.5
    );
    let normal = normalize(input.normal);
    // Direction from every world-space surface point toward the editor's
    // global directional light. It is intentionally independent of object
    // transform and camera orientation.
    let light_direction = normalize(vec3<f32>(-0.35, 0.8, 0.45));
    var diffuse = select(0.0, max(dot(normal, light_direction), 0.0), camera.global_light_enabled != 0u);
    for (var index = 0u; index < camera.point_light_count; index += 1u) {
        let point = camera.point_positions[index];
        let offset = point.xyz - input.world_position;
        let distance_squared = dot(offset, offset);
        if distance_squared > 0.00000001 {
            let light_color = camera.point_colors[index].rgb;
            let luminance = (light_color.r + light_color.g + light_color.b) / 3.0;
            let attenuation = point.w * luminance / (1.0 + distance_squared);
            diffuse += max(dot(normal, normalize(offset)), 0.0) * attenuation;
        }
    }
    diffuse = clamp(diffuse, 0.0, 2.0);
    let lighting = clamp(select(0.28 + diffuse * 0.72, select(0.42, 0.86, diffuse > 0.5), input.material.x > 0.5), 0.0, 2.0);
    return vec4<f32>(
        sampled_rgb * input.color.rgb * input.base_color.rgb * lighting,
        sampled.a * input.color.a * input.base_color.a
    );
}
