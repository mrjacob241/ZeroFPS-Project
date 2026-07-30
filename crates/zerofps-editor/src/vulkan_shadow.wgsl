struct ShadowUniform {
    origin: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    parameters: vec4<f32>,
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

struct ShadowVertex {
    @builtin(position) clip_position: vec4<f32>,
    @location(0) depth_value: f32,
};

@group(0) @binding(0) var<uniform> shadow: ShadowUniform;

fn quaternion_rotate(q: vec4<f32>, value: vec3<f32>) -> vec3<f32> {
    let t = 2.0 * cross(q.xyz, value);
    return value + q.w * t + cross(q.xyz, t);
}

@vertex
fn vs_shadow(input: VertexInput) -> ShadowVertex {
    let world = quaternion_rotate(
        input.object_rotation,
        input.position.xyz * input.object_scale.xyz
    ) + input.object_translation.xyz;
    let relative = world - shadow.origin.xyz;
    let x = dot(relative, shadow.right.xyz);
    let y = dot(relative, shadow.up.xyz);
    let z = dot(relative, shadow.forward.xyz);
    let extent = max(shadow.parameters.x, 0.000001);
    let near_depth = shadow.parameters.y;
    let far_depth = max(shadow.parameters.z, near_depth + 0.000001);
    var out: ShadowVertex;
    out.clip_position = vec4<f32>(
        x / extent,
        y / extent,
        clamp((z - near_depth) / (far_depth - near_depth), 0.0, 1.0),
        1.0
    );
    out.depth_value = z;
    return out;
}

@vertex
fn vs_point_shadow(input: VertexInput) -> ShadowVertex {
    let world = quaternion_rotate(
        input.object_rotation,
        input.position.xyz * input.object_scale.xyz
    ) + input.object_translation.xyz;
    let relative = world - shadow.origin.xyz;
    let x = dot(relative, shadow.right.xyz);
    let y = dot(relative, shadow.up.xyz);
    let z = dot(relative, shadow.forward.xyz);
    let near_depth = 0.00001;
    let far_depth = max(shadow.parameters.x, near_depth + 0.00001);
    let safe_z = max(z, near_depth);
    var out: ShadowVertex;
    out.clip_position = vec4<f32>(
        x,
        y,
        clamp((safe_z - near_depth) / (far_depth - near_depth), 0.0, 1.0) * safe_z,
        z
    );
    out.depth_value = length(relative);
    return out;
}

@fragment
fn fs_shadow(input: ShadowVertex) -> @location(0) f32 {
    return input.depth_value;
}
