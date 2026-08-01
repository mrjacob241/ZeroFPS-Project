struct ShadowUniform {
    origin: vec4<f32>,
    right: vec4<f32>,
    up: vec4<f32>,
    forward: vec4<f32>,
    parameters: vec4<f32>,
    atlas: vec4<f32>,
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
    @location(1) @interpolate(flat) atlas_bounds: vec4<f32>,
};

@group(0) @binding(0) var<uniform> shadow: ShadowUniform;
@group(0) @binding(1) var<storage, read> point_shadow_views: array<ShadowUniform>;

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
    out.atlas_bounds = vec4<f32>(-1.0, -1.0, -1.0, -1.0);
    return out;
}

@vertex
fn vs_point_shadow(
    input: VertexInput,
    @builtin(instance_index) view_index: u32,
) -> ShadowVertex {
    let point_shadow = point_shadow_views[view_index];
    let world = quaternion_rotate(
        input.object_rotation,
        input.position.xyz * input.object_scale.xyz
    ) + input.object_translation.xyz;
    let relative = world - point_shadow.origin.xyz;
    let x = dot(relative, point_shadow.right.xyz);
    let y = dot(relative, point_shadow.up.xyz);
    let z = dot(relative, point_shadow.forward.xyz);
    let near_depth = 0.00001;
    let far_depth = max(point_shadow.parameters.x, near_depth + 0.00001);
    let safe_z = max(z, near_depth);
    let projected_x = x / safe_z;
    let projected_y = y / safe_z;
    let atlas_x = point_shadow.atlas.z + projected_x * point_shadow.atlas.x;
    let atlas_y = point_shadow.atlas.w + projected_y * point_shadow.atlas.y;
    var out: ShadowVertex;
    out.clip_position = vec4<f32>(
        atlas_x * safe_z,
        atlas_y * safe_z,
        clamp((safe_z - near_depth) / (far_depth - near_depth), 0.0, 1.0) * safe_z,
        z
    );
    out.depth_value = length(relative);
    let atlas_width = point_shadow.parameters.y;
    let atlas_height = point_shadow.parameters.z;
    out.atlas_bounds = vec4<f32>(
        (point_shadow.atlas.z - point_shadow.atlas.x + 1.0) * 0.5 * atlas_width,
        (1.0 - point_shadow.atlas.w - point_shadow.atlas.y) * 0.5 * atlas_height,
        (point_shadow.atlas.z + point_shadow.atlas.x + 1.0) * 0.5 * atlas_width,
        (1.0 - point_shadow.atlas.w + point_shadow.atlas.y) * 0.5 * atlas_height,
    );
    return out;
}

@fragment
fn fs_shadow(input: ShadowVertex) -> @location(0) f32 {
    if input.atlas_bounds.x >= 0.0 && (
        input.clip_position.x < input.atlas_bounds.x
        || input.clip_position.y < input.atlas_bounds.y
        || input.clip_position.x >= input.atlas_bounds.z
        || input.clip_position.y >= input.atlas_bounds.w
    ) {
        discard;
    }
    return input.depth_value;
}
