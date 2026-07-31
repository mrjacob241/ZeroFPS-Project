struct Camera {
    yaw: f32,
    pitch: f32,
    roll: f32,
    zoom: f32,
    grid_spacing: f32,
    _camera_padding_0: f32,
    _camera_padding_1: f32,
    _camera_padding_2: f32,
    camera_target: vec4<f32>,
    viewport: vec2<f32>,
    projection: u32,
    _padding: u32,
    global_light_enabled: u32,
    point_light_count: u32,
    directional_shadow_enabled: u32,
    _lighting_padding: u32,
    point_positions: array<vec4<f32>, 8>,
    point_colors: array<vec4<f32>, 8>,
    point_shadow_regions: array<vec4<f32>, 8>,
    shadow_origin: vec4<f32>,
    shadow_right: vec4<f32>,
    shadow_up: vec4<f32>,
    shadow_forward: vec4<f32>,
    shadow_parameters: vec4<f32>,
    point_shadow_atlas_size: vec4<f32>,
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
    @location(6) @interpolate(flat) object_id: f32,
};

@group(0) @binding(0) var<uniform> camera: Camera;
@group(0) @binding(1) var directional_shadow_depth: texture_2d<f32>;
@group(0) @binding(2) var point_shadow_depth: texture_2d<f32>;
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

fn spherical_light_lambert(normal_dot_light: f32, distance: f32, radius: f32) -> f32 {
    let angular_radius = select(
        1.0,
        clamp(max(radius, 0.0) / distance, 0.0, 1.0),
        distance > 0.000001
    );
    let rounded = 0.5 * (
        normal_dot_light
        + sqrt(normal_dot_light * normal_dot_light + angular_radius * angular_radius)
    );
    let normalization = 2.0 / (1.0 + sqrt(1.0 + angular_radius * angular_radius));
    return clamp(rounded * normalization, 0.0, 1.0);
}

fn directional_shadow_visibility(surface: vec3<f32>, normal: vec3<f32>) -> f32 {
    if camera.directional_shadow_enabled == 0u {
        return 1.0;
    }
    let relative = surface - camera.shadow_origin.xyz;
    let extent = camera.shadow_parameters.x;
    let uv = vec2<f32>(
        dot(relative, camera.shadow_right.xyz) / (extent * 2.0) + 0.5,
        0.5 - dot(relative, camera.shadow_up.xyz) / (extent * 2.0)
    );
    if any(uv < vec2<f32>(0.0)) || any(uv > vec2<f32>(1.0)) {
        return 1.0;
    }
    let resolution = i32(camera.shadow_parameters.z);
    let center = vec2<i32>(floor(uv * f32(resolution)));
    let receiver_depth = dot(relative, camera.shadow_forward.xyz);
    let normal_light = clamp(
        dot(normal, normalize(vec3<f32>(-0.35, 0.8, 0.45))),
        0.0,
        1.0
    );
    let receiver_bias =
        camera.shadow_parameters.y * (1.0 + (1.0 - normal_light) * 5.0);
    var visible = 0.0;
    var samples = 0.0;
    let filter_radius = i32(camera.shadow_parameters.w);
    for (var y = -2; y <= 2; y += 1) {
        for (var x = -2; x <= 2; x += 1) {
            if abs(x) <= filter_radius && abs(y) <= filter_radius {
            let texel = center + vec2<i32>(x, y);
            if any(texel < vec2<i32>(0)) || any(texel >= vec2<i32>(resolution)) {
                visible += 1.0;
            } else {
                let caster_depth = textureLoad(directional_shadow_depth, texel, 0).x;
                visible += select(
                    0.0,
                    1.0,
                    receiver_depth <= caster_depth + receiver_bias
                );
            }
            samples += 1.0;
            }
        }
    }
    return visible / samples;
}

fn cube_shadow_coordinate(direction: vec3<f32>) -> vec3<f32> {
    let absolute = abs(direction);
    var face = 0.0;
    var forward = vec3<f32>(1.0, 0.0, 0.0);
    var right = vec3<f32>(0.0, -1.0, 0.0);
    var up = vec3<f32>(0.0, 0.0, 1.0);
    if absolute.x >= absolute.y && absolute.x >= absolute.z {
        if direction.x < 0.0 {
            face = 1.0;
            forward = vec3<f32>(-1.0, 0.0, 0.0);
            right = vec3<f32>(0.0, 1.0, 0.0);
        }
    } else if absolute.y >= absolute.z {
        if direction.y >= 0.0 {
            face = 2.0;
            forward = vec3<f32>(0.0, 1.0, 0.0);
            right = vec3<f32>(1.0, 0.0, 0.0);
        } else {
            face = 3.0;
            forward = vec3<f32>(0.0, -1.0, 0.0);
            right = vec3<f32>(-1.0, 0.0, 0.0);
        }
    } else if direction.z >= 0.0 {
        face = 4.0;
        forward = vec3<f32>(0.0, 0.0, 1.0);
        right = vec3<f32>(1.0, 0.0, 0.0);
        up = vec3<f32>(0.0, -1.0, 0.0);
    } else {
        face = 5.0;
        forward = vec3<f32>(0.0, 0.0, -1.0);
        right = vec3<f32>(1.0, 0.0, 0.0);
        up = vec3<f32>(0.0, 1.0, 0.0);
    }
    let face_depth = dot(direction, forward);
    return vec3<f32>(
        face,
        dot(direction, right) / face_depth * 0.5 + 0.5,
        0.5 - dot(direction, up) / face_depth * 0.5
    );
}

fn point_shadow_visibility(
    surface: vec3<f32>,
    normal: vec3<f32>,
    light_index: u32
) -> f32 {
    let region = camera.point_shadow_regions[light_index];
    let resolution = i32(region.y);
    if resolution == 0 {
        return 1.0;
    }
    let point = surface - camera.point_positions[light_index].xyz;
    let distance = max(length(point), 0.00001);
    let direction = point / distance;
    let normal_light = clamp(dot(normal, -point / distance), 0.0, 1.0);
    let receiver_bias =
        region.z * distance * (1.0 + (1.0 - normal_light) * 5.0);
    let kernel = i32(clamp(
        ceil(camera.point_colors[light_index].w / distance * f32(resolution)),
        region.w,
        3.0
    ));
    let reference = select(
        vec3<f32>(0.0, 1.0, 0.0),
        vec3<f32>(0.0, 0.0, 1.0),
        abs(direction.z) < 0.9
    );
    let tangent = normalize(cross(reference, direction));
    let bitangent = normalize(cross(direction, tangent));
    let angular_step = 2.0 / f32(resolution);
    var visible = 0.0;
    var samples = 0.0;
    for (var y = -3; y <= 3; y += 1) {
        for (var x = -3; x <= 3; x += 1) {
            if abs(x) <= kernel && abs(y) <= kernel {
                let sample_direction = normalize(
                    direction
                        + tangent * f32(x) * angular_step
                        + bitangent * f32(y) * angular_step
                );
                let coordinate = cube_shadow_coordinate(sample_direction);
                let local = clamp(
                    vec2<i32>(floor(coordinate.yz * f32(resolution))),
                    vec2<i32>(0),
                    vec2<i32>(resolution - 1)
                );
                let texel = vec2<i32>(
                    i32(coordinate.x) * resolution + local.x,
                    i32(region.x) + local.y
                );
                let caster_depth = textureLoad(point_shadow_depth, texel, 0).x;
                visible += select(
                    0.0,
                    1.0,
                    distance <= caster_depth + receiver_bias
                );
                samples += 1.0;
            }
        }
    }
    return visible / samples;
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
    let base_x = point.x * cy - point.y * sy;
    let forward = point.x * sy + point.y * cy;
    let base_y = point.z * cp - forward * sp;
    let sr = sin(camera.roll);
    let cr = cos(camera.roll);
    let view_x = base_x * cr - base_y * sr;
    let view_y = base_x * sr + base_y * cr;
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
    out.object_id = input.object_translation.w;
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
    var diffuse = select(
        0.0,
        max(dot(normal, light_direction), 0.0)
            * directional_shadow_visibility(input.world_position, normal),
        camera.global_light_enabled != 0u
    );
    for (var index = 0u; index < camera.point_light_count; index += 1u) {
        let point = camera.point_positions[index];
        let offset = point.xyz - input.world_position;
        let distance_squared = dot(offset, offset);
        if distance_squared > 0.00000001 {
            let light_color = camera.point_colors[index].rgb;
            let luminance = (light_color.r + light_color.g + light_color.b) / 3.0;
            let attenuation = point.w * luminance / (1.0 + distance_squared);
            diffuse += spherical_light_lambert(
                dot(normal, normalize(offset)),
                sqrt(distance_squared),
                camera.point_colors[index].w
            ) * attenuation * point_shadow_visibility(input.world_position, normal, index);
        }
    }
    diffuse = clamp(diffuse, 0.0, 2.0);
    let lighting = clamp(select(0.28 + diffuse * 0.72, select(0.42, 0.86, diffuse > 0.5), input.material.x > 0.5), 0.0, 2.0);
    return vec4<f32>(
        sampled_rgb * input.color.rgb * input.base_color.rgb * lighting,
        sampled.a * input.color.a * input.base_color.a
    );
}
