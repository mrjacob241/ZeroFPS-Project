struct Parameters {
    a_width: u32,
    a_height: u32,
    b_width: u32,
    b_height: u32,
    output_width: u32,
    output_height: u32,
    alpha_bits: u32,
    operation: u32,
}

@group(0) @binding(0) var<storage, read> image_a: array<u32>;
@group(0) @binding(1) var<storage, read> image_b: array<u32>;
@group(0) @binding(2) var output_image: texture_storage_2d<rgba8unorm, write>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

fn unpack(value: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(value & 255u), f32((value >> 8u) & 255u),
        f32((value >> 16u) & 255u), f32((value >> 24u) & 255u)
    ) / 255.0;
}

fn pack(value: vec4<f32>) -> u32 {
    let v = vec4<u32>(clamp(round(value * 255.0), vec4<f32>(0.0), vec4<f32>(255.0)));
    return v.x | (v.y << 8u) | (v.z << 16u) | (v.w << 24u);
}

fn combine(a: vec4<f32>, b: vec4<f32>) -> vec4<f32> {
    switch parameters.operation {
        case 0u: { return a + b; }
        case 1u: { return a - b; }
        case 2u: { return a * b; }
        case 3u: { return select(a / b, vec4<f32>(0.0), abs(b) <= vec4<f32>(0.000001)); }
        case 4u: { return pow(max(a, vec4<f32>(0.0)), b); }
        case 5u: { return min(a, b); }
        case 6u: { return max(a, b); }
        case 7u: { return abs(a - b); }
        default: {
            let alpha = bitcast<f32>(parameters.alpha_bits);
            return alpha * a + (1.0 - alpha) * b;
        }
    }
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if id.x >= parameters.output_width || id.y >= parameters.output_height { return; }
    let ax = min(id.x * parameters.a_width / parameters.output_width, parameters.a_width - 1u);
    let ay = min(id.y * parameters.a_height / parameters.output_height, parameters.a_height - 1u);
    let bx = min(id.x * parameters.b_width / parameters.output_width, parameters.b_width - 1u);
    let by = min(id.y * parameters.b_height / parameters.output_height, parameters.b_height - 1u);
    let result = combine(
        unpack(image_a[ay * parameters.a_width + ax]),
        unpack(image_b[by * parameters.b_width + bx])
    );
    textureStore(
        output_image,
        vec2<i32>(id.xy),
        clamp(result, vec4<f32>(0.0), vec4<f32>(1.0))
    );
}
