use std::{
    hint::black_box,
    time::{Duration, Instant},
};

use wgpu::util::DeviceExt;

#[repr(C)]
#[derive(Clone, Copy, bytemuck::Pod, bytemuck::Zeroable)]
struct Parameters {
    a_width: u32,
    a_height: u32,
    b_width: u32,
    b_height: u32,
    output_width: u32,
    output_height: u32,
    alpha_bits: u32,
    _padding: u32,
}

fn random_pixels(width: u32, height: u32, mut state: u64) -> Vec<u32> {
    (0..width as usize * height as usize)
        .map(|_| {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state as u32
        })
        .collect()
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn main() {
    if let Err(error) = pollster::block_on(run()) {
        eprintln!("Vulkan benchmark unavailable: {error}");
        eprintln!("The CPU benchmark remains available as the fallback backend.");
        std::process::exit(1);
    }
}

async fn run() -> Result<(), String> {
    let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        ..Default::default()
    });
    let adapter = instance
        .request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        })
        .await
        .ok_or_else(|| "no Vulkan adapter was found".to_string())?;
    let info = adapter.get_info();
    if info.backend != wgpu::Backend::Vulkan {
        return Err(format!(
            "selected backend is {:?}, not Vulkan",
            info.backend
        ));
    }
    let (device, queue) = adapter
        .request_device(
            &wgpu::DeviceDescriptor {
                label: Some("ZeroFPS Vulkan mix benchmark"),
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::default(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        )
        .await
        .map_err(|error| format!("Vulkan device creation failed: {error}"))?;

    let shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
        label: Some("RGBA nearest-resample mix"),
        source: wgpu::ShaderSource::Wgsl(
            r#"
struct Parameters {
    a_width: u32,
    a_height: u32,
    b_width: u32,
    b_height: u32,
    output_width: u32,
    output_height: u32,
    alpha_bits: u32,
    padding: u32,
}
@group(0) @binding(0) var<storage, read> image_a: array<u32>;
@group(0) @binding(1) var<storage, read> image_b: array<u32>;
@group(0) @binding(2) var<storage, read_write> output_image: array<u32>;
@group(0) @binding(3) var<uniform> parameters: Parameters;

fn unpack_rgba(value: u32) -> vec4<f32> {
    return vec4<f32>(
        f32(value & 255u),
        f32((value >> 8u) & 255u),
        f32((value >> 16u) & 255u),
        f32((value >> 24u) & 255u)
    );
}

fn pack_rgba(value: vec4<f32>) -> u32 {
    let rounded = vec4<u32>(clamp(round(value), vec4<f32>(0.0), vec4<f32>(255.0)));
    return rounded.x | (rounded.y << 8u) | (rounded.z << 16u) | (rounded.w << 24u);
}

@compute @workgroup_size(16, 16, 1)
fn main(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= parameters.output_width || id.y >= parameters.output_height) {
        return;
    }
    let ax = min(id.x * parameters.a_width / parameters.output_width, parameters.a_width - 1u);
    let ay = min(id.y * parameters.a_height / parameters.output_height, parameters.a_height - 1u);
    let bx = min(id.x * parameters.b_width / parameters.output_width, parameters.b_width - 1u);
    let by = min(id.y * parameters.b_height / parameters.output_height, parameters.b_height - 1u);
    let alpha = bitcast<f32>(parameters.alpha_bits);
    let mixed = alpha * unpack_rgba(image_a[ay * parameters.a_width + ax])
        + (1.0 - alpha) * unpack_rgba(image_b[by * parameters.b_width + bx]);
    output_image[id.y * parameters.output_width + id.x] = pack_rgba(mixed);
}
"#
            .into(),
        ),
    });
    let bind_group_layout = device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
        label: Some("mix bindings"),
        entries: &[
            storage_entry(0, true),
            storage_entry(1, true),
            storage_entry(2, false),
            wgpu::BindGroupLayoutEntry {
                binding: 3,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Uniform,
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            },
        ],
    });
    let pipeline_layout = device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
        label: Some("mix pipeline layout"),
        bind_group_layouts: &[&bind_group_layout],
        push_constant_ranges: &[],
    });
    let pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
        label: Some("Vulkan RGBA mix"),
        layout: Some(&pipeline_layout),
        module: &shader,
        entry_point: Some("main"),
        compilation_options: Default::default(),
        cache: None,
    });

    println!(
        "Vulkan device: {} ({:?}, driver {})",
        info.name, info.device_type, info.driver_info
    );
    println!(
        "10 samples per case; upload and buffer creation excluded; submit + GPU completion included"
    );
    println!("A resolution | B resolution | median ms | mean ms | min ms | max ms | output MPix/s");
    let cases = [
        ((128, 128), (64, 96)),
        ((256, 256), (128, 192)),
        ((512, 512), (256, 384)),
        ((1024, 1024), (512, 768)),
        ((2048, 2048), (1024, 1536)),
        ((4096, 4096), (2048, 3072)),
    ];
    for (case, &((aw, ah), (bw, bh))) in cases.iter().enumerate() {
        let ow = aw.max(bw);
        let oh = ah.max(bh);
        let a = random_pixels(aw, ah, 0x1234_5678_9abc_def0 ^ case as u64);
        let b = random_pixels(bw, bh, 0xfedc_ba98_7654_3210 ^ case as u64);
        let a_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("image A"),
            contents: bytemuck::cast_slice(&a),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let b_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("image B"),
            contents: bytemuck::cast_slice(&b),
            usage: wgpu::BufferUsages::STORAGE,
        });
        let output = device.create_buffer(&wgpu::BufferDescriptor {
            label: Some("mixed output"),
            size: ow as u64 * oh as u64 * 4,
            usage: wgpu::BufferUsages::STORAGE | wgpu::BufferUsages::COPY_SRC,
            mapped_at_creation: false,
        });
        let parameters = Parameters {
            a_width: aw,
            a_height: ah,
            b_width: bw,
            b_height: bh,
            output_width: ow,
            output_height: oh,
            alpha_bits: 0.37f32.to_bits(),
            _padding: 0,
        };
        let parameter_buffer = device.create_buffer_init(&wgpu::util::BufferInitDescriptor {
            label: Some("mix parameters"),
            contents: bytemuck::bytes_of(&parameters),
            usage: wgpu::BufferUsages::UNIFORM,
        });
        let bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: Some("mix resources"),
            layout: &bind_group_layout,
            entries: &[
                binding(0, &a_buffer),
                binding(1, &b_buffer),
                binding(2, &output),
                binding(3, &parameter_buffer),
            ],
        });
        dispatch(&device, &queue, &pipeline, &bind_group, ow, oh);
        let mut samples = Vec::with_capacity(10);
        for _ in 0..10 {
            let start = Instant::now();
            dispatch(&device, &queue, &pipeline, &bind_group, ow, oh);
            samples.push(start.elapsed());
        }
        samples.sort_unstable();
        let median = (milliseconds(samples[4]) + milliseconds(samples[5])) * 0.5;
        let mean = samples.iter().map(|&time| milliseconds(time)).sum::<f64>() / 10.0;
        let minimum = milliseconds(samples[0]);
        let maximum = milliseconds(samples[9]);
        let throughput = ow as f64 * oh as f64 / (median / 1_000.0) / 1_000_000.0;
        black_box(&output);
        println!(
            "{aw:4}x{ah:<4} | {bw:4}x{bh:<4} | {median:9.3} | {mean:7.3} | \
             {minimum:6.3} | {maximum:6.3} | {throughput:12.2}"
        );
    }
    Ok(())
}

fn storage_entry(binding: u32, read_only: bool) -> wgpu::BindGroupLayoutEntry {
    wgpu::BindGroupLayoutEntry {
        binding,
        visibility: wgpu::ShaderStages::COMPUTE,
        ty: wgpu::BindingType::Buffer {
            ty: wgpu::BufferBindingType::Storage { read_only },
            has_dynamic_offset: false,
            min_binding_size: None,
        },
        count: None,
    }
}

fn binding(binding: u32, buffer: &wgpu::Buffer) -> wgpu::BindGroupEntry<'_> {
    wgpu::BindGroupEntry {
        binding,
        resource: buffer.as_entire_binding(),
    }
}

fn dispatch(
    device: &wgpu::Device,
    queue: &wgpu::Queue,
    pipeline: &wgpu::ComputePipeline,
    bind_group: &wgpu::BindGroup,
    width: u32,
    height: u32,
) {
    let mut encoder = device.create_command_encoder(&wgpu::CommandEncoderDescriptor {
        label: Some("mix benchmark encoder"),
    });
    {
        let mut pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
            label: Some("RGBA mix pass"),
            timestamp_writes: None,
        });
        pass.set_pipeline(pipeline);
        pass.set_bind_group(0, bind_group, &[]);
        pass.dispatch_workgroups(width.div_ceil(16), height.div_ceil(16), 1);
    }
    queue.submit([encoder.finish()]);
    device.poll(wgpu::Maintain::Wait);
}
