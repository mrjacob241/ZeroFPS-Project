use std::{
    hint::black_box,
    time::{Duration, Instant},
};

struct Image {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
}

fn random_image(width: usize, height: usize, mut state: u64) -> Image {
    let mut pixels = vec![0; width * height * 4];
    for byte in &mut pixels {
        state ^= state << 13;
        state ^= state >> 7;
        state ^= state << 17;
        *byte = state as u8;
    }
    Image {
        width,
        height,
        pixels,
    }
}

fn mix_different_resolutions(a: &Image, b: &Image, alpha: f32) -> Vec<u8> {
    let width = a.width.max(b.width);
    let height = a.height.max(b.height);
    let mut output = vec![0; width * height * 4];
    for y in 0..height {
        let ay = (y * a.height / height).min(a.height - 1);
        let by = (y * b.height / height).min(b.height - 1);
        for x in 0..width {
            let ax = (x * a.width / width).min(a.width - 1);
            let bx = (x * b.width / width).min(b.width - 1);
            let a_offset = (ay * a.width + ax) * 4;
            let b_offset = (by * b.width + bx) * 4;
            let output_offset = (y * width + x) * 4;
            for channel in 0..4 {
                output[output_offset + channel] = (alpha * a.pixels[a_offset + channel] as f32
                    + (1.0 - alpha) * b.pixels[b_offset + channel] as f32)
                    .round() as u8;
            }
        }
    }
    output
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1_000.0
}

fn main() {
    let cases = [
        ((128, 128), (64, 96)),
        ((256, 256), (128, 192)),
        ((512, 512), (256, 384)),
        ((1024, 1024), (512, 768)),
        ((2048, 2048), (1024, 1536)),
        ((4096, 4096), (2048, 3072)),
    ];
    println!("Rust compositor mix benchmark (release mode, 10 samples per case)");
    println!("A resolution | B resolution | median ms | mean ms | min ms | max ms | output MPix/s");
    for (case, &((aw, ah), (bw, bh))) in cases.iter().enumerate() {
        let a = random_image(aw, ah, 0x1234_5678_9abc_def0 ^ case as u64);
        let b = random_image(bw, bh, 0xfedc_ba98_7654_3210 ^ case as u64);
        black_box(mix_different_resolutions(&a, &b, 0.37));
        let mut samples = Vec::with_capacity(10);
        for _ in 0..10 {
            let start = Instant::now();
            let output = mix_different_resolutions(black_box(&a), black_box(&b), black_box(0.37));
            black_box(output);
            samples.push(start.elapsed());
        }
        samples.sort_unstable();
        let median = (milliseconds(samples[4]) + milliseconds(samples[5])) * 0.5;
        let mean = samples.iter().map(|&time| milliseconds(time)).sum::<f64>() / 10.0;
        let minimum = milliseconds(samples[0]);
        let maximum = milliseconds(samples[9]);
        let output_pixels = aw.max(bw) * ah.max(bh);
        let throughput = output_pixels as f64 / (median / 1_000.0) / 1_000_000.0;
        println!(
            "{aw:4}x{ah:<4} | {bw:4}x{bh:<4} | {median:9.3} | {mean:7.3} | \
             {minimum:6.3} | {maximum:6.3} | {throughput:12.2}"
        );
    }
}
