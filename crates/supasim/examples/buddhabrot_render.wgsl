// Courtesy of ChatGPTs

const WORKGROUP_SIZE_X: u32 = 16;

struct OutputSize {
    width: u32,
    height: u32,
}

@group(0) @binding(0)
var<uniform> size: OutputSize;

@group(0) @binding(1)
var<uniform> max_value: u32;

@group(1) @binding(0)
var<storage, read> buffer: array<u32>;

// Storage texture for output
// Note: WGSL currently does not support dynamic texture formats via pipeline constants.
// You must create separate pipelines for different formats.
@group(2) @binding(0)
var output_image: texture_storage_2d<rgba8unorm, write>;

@compute @workgroup_size(WORKGROUP_SIZE_X, WORKGROUP_SIZE_X)
fn render(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let x = global_id.x;
    let y = global_id.y;

    if (x >= size.width || y >= size.height) {
        return;
    }

    let index = y * size.width + x;
    let value = buffer[index];

    // Normalize value to [0.0, 1.0]
    let scaled = f32(value) / f32(max_value);

    // Output grayscale color
    let color = vec4f(scaled, scaled, scaled, 1.0);
    textureStore(output_image, vec2u(x, y), color);
}
