// Courtesy of ChatGPTs

@group(0) @binding(0)
var<storage, read> values: array<u32>;

@group(0) @binding(1)
var output_tex: texture_storage_2d<rgba8unorm, write>;

@group(0) @binding(2)
var<uniform> config: Uniforms;

struct Uniforms {
    width: u32,
    height: u32,
    max_value: u32,
};

@compute @workgroup_size(16, 16)
fn render(@builtin(global_invocation_id) id: vec3<u32>) {
    if (id.x >= config.width || id.y >= config.height) {
        return;
    }

    let index = id.y * config.width + id.x;
    let raw = values[index];
    let normalized = f32(raw) / f32(config.max_value);
    let color = vec4<f32>(normalized, normalized, normalized, 1.0); // grayscale

    textureStore(output_tex, vec2<i32>(id.xy), color);
}
