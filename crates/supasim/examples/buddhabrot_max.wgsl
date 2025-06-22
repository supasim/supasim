// Courtesy of ChatGPT

const WORKGROUP_SIZE: u32 = 256;

struct OutputSize {
    width: u32,
    height: u32,
}

@group(0) @binding(0)
var<uniform> size: OutputSize;

@group(0) @binding(1)
var<storage, read_write> output: atomic<u32>;

@group(1) @binding(0)
var<storage, read> buffer: array<u32>;

var<workgroup> local_max: array<u32, WORKGROUP_SIZE>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn find_max(
    @builtin(global_invocation_id) global_id: vec3<u32>,
    @builtin(local_invocation_id) local_id: vec3<u32>,
) {
    let idx = global_id.x;
    if (idx == 0u) {
        output = 1u;
    }
    let total_size = size.width * size.height;
    
    // Store each thread's value into workgroup memory
    if (idx < total_size) { local_max[local_id.x] = buffer[idx]; } else { local_max[local_id.x] = 0u; }

    workgroupBarrier();

    // Parallel reduction to find the max in the workgroup
    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (local_id.x < stride) {
            local_max[local_id.x] = max(local_max[local_id.x], local_max[local_id.x + stride]);
        }
        workgroupBarrier();

        if (stride == 1u) {
            break;
        }
        stride = stride / 2u;
    }

    // Only one thread writes the result of this workgroup
    if (local_id.x == 0u) {
        atomicMax(&output, local_max[0]);
    }
}