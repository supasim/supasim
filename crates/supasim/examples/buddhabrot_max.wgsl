// Courtesy of ChatGPT

const WORKGROUP_SIZE: u32 = 256;

@group(0) @binding(0)
var<storage, read> input: array<u32>;

@group(0) @binding(1)
var<storage, read_write> output: array<u32>;

@compute @workgroup_size(WORKGROUP_SIZE)
fn max(@builtin(global_invocation_id) global_id: vec3<u32>,
        @builtin(local_invocation_id) local_id: vec3<u32>,
        @builtin(workgroup_id) workgroup_id: vec3<u32>) {
    let gid = global_id.x;
    let lid = local_id.x;
    let group = workgroup_id.x;

    // Shared memory for reduction
    var<workgroup> shared_max: array<u32, WORKGROUP_SIZE>;

    // Load data into shared memory
    if (gid < arrayLength(&input)) {
        shared_max[lid] = input[gid];
    } else {
        shared_max[lid] = 0u; // Identity for max
    }
    workgroupBarrier();

    // Reduction within workgroup
    var stride = WORKGROUP_SIZE / 2u;
    loop {
        if (lid < stride) {
            shared_max[lid] = max(shared_max[lid], shared_max[lid + stride]);
        }
        workgroupBarrier();
        if (stride == 1u) { break; }
        stride = stride / 2u;
    }

    // Write result of this workgroup to output
    if (lid == 0u) {
        output[group] = shared_max[0];
    }
}
