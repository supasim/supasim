#include <metal_stdlib>
#include <simd/simd.h>

using namespace metal;

struct RWStructuredBuffer
{
    uint _m0[1];
};

struct StructuredBuffer
{
    uint _m0[1];
};

kernel void main0(device RWStructuredBuffer& entryPointParams_parameters_result [[buffer(0)]], const device StructuredBuffer& entryPointParams_parameters_buffer0 [[buffer(1)]], const device StructuredBuffer& entryPointParams_parameters_buffer1 [[buffer(2)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    entryPointParams_parameters_result._m0[gl_GlobalInvocationID.x] = entryPointParams_parameters_buffer0._m0[gl_GlobalInvocationID.x] + entryPointParams_parameters_buffer1._m0[gl_GlobalInvocationID.x];
}

