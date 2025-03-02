#version 450
layout(column_major) uniform;
layout(column_major) buffer;

#line 1 0
layout(std430, binding = 0) readonly buffer StructuredBuffer_uint_t_0 {
    uint _data[];
} entryPointParams_parameters_buffer0_0;

#line 1
layout(std430, binding = 1) readonly buffer StructuredBuffer_uint_t_1 {
    uint _data[];
} entryPointParams_parameters_buffer1_0;

#line 1
layout(std430, binding = 2) buffer StructuredBuffer_uint_t_2 {
    uint _data[];
} entryPointParams_parameters_result_0;

#line 10
layout(local_size_x = 1, local_size_y = 1, local_size_z = 1) in;
void main()
{

#line 12
    uint index_0 = gl_GlobalInvocationID.x;
    entryPointParams_parameters_result_0._data[uint(index_0)] = entryPointParams_parameters_buffer0_0._data[uint(index_0)] + entryPointParams_parameters_buffer1_0._data[uint(index_0)];
    return;
}

