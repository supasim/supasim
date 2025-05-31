struct Parameters
{
    StructuredBuffer<uint32_t> buffer0;
    StructuredBuffer<uint32_t> buffer1;
    RWStructuredBuffer<uint32_t> result;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void add(uint3 threadId: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
    uint index = threadId.x;
    parameters.result[index] = parameters.buffer0[index] + parameters.buffer1[index];
}
