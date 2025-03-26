struct Parameters
{
    RWStructuredBuffer<uint32_t> buffer;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void main(uint3 threadId: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
    uint index = threadId.x;
    parameters.buffer[index] = parameters.buffer[index] * 2;
}
