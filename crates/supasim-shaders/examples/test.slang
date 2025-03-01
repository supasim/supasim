struct Parameters
{
    StructuredBuffer<float> buffer0;
    StructuredBuffer<float> buffer1;
    RWStructuredBuffer<float> result;
}

Parameters parameters;
RWStructuredBuffer<float> buffer5;

[shader("compute")]
[numthreads(1, 1, 1)]
void computeMain(uint3 threadId: SV_DispatchThreadID /*, ParameterBlock<Parameters> parameters*/, RWStructuredBuffer<float> buffer3)
{
    uint index = threadId.x;
    parameters.result[index] = parameters.buffer0[index] + parameters.buffer1[index];
}
