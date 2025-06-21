// Courtesy of ChatGPT

struct Parameters
{
    [[vk::binding(0, 0)]]
    StructuredBuffer<uint32_t> imageSize;
    [[vk::binding(1, 0)]]
    RWStructuredBuffer<uint32_t> buffer;
}

[shader("compute")]
[numthreads(32, 32, 1)]
void main(uint3 threadId: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
}
