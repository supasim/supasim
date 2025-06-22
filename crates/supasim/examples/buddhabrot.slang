struct InputOptions
{
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint64_t randomSeed;
}

struct Parameters
{
    [[vk::binding(0, 0)]]
    InputOptions imageSize;
    [[vk::binding(1, 0)]]
    RWStructuredBuffer<Atomic<uint32_t>> buffer;
}

// Simple hash function to generate a uint from a seed and index
uint Hash(uint seed, uint value)
{
    uint x = seed ^ value;
    x ^= x >> 17;
    x *= 0xed5ad4bb;
    x ^= x >> 11;
    x *= 0xac4c1b51;
    x ^= x >> 15;
    x *= 0x31848bab;
    x ^= x >> 14;
    return x;
}

// Convert uint to float in [0, 1)
float UIntToFloat01(uint x)
{
    return (x & 0x00FFFFFF) / 16777216.0;
}

[shader("compute")]
[numthreads(32, 32, 1)]
void Main(uint3 threadId: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
    if (threadId.x >= parameters.imageSize.imageWidth ||
        threadId.y >= parameters.imageSize.imageHeight)
    {
        return;
    }

    uint index = threadId.y * parameters.imageSize.imageWidth + threadId.x;

    uint seedX = Hash(uint(parameters.imageSize.randomSeed), index * 2);
    uint seedY = Hash(uint(parameters.imageSize.randomSeed), index * 2 + 1);

    float randX = UIntToFloat01(seedX);
    float randY = UIntToFloat01(seedY);

    float2 randomVec = float2(randX, randY);

    uint newIndex = seedX % (parameters.imageSize.imageWidth * parameters.imageSize.imageHeight);

    // Example write
    // InterlockedAdd(parameters.buffer[newIndex], 1);
    parameters.buffer[newIndex].add(1);
}
