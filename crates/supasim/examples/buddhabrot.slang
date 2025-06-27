struct InputOptions
{
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint64_t randomSeed;
}

struct Parameters
{
    [[vk::binding(0, 0)]]
    RWStructuredBuffer<InputOptions> imageSize;
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
[numthreads(16, 16, 1)]
void Main(uint3 threadId: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
    InputOptions options = parameters.imageSize[0];
    if (threadId.x >= options.imageWidth ||
        threadId.y >= options.imageHeight)
    {
        return;
    }

    uint index = threadId.y * options.imageWidth + threadId.x;

    uint seedX = Hash(uint(options.randomSeed), index * 2);
    uint seedY = Hash(uint(options.randomSeed), index * 2 + 1);

    float cx = UIntToFloat01(seedX) * 4.0 - 2.0;
    float cy = UIntToFloat01(seedY) * 4.0 - 2.0;

    float2 c = float2(cx, cy);

    float2 pos = float2(0.0);

    const int MAX_ITERATIONS = 128;

    int num = 0;
    if (dot(pos, pos) <= 4)
    {
        for (int i = 0; i < MAX_ITERATIONS; i++)
        {
            pos = square(pos) + c;
            if (dot(pos, pos) > 4)
            {
                num = i;
                break;
            }
        }
        pos = float2(0.0);
        for (int i = 0; i < num; i++)
        {
            pos = square(pos) + c;
            int index = getPixelIndex(pos, parameters.imageSize[0].imageWidth, parameters.imageSize[0].imageHeight);
            parameters.buffer[index].add(1);
        }
    }

    // Example write
    // InterlockedAdd(parameters.buffer[newIndex], 1);
    // parameters.buffer[newIndex].add(1);
}

float2 square(float2 pos)
{
    return float2(pos.x * pos.x - pos.y * pos.y, 2 * pos.x * pos.y);
}
int getPixelIndex(float2 pos, uint32_t width, uint32_t height)
{
    float2 posNorm = float2(pos.x / 4.0 + 0.5, pos.y / 4.0 + 0.5);
    int xpos = (int)(posNorm.x * (float)width);
    int ypos = (int)(posNorm.y * (float)height);
    return xpos + ypos * width;
}
