/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 SupaMaggie70 (Magnus Larsson)


  SupaSim is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 3
  of the License, or (at your option) any later version.

  SupaSim is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
END LICENSE */
struct InputOptions
{
    uint32_t imageWidth;
    uint32_t imageHeight;
    uint64_t randomSeed;
}

struct Parameters
{
    StructuredBuffer<InputOptions> imageSize;
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

    // Setting this too high can result in the GPU killing it and the process
    // I don't have a good workaround. On my system 2048 is fine but 4096 isn't.
    // 512 is set here to hopefully make most systems able to run it.
    const int MAX_ITERATIONS = 512;

    int num = 0;
    // I have removed a conditional check for starting outside of the exit circle.
    // This check would darken the areas around the fractal slightly, making it seem
    // less cluttered, but only if there wasn't already something being subtracted from num.
    // Unfortunately, subtracting 1 makes it look bad in other ways.
    for (int i = 0; i < MAX_ITERATIONS; i++)
    {
        pos = square(pos) + c;
        if (dot(pos, pos) > 4)
        {
            // This makes it no longer precisely a buddhabrot render, but I think it looks cooler and less cluttered
            num = i - 10;
            break;
        }
    }
    pos = float2(0.0);
    for (int i = 0; i < num; i++)
    {
        pos = square(pos) + c;
        // Flip the x and y here so Buddhabrot is "sitting up"
        int index = getPixelIndex(float2(pos.y, pos.x), parameters.imageSize[0].imageWidth, parameters.imageSize[0].imageHeight);
        parameters.buffer[index].add(1);
    }
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
