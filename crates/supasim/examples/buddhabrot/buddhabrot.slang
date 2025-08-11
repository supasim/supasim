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
    uint32_t iterationSetIndex;
    uint32_t skipLastPoints;
    uint64_t workgroupDim;
    uint64_t randomSeed;
}
struct TempData
{
    float2 position;
    float2 c;
    int32_t numIterations;
    int32_t hasCompleted;
}

struct Parameters
{
    StructuredBuffer<InputOptions> imageSize;
    RWStructuredBuffer<Atomic<uint32_t>> buffer;
    RWStructuredBuffer<TempData> tempBuffer;
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

static const int ITERATIONS_PER = 512;

[shader("compute")]
[numthreads(16, 16, 1)]
void Run(uint3 id: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
    InputOptions options = parameters.imageSize[0];
    uint3 totalSize = uint3(16, 16, 1) * (uint)options.workgroupDim;
    uint index = id.z * totalSize.x * totalSize.y + id.y * totalSize.x + id.x;

    int iterationSetIndex = (int)parameters.imageSize[0].iterationSetIndex;
    TempData data = parameters.tempBuffer[index];

    if (iterationSetIndex == 0)
    {

        uint64_t seed = parameters.imageSize[0].randomSeed;

        uint seedX = Hash(uint(seed), index * 2);
        uint seedY = Hash(uint(seed), index * 2 + 1);

        float cx = UIntToFloat01(seedX) * 4.0 - 2.0;
        float cy = UIntToFloat01(seedY) * 4.0 - 2.0;

        data.c = float2(cx, cy);
        data.hasCompleted = 0;
        data.numIterations = 0;
        data.position = float2(0.0);
    }

    if (data.hasCompleted == 0)
    {
        for (int i = 0; i < ITERATIONS_PER && data.hasCompleted == 0; i++)
        {
            data.position = square(data.position) + data.c;
            if (dot(data.position, data.position) > 4)
            {
                data.hasCompleted = 1;
                data.numIterations = iterationSetIndex * ITERATIONS_PER + i + 1;
                break;
            }
        }
    }
    parameters.tempBuffer[index] = data;
}

[shader("compute")]
[numthreads(16, 16, 1)]
void Finalize(uint3 id: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
    InputOptions options = parameters.imageSize[0];
    uint3 totalSize = uint3(16, 16, 1) * (uint)options.workgroupDim;
    uint index = id.z * totalSize.x * totalSize.y + id.y * totalSize.x + id.x;

    TempData data = parameters.tempBuffer[index];
    if (options.iterationSetIndex == 0)
    {
        data.position = float2(0.0);
    }
    if (data.hasCompleted == 1)
    {
        for (int i = 0; i < min(ITERATIONS_PER, data.numIterations - (int)options.skipLastPoints - ITERATIONS_PER * (int)options.iterationSetIndex); i++)
        {
            data.position = square(data.position) + data.c;
            int pixelIndex = getPixelIndex(float2(data.position.y, data.position.x), parameters.imageSize[0].imageWidth, parameters.imageSize[0].imageHeight);
            parameters.buffer[pixelIndex].add(1);
        }
    }
    parameters.tempBuffer[index] = data;
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
