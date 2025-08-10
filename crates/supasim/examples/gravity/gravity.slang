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
    float gravityStrength;
    uint32_t binSize;
    float chunkWidth;
    uint32_t3 numChunks;
    // Should be very large for high precisions
    float accelerationPrecisionFactor;
}
struct Particle
{
    float3 position;
    float3 velocity;
    // Accelerations, as velocity may be used directly for some calculations. These are stored as quantized ints
    // so that they can be used with fast and widely supported integer atomics
    Atomic<int> ax;
    Atomic<int> ay;
    Atomic<int> az;
    uint32_t binIndex;
}
struct ChunkInfo
{
    // Lock is used to set the bin index
    Atomic<uint32_t> numParticles;
    Atomic<uint32_t> firstBinIndex;
    Atomic<uint32_t> lock;
}

struct Parameters
{
    StructuredBuffer<InputOptions> options;
    RWStructuredBuffer<Atomic<uint32_t>> numParticles;
    RWStructuredBuffer<Particle> particleBuffer;
    RWStructuredBuffer<ChunkInfo> chunkTracker;
    RWStructuredBuffer<uint32_t> bins;
}
