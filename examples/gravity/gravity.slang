/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
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
