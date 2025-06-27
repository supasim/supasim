struct InputOptions {
    float gravityStrength;
    uint32_t binSize;
    float chunkWidth;
}
struct Particle {
    float3 position;
    float3 velocity;
    // Velocity may be used directly for some calculations
    Atomic<float3> acceleration;
    uint32_t binIndex;
}
struct ChunkInfo {
    Atomic<uint32_t> numParticles;
    Atomic<uint32_t> firstBinIndex;
    Atomic<bool> lock;
}

struct Parameters {
    StructuredBuffer<InputOptions> options;
    RWStructuredBuffer<Atomic<uint32_t>> numParticles;
    RWStructuredBuffer<Particle> particleBuffer;
    RWStructuredBuffer<ChunkInfo> chunkTracker;
    RWStructuredBuffer<uint32_t> bins;
}