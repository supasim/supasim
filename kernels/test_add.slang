/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

struct Parameters
{
    [binding(0)]
    StructuredBuffer<uint32_t> buffer0;
    [binding(1)]
    StructuredBuffer<uint32_t> buffer1;
    [binding(2)]
    RWStructuredBuffer<uint32_t> result;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void add(uint3 threadId: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
    uint index = threadId.x;
    parameters.result[index] = parameters.buffer0[index] + parameters.buffer1[index];
}
