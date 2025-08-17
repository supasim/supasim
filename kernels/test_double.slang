/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

struct Parameters
{
    [binding(0)]
    RWStructuredBuffer<uint32_t> buffer;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void double(uint3 threadId: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
    uint index = threadId.x;
    parameters.buffer[index] = parameters.buffer[index] * 2;
}
