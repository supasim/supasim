/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

// You only need to check odd numbers. Even numbers instantly decrease

struct GlobalData
{
    uint64_t minLimit;
    uint64_t maxLimit;
    uint64_t numUnsolved;
    uint64_t newNumUnsolved;
    uint64_t smallestUnsolved;

    uint32_t3 dispatchSize;
    uint32_t unsolvedBufferSize;
}

struct UnsolvedElement
{
    uint64_t number;
    uint64_t currentValue;
    uint64_t attemptsSoFar;
}

struct Parameters
{
    [binding(0)]
    RWStructuredBuffer<GlobalData> globalData;
    [binding(1)]
    RWStructuredBuffer<UnsolvedElement> unsolved;
}

static const int NUM_ITERATIONS = 16;

groupshared uint64_t collatzNumUnsolved;
[shader("compute")]
[numthreads(16, 16, 1)]
void collatzMain(uint3 id: SV_DispatchThreadID, ParameterBlock<Parameters> params, uint id_in_wg: SV_GroupIndex)
{
    uint3 totalSize = uint3(16, 16, 1) * params.globalData[0].dispatchSize;
    uint totalCount = totalSize.x * totalSize.y * totalSize.z;
    uint index = id.z * totalSize.x * totalSize.y + id.y * totalSize.x + id.x;

    UnsolvedElement element;
    if (index < params.globalData[0].numUnsolved)
    {
        element = params.unsolved[index];
    }
    else
    {
        element.attemptsSoFar = 0;
        element.number = params.globalData[0].maxLimit - params.globalData[0].numUnsolved + index * 2;
        element.currentValue = element.number;
    }
    bool solved = false;
    for (int i = 0; i < NUM_ITERATIONS; i++)
    {
        if (element.currentValue < element.number)
        {
            solved = true;
            break;
        }
        else if ((element.currentValue % 2) == 0)
        {
            element.currentValue /= 2;
        }
        else
        {
            element.currentValue *= 3;
            element.currentValue += 1;
        }
    }
    if (id_in_wg == 0)
    {
        collatzNumUnsolved = 0;
    }
    GroupMemoryBarrierWithGroupSync();
    uint64_t myIndexInUnsolved;
    if (!solved)
    {
        InterlockedAdd(collatzNumUnsolved, 1, myIndexInUnsolved);
    }
    GroupMemoryBarrierWithGroupSync();
    if (collatzNumUnsolved > 0)
    {
        if (id_in_wg == 0)
        {
            InterlockedAdd(params.globalData[0].newNumUnsolved, collatzNumUnsolved, collatzNumUnsolved);
        }
        GroupMemoryBarrierWithGroupSync();
        if (!solved)
        {
            params.unsolved[params.globalData[0].unsolvedBufferSize / 2 + collatzNumUnsolved + myIndexInUnsolved];
        }
    }
}

groupshared uint64_t sortSmallest;
[shader("compute")]
[numthreads(16, 16, 1)]
void sortUnsolved1(uint3 id: SV_DispatchThreadID, ParameterBlock<Parameters> params, uint id_in_wg: SV_GroupIndex)
{
    uint3 totalSize = uint3(16, 16, 1) * params.globalData[0].dispatchSize;
    uint totalCount = totalSize.x * totalSize.y * totalSize.z;
    uint index = id.z * totalSize.x * totalSize.y + id.y * totalSize.x + id.x;
    if (index == 0)
    {
        params.globalData[0].maxLimit += 2 * (totalCount - params.globalData[0].numUnsolved);
        params.globalData[0].numUnsolved = params.globalData[0].newNumUnsolved;
        sortSmallest = uint64_t.maxValue;
    }

    GroupMemoryBarrierWithGroupSync();
    for (int i = index; i < params.globalData[0].newNumUnsolved; i++)
    {

        UnsolvedElement element = params.unsolved[params.globalData[0].unsolvedBufferSize / 2 + i];
        InterlockedMin(sortSmallest, element.number);
        params.unsolved[i] = element;
    }
    GroupMemoryBarrierWithGroupSync();
    if (id_in_wg == 0)
    {
        InterlockedMin(params.globalData[0].smallestUnsolved, sortSmallest);
    }
}

[shader("compute")]
[numthreads(1, 1, 1)]
void sortUnsolved2(uint3 id: SV_DispatchThreadID, ParameterBlock<Parameters> params)
{
    params.globalData[0].newNumUnsolved = 0;
    if (params.globalData[0].smallestUnsolved == uint64_t.maxValue)
    {
        params.globalData[0].smallestUnsolved = params.globalData[0].maxLimit;
    }
    params.globalData[0].minLimit = params.globalData[0].smallestUnsolved;
    params.globalData[0].smallestUnsolved = uint64_t.maxValue;
}
