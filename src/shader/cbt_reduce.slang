
#include "common.slang"

[shader("compute")]
void reduce(uint3 threadId: SV_DispatchThreadID)
{
    let interior_offset = num_internal / 2;
    let tid = threadId.x;
    let scene = pc.scene;
    if (tid <= interior_offset)
    {
        scene.cbt_interior[interior_offset + tid] = countbits(scene.cbt_leaves[tid]);
    }
    // index_of_last_level = depth
    // 32 (depth) -> 16 (depth - 1) -> 8 (depth - 2) -> 4 (depth - 3) -> 2 (depth - 4) -> 1 (depth - 5)
    // all these levels have been filled
    let deepest_filled_level = scene.cbt_depth - 5;

    for (uint level = 0; level < deepest_filled_level; ++level)
    {
        let level_start = (1 << level) - 1;
        let level_end = (1 << (level + 1)) - 1;
        if (level_start <= tid && tid < level_end)
        {
            scene.cbt_interior[tid] = scene.cbt_interior[2 * tid + 1] + scene.cbt_interior[2 * tid + 2];
        }
        GroupMemoryBarrierWithGroupSync();
    }
}

