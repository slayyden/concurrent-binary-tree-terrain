
#include "common.slang"

[shader("compute")]
[numthreads(1024)]
void reduce(uint3 threadId: SV_DispatchThreadID)
{
    let interior_offset = num_internal / 2;
    let tid = threadId.x;
    let scene = pc.scene;

    for (int i = 0; i < 4; ++i)
    {
        let j = tid * 4 + i;
        if (j <= interior_offset)
        {
            scene.cbt_interior[interior_offset + j] = countbits(scene.cbt_leaves[j]);
        }
    }

    DeviceMemoryBarrierWithGroupSync();
    // index_of_last_level = depth
    // 32 (depth) -> 16 (depth - 1) -> 8 (depth - 2) -> 4 (depth - 3) -> 2 (depth - 4) -> 1 (depth - 5)
    // all these levels have been filled
    let deepest_filled_level = scene.cbt_depth - 5;

    for (uint level = 0; level < deepest_filled_level; ++level)
    {
        let level_start = (1 << level) - 1;
        let level_end = (1 << (level + 1)) - 1;
        for (int i = 0; i < 4; ++i)
        {
            let j = tid * 4 + i;
            if (level_start <= j && j < level_end)
            {
                scene.cbt_interior[j] = scene.cbt_interior[2 * j + 1] + scene.cbt_interior[2 * j + 2];
            }
        }
        DeviceMemoryBarrierWithGroupSync();
    }

    if (tid.x == 0)
    {
        pc.dispatch.num_allocated_blocks = scene.cbt_interior[0];
    }
}

