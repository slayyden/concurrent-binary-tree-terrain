
#include "common.slang"

[shader("compute")]
[numthreads(1024, 1, 1)]
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
            scene.cbt_interior[interior_offset + j] = countbits(scene.cbt_leaves[j].load());
        }
    }

    DeviceMemoryBarrier();
    // index_of_last_level = depth
    // 32 (depth) -> 16 (depth - 1) -> 8 (depth - 2) -> 4 (depth - 3) -> 2 (depth - 4) -> 1 (depth - 5)
    // all these levels have been filled
    let deepest_filled_level = scene.cbt_depth - 5;

    for (int level = deepest_filled_level - 1; level >= 0; --level)
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
        DeviceMemoryBarrier();
    }

    if (tid.x == 0)
    {
        let num_bisectors = scene.cbt_interior[0];
        pc.dispatch.remaining_memory_count = (1 << 17) - num_bisectors;
        pc.dispatch.num_allocated_blocks = num_bisectors;
        pc.dispatch.draw_indirect_command.vertex_count = num_bisectors * 3;
        pc.dispatch.dispatch_vertex_compute_command.x = div_ceil(num_bisectors, WORKGROUP_SIZE);
    }
}

