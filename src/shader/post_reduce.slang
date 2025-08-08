#include "common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void post_reduce(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x == 0 && tid.y == 0 && tid.z == 0)
    {
        let num_bisectors : uint = pc.scene.cbt_interior[0];
        pc.dispatch.remaining_memory_count.store((int)(1 << 17) - (int)num_bisectors);
        pc.dispatch.num_allocated_blocks = num_bisectors;
        pc.dispatch.draw_indirect_command.vertex_count = 3 * num_bisectors;
        pc.dispatch.dispatch_vertex_compute_command.x = div_ceil(num_bisectors, WORKGROUP_SIZE);

        // pc.dispatch.debug_counter += 1;
        pc.scene.debug_counter += 1;
    }
}
