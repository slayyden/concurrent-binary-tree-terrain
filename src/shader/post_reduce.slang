#include "common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void post_reduce(uint3 tid: SV_DispatchThreadID)
{
    let uniform_buffer = pc.uniform_buffer;
    let scene = uniform_buffer.scene_next;
    let dispatch = uniform_buffer.dispatch_next;
    if (tid.x == 0 && tid.y == 0 && tid.z == 0)
    {
        let num_bisectors : uint = scene.cbt_interior[0];
        dispatch.remaining_memory_count.store((int)(1 << 17) - (int)num_bisectors);
        dispatch.num_allocated_blocks = num_bisectors;
        dispatch.draw_indirect_command.vertex_count = 3 * num_bisectors;
        dispatch.dispatch_vertex_compute_command.x = div_ceil(num_bisectors, WORKGROUP_SIZE);
    }
}
