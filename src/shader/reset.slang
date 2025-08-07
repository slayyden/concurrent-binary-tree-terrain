#include "common.slang"

/*
struct PushConstant
{
    float4x4 view_project;
    float3 *positions;
    uint32_t *curr_id;
    SceneData *scene;
    DispatchData *dispatch;
}

struct DispatchData
{
    DrawIndirectCommand draw_indirect_command;
    Atomic<uint> remaining_memory_count; // written at reset
    uint2 pad0;
    Atomic<uint> allocation_counter;
    uint2 pad1;
    Atomic<uint> want_split_buffer_count;
    uint2 pad3;
    Atomic<uint> splitting_buffer_count;
    uint2 pad5;
    Atomic<uint> want_merge_buffer_count;
    uint2 pad4;
    Atomic<uint> merging_bisector_count;
    uint2 pad2;
    uint num_allocated_blocks; // NOT reset
    uint2 pad7;
    Atomic<uint> vertex_buffer_count;
}
*/

[shader("compute")]
[numthreads(1, 1, 1)]
void reset(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x != 0 || tid.y != 0 || tid.z != 0)
    {
        return;
    }
    let dispatch = pc.dispatch;
    dispatch.num_allocated_blocks = pc.scene.cbt_interior[0];
    dispatch.allocation_counter.store(0);
    dispatch.want_split_buffer_count.store(0);
    dispatch.splitting_buffer_count.store(0);
    dispatch.want_merge_buffer_count.store(0);
    dispatch.merging_bisector_count.store(0);
}
