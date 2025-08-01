#include "cbt.slang"

void classify(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.scene.cbt_interior[0])
    {
        return;
    }

    let curr_id = one_bit_to_id(tid.x);
    let heap_id = pc.scene.heapid_buffer[curr_id];
    float3 verts[3] = {
        pc.scene.vertex_buffer[curr_id * 3],
        pc.scene.vertex_buffer[curr_id * 3 + 1],
        pc.scene.vertex_buffer[curr_id * 3 + 2],
    };

    let average_position = 0.3333 * (verts[0] + verts[1] + verts[2]);
    if (dot(average_position, average_position) < 0.5 && (heap_id & (1 << 31)) == 0)
    {
        let bisector_slot = pc.dispatch.want_split_buffer_count.add(1);
        pc.scene.bisector_split_command_buffer[bisector_slot] = curr_id;
    }
}
