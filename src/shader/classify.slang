#include "cbt.slang"

TriState classify_triangle(float3[3] verts, uint heapid)
{
    let depth = heap_id_depth(heapid);
    if (heapid == 0b1000)
    {
        return TriState::BISECT_ELEMENT;
    }
    else
    {
        return TriState::UNCHANGED_ELEMENT;
    }
    // return TriState::UNCHANGED_ELEMENT;
    let normal = cross(verts[1] - verts[0], verts[2] - verts[1]);
    let normal_normalized = normalize(normal);
    if (dot(normal_normalized, pc.lookdir) > 0 - 10e-2)
    {
        return TriState::BACK_FACE_CULLED;
    }
    let tri_center = 0.3333 * (verts[0] + verts[1] + verts[2]);
    let center_to_cam = tri_center - pc.cam_pos;
    let center_to_cam_normalized = normalize(center_to_cam);
    let dist_to_cam = length(center_to_cam);

    // a is proportional to the screen area
    let a = dot(center_to_cam_normalized, normal) / dist_to_cam;
    if (a > 0.1 && (heapid & (1 << 31)) == 0)
    {
        // return TriState::BISECT_ELEMENT;
    }

    if (a < 0.01 && heap_id_depth(heapid) != pc.scene.base_depth)
    {
        return TriState::SIMPLIFY_ELEMENT;
    }

    return TriState::UNCHANGED_ELEMENT;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void classify(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.scene.cbt_interior[0])
    {
        return;
    }

    let curr_id = one_bit_to_id(tid.x);
    let heap_id = pc.scene.heapid_buffer[curr_id];
    // Reset some values
    pc.scene.currid_buffer[tid.x] = curr_id;
    // UNDO pc.scene.bisector_split_command_buffer[curr_id] = NO_SPLIT;
    // pc.scene.bisector_state_buffer[curr_id] = TriState::;

    float3 verts[3] = {
        pc.scene.vertex_buffer[curr_id * 3],
        pc.scene.vertex_buffer[curr_id * 3 + 1],
        pc.scene.vertex_buffer[curr_id * 3 + 2],
    };

    let state = classify_triangle(verts, heap_id);
    pc.scene.bisector_state_buffer[curr_id] = state;

    if (state == TriState::BISECT_ELEMENT)
    {
        // let bisector_slot = pc.dispatch.want_split_buffer_count.add(1);
        // pc.scene.want_split_buffer[bisector_slot] = curr_id;
        pc.dispatch.want_split_buffer_count = 1;
        pc.scene.want_split_buffer[0] = 0;
    }
    else if (state == TriState::SIMPLIFY_ELEMENT)
    {
        let bisector_slot = pc.dispatch.want_merge_buffer_count.add(1);
        pc.scene.want_merge_buffer[bisector_slot] = curr_id;
    }
}
