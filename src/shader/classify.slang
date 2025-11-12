#include "cbt.slang"

TriState classify_triangle(float3[3] verts, uint heapid)
{
    let uniform_buffer = pc.uniform_buffer;
    let scene = uniform_buffer.scene_next;
    let dispatch = uniform_buffer.dispatch_next;

    let normal = cross(verts[1] - verts[0], verts[2] - verts[1]);

    let normal_normalized = normalize(normal);
    let VdotN = dot(normal_normalized, uniform_buffer.lookdir);
    if (VdotN > 0 + 10e-2)
    {
        return TriState::BACK_FACE_CULLED;
    }

    let area = tri_area(verts, VdotN);

    // a is proportional to the screen area
    if (area > 0.01 && heap_id_depth(heapid) < 29)
    {
        return TriState::BISECT_ELEMENT;
    }

    if (area < 0.001 && heap_id_depth(heapid) != scene.base_depth)
    {
        return TriState::SIMPLIFY_ELEMENT;
    }
    return TriState::UNCHANGED_ELEMENT;
}

[shader("compute")]
[numthreads(64, 1, 1)]
void classify(uint3 tid: SV_DispatchThreadID)
{
    let uniform_buffer = pc.uniform_buffer;
    let scene = uniform_buffer.scene_next;
    let dispatch = uniform_buffer.dispatch_next;
    let vertex_buffer = scene.vertex_buffer;
    if (tid.x >= scene.cbt_interior[0])
    {
        return;
    }

    // let curr_id = one_bit_to_id(tid.x);
    let curr_id = scene.curr_id_buffer[tid.x];
    let heap_id = scene.heapid_buffer[curr_id];
    // Reset some values
    // pc.scene_next.currid_buffer[tid.x] = curr_id;
    scene.bisector_split_command_buffer[curr_id].store(NO_SPLIT);
    scene.allocation_indices_buffer[curr_id] = uint4(INVALID_INDEX);
    // pc.scene_next.bisector_state_buffer[curr_id] = TriState::;

    float3 verts[3] = {
        vertex_buffer[curr_id * 3],
        vertex_buffer[curr_id * 3 + 1],
        vertex_buffer[curr_id * 3 + 2],
    };

    let state = classify_triangle(verts, heap_id);
    scene.bisector_state_buffer[curr_id] = state;

    if (state == TriState::BISECT_ELEMENT)
    {
        let bisector_slot = dispatch.want_split_buffer_count.add(1);
        scene.want_split_buffer[bisector_slot] = curr_id;
    }
    else if (state == TriState::SIMPLIFY_ELEMENT && heap_id % 2 == 1)
    {
        let bisector_slot = dispatch.want_merge_buffer_count.add(1);
        scene.want_merge_buffer[bisector_slot] = curr_id;
    }
}