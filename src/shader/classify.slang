#include "cbt.slang"

TriState classify_triangle(float3[3] verts, uint heapid)
{
    if (/*heapid % 8 == 0 &&*/ heap_id_depth(heapid) < 28)
    {
        return TriState::BISECT_ELEMENT;
    }
    return TriState::UNCHANGED_ELEMENT;
    let normal = cross(verts[1] - verts[0], verts[2] - verts[1]);
    /*
    let depth = heap_id_depth(heapid);
    if (depth < 25)
    {
        return TriState::BISECT_ELEMENT;
    }
    return TriState::UNCHANGED_ELEMENT;*/

    let normal_normalized = normalize(normal);
    if (dot(normal_normalized, pc.lookdir) > 0 + 10e-2)
    {
        return TriState::BACK_FACE_CULLED;
    }

    var v0_projected = mul(float4(verts[0], 1), pc.view_project);
    var v1_projected = mul(float4(verts[1], 1), pc.view_project);
    var v2_projected = mul(float4(verts[2], 1), pc.view_project);
    v0_projected.xy /= v0_projected.w;
    v1_projected.xy /= v1_projected.w;
    v2_projected.xy /= v2_projected.w;
    v0_projected.xy = map_range(v0_projected.xy, float2(-1, -1), float2(1, 1), float2(0, 0), float2(SCREEN_WIDTH, SCREEN_HEIGHT));
    v1_projected.xy = map_range(v1_projected.xy, float2(-1, -1), float2(1, 1), float2(0, 0), float2(SCREEN_WIDTH, SCREEN_HEIGHT));
    v2_projected.xy = map_range(v2_projected.xy, float2(-1, -1), float2(1, 1), float2(0, 0), float2(SCREEN_WIDTH, SCREEN_HEIGHT));

    let edge0 = v1_projected.xy - v0_projected.xy;
    let edge1 = v2_projected.xy - v0_projected.xy;
    let det = abs(edge0.x * edge1.y - edge0.y * edge1.x);
    let area = (0.5 * det) / (SCREEN_HEIGHT * SCREEN_WIDTH);
    /*
    let tri_center = 0.3333 * (verts[0] + verts[1] + verts[2]);
    let center_to_cam = tri_center - pc.cam_pos;
    let center_to_cam_normalized = normalize(center_to_cam);
    let dist_to_cam = length(center_to_cam);
    let area = length(normal) / 2.0;*/

    // a is proportional to the screen area
    if (area > 1 && heap_id_depth(heapid) < 18)
    {
        return TriState::BISECT_ELEMENT;
    }
    return TriState::UNCHANGED_ELEMENT;

    if (area < 0.001 && heap_id_depth(heapid) != pc.scene.base_depth)
    {
        return TriState::SIMPLIFY_ELEMENT;
    }
}

[shader("compute")]
[numthreads(64, 1, 1)]
void classify(uint3 tid: SV_DispatchThreadID)
{
    let vertex_buffer = pc.scene.vertex_buffer;
    if (tid.x >= pc.scene.cbt_interior[0])
    {
        return;
    }

    // let curr_id = one_bit_to_id(tid.x);
    let curr_id = pc.scene.curr_id_buffer[tid.x];
    let heap_id = pc.scene.heapid_buffer[curr_id];
    // Reset some values
    // pc.scene.currid_buffer[tid.x] = curr_id;
    pc.scene.bisector_split_command_buffer[curr_id].store(NO_SPLIT);
    pc.scene.allocation_indices_buffer[curr_id] = uint4(INVALID_INDEX);
    // pc.scene.bisector_state_buffer[curr_id] = TriState::;

    float3 verts[3] = {
        get_vertex_buffer(pc, curr_id * 3),
        get_vertex_buffer(pc, curr_id * 3 + 1),
        get_vertex_buffer(pc, curr_id * 3 + 2)
    };

    let state = classify_triangle(verts, heap_id);
    pc.scene.bisector_state_buffer[curr_id] = state;

    if (state == TriState::BISECT_ELEMENT)
    {
        let bisector_slot = pc.dispatch.want_split_buffer_count.add(1);
        pc.scene.want_split_buffer[bisector_slot] = curr_id;
    }
    else if (state == TriState::SIMPLIFY_ELEMENT && heap_id % 2 == 1)
    {
        let bisector_slot = pc.dispatch.want_merge_buffer_count.add(1);
        pc.scene.want_merge_buffer[bisector_slot] = curr_id;
    }
}
