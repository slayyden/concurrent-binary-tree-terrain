#include "cbt.slang"

struct AABB {
    float3 min;
    float3 max;
}

AABB triangle_aabb(float3[3] verts) {
    let min = float3(
        min3(verts[0].x, verts[1].x, verts[2].x),
        min3(verts[0].y, verts[1].y, verts[2].y),
        min3(verts[0].z, verts[1].z, verts[2].z),
    );
    let max = float3(
        max3(verts[0].x, verts[1].x, verts[2].x),
        max3(verts[0].y, verts[1].y, verts[2].y),
        max3(verts[0].z, verts[1].z, verts[2].z),
    );
    return AABB(min, max);
}

// returns true iff the point q lies within the frustum 
bool in_frustum(float3 q, float horizontal_slope, float vertical_slope) {
	bool in_front_of_camera = q.z > 0;
	bool in_horizontal      = q.x <= q.z * horizontal_slope;
	bool in_vertical        = q.y <= q.z * vertical_slope;
	return in_front_of_camera && in_horizontal && in_vertical;
}


TriState classify_triangle(float3[3] verts, uint heapid)
{
    let uniform_buffer = pc.uniform_buffer;
    let scene = uniform_buffer.scene_next;
    let dispatch = uniform_buffer.dispatch_next;

    let normal = cross(verts[1] - verts[0], verts[2] - verts[1]);

    // backface culling
    let normal_normalized = normalize(normal);
    let VdotN = dot(normal_normalized, uniform_buffer.lookdir);
    if (VdotN > 0 + 10e-2)
    {
        return TriState::BACK_FACE_CULLED;
    }

    // frustum culling
    // TODO: use view matrix
    // vertices in camera coords
    let vertices_camera : float3[3] = {
        mul(float4(verts[0], 1), uniform_buffer.view).xyz,
        mul(float4(verts[1], 1), uniform_buffer.view).xyz,
        mul(float4(verts[2], 1), uniform_buffer.view).xyz,
    };
    let aabb = triangle_aabb(vertices_camera);
    let horizontal_slope = uniform_buffer.frustum_slopes.horizontal_slope;
    let vertical_slope = uniform_buffer.frustum_slopes.vertical_slope;
    let in_frustum = in_frustum(aabb.min, horizontal_slope, vertical_slope) || in_frustum(aabb.max, horizontal_slope, vertical_slope);
    if (!in_frustum) {
        return TriState::SIMPLIFY_ELEMENT;
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