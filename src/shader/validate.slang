#include "vertex_compute.slang"

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void validate(uint3 tid: SV_DispatchThreadID)
{
    let vertex_buffer = pc.vertex_buffer;
    if (tid.x >= pc.scene.cbt_interior[0])
    {
        return;
    }
    var debug_data : Ptr<DebugData> = &pc.dispatch.debug_data;

    let curr_id = pc.scene.curr_id_buffer[tid.x];
    let neighbors = pc.scene.neighbors_buffer[curr_id];
    let vertices : float3[3] = {
        vertex_buffer[tid.x * 3],
        vertex_buffer[tid.x * 3 + 1],
        vertex_buffer[tid.x * 3 + 2],
    };

    // ensure each valid neighbor shares two vertices
    for (int i = 0; i < 3; ++i)
    {
        let neighbor = neighbors[i];
        if (neighbor == INVALID_INDEX)
        {
            continue;
        }

        let neighbor_heapid = pc.scene.heapid_buffer[neighbor];
        let neighbor_vertices = heapid_to_vertices(neighbor_heapid);

        // c\n 0 1 2
        //   0 x x x
        //   1   x x
        //   2     x
        var num_equal = 0;
        var equal_indices : int2[2] = { int2(-1, -1), int2(-1, -1) };
        // check if two of the vertices are equal
        for (int neighbor_vtx_idx = 0; neighbor_vtx_idx < 3; neighbor_vtx_idx++)
        {
            let neighbor_vert = neighbor_vertices[neighbor_vtx_idx];
            for (int curr_vtx_idx = 0; curr_vtx_idx < 3; curr_vtx_idx++)
            {
                let curr_vert = vertices[curr_vtx_idx];
                let dist = distance(neighbor_vert, curr_vert);
                if (dist < (33.333f / (1 << 17)))
                {
                    if (num_equal < 2)
                    {
                        equal_indices[num_equal] = int2(neighbor_vtx_idx, curr_vtx_idx);
                    }
                    num_equal++;
                }
            }
        }
        if (num_equal != 2)
        {
            let original = debug_data.curr_heapid.compareExchange(0, pc.scene.heapid_buffer[curr_id]);
            if (original == 0)
            {
                debug_data.curr_id = curr_id;
                debug_data.neighbor_id = neighbor;
                debug_data.neighbor_heapid = neighbor_heapid;
                debug_data.curr_verts = vertices;
                debug_data.neighbor_verts = neighbor_vertices;
                debug_data.num_equal = num_equal;
                debug_data.equal_indices = equal_indices;
            }
        }
    }
}
