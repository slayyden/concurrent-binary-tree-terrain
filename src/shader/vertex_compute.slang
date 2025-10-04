#include "cbt.slang"
float3[3] heapid_to_vertices(
    uint heapid)
{
    // heapid: 0b000...01 root_bisector_index split_code
    // the most significant 1 is a "tag" that allows us to compute the length of the split code
    // root_bisector_index takes `base_level` bits
    // split_code takes up a number of bits equal to the number of tree edges from the bisector to its root bisector
    let base_level = pc.scene_next.base_depth;
    let root_bisector_vertices = pc.scene_next.root_bisector_vertices;
    let depth = heap_id_depth(heapid);
    let num_split_code_bits = depth - base_level;

    let heapid_without_tag = (~(1 << depth)) & heapid;
    let root_bisector_index = heapid_without_tag >> num_split_code_bits;
    let split_code = (~(0xFFFF_FFFF << num_split_code_bits)) & heapid;

    float3 curr_bisector[3] = { root_bisector_vertices[root_bisector_index * 3],
                                root_bisector_vertices[root_bisector_index * 3 + 1],
                                root_bisector_vertices[root_bisector_index * 3 + 2] };

    if (depth == base_level)
    {
        return curr_bisector;
    }

    uint peak_idx = 2;
    for (int i = depth - base_level - 1; i >= 0; --i)
    {
        let bit = (split_code >> i) & 0b0000_0000_0000_0001;

        let replacement_slot = select((bit == 0), peak_idx + 2, peak_idx + 1) % 3;
        let refinement_edge_0 = curr_bisector[((peak_idx + 1) % 3)];
        let refinement_edge_1 = curr_bisector[((peak_idx + 2) % 3)];

        let new_vertex = 0.5 * (refinement_edge_0 + refinement_edge_1);
        curr_bisector[replacement_slot] = new_vertex;
        peak_idx = replacement_slot;
    };

    return curr_bisector;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void vertex_compute(uint3 tid: SV_DispatchThreadID)
{
    let vertex_buffer = pc.scene_next.vertex_buffer;
    if (tid.x >= pc.scene_next.cbt_interior[0])
    {
        return;
    }

    let curr_id = one_bit_to_id(tid.x);
    pc.scene_next.curr_id_buffer[tid.x] = curr_id;
    let heapid = pc.scene_next.heapid_buffer[curr_id];
    // let vertices_slot = pc.dispatch.vertex_buffer_count.add(1);
    let vertices_slot = tid.x;
    let triangle = heapid_to_vertices(heapid);

    // displace verts here
    vertex_buffer[vertices_slot * 3] = triangle[0];
    vertex_buffer[vertices_slot * 3 + 1] = triangle[1];
    vertex_buffer[vertices_slot * 3 + 2] = triangle[2];
}
