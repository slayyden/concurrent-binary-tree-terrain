#include "common.slang"
[shader("compute")]
[numthreads(1024)]
void prepare_merge(
    uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dispatch.want_merge_buffer_count.load())
    {
        return;
    }
    let curr_id = pc.scene.want_merge_buffer[tid.x];
    let neighbor_buffer = pc.scene.neighbors_buffer;
    let heap_id_buffer = pc.scene.heapid_buffer;
    let bisector_state_buffer = pc.scene.bisector_state_buffer;
    let simplification_buffer = pc.scene.merging_bisector_buffer;

    uint curr_heapid = heap_id_buffer[curr_id];
    uint3 curr_neighbors = neighbor_buffer[curr_id];
    uint curr_depth = heap_id_depth(curr_heapid);

    uint next_id = curr_neighbors[NEXT];
    uint next_heapid = heap_id_buffer[next_id];
    uint next_depth = heap_id_depth(next_heapid);
    TriState next_state = bisector_state_buffer[next_id];

    if (next_depth != curr_depth || next_state != SIMPLIFY)
    {
        return;
    }

    uint3 next_neighbors = neighbor_buffer[next_id];
    uint num_pairs_to_merge = 1;

    uint twin_highid = next_neighbors[NEXT];
    uint twin_lowid = curr_neighbors[PREV];

    if (twin_lowid != 0xFFFFFFFF)
    {
        uint twin_low_heapid = heap_id_buffer[twin_lowid];
        uint twin_high_heapid = heap_id_buffer[twin_highid];

        if (curr_heapid < twin_high_heapid)
        {
            return;
        }

        uint twin_low_depth = heap_id_depth(twin_low_heapid);
        uint twin_high_depth = heap_id_depth(twin_high_heapid);

        if (twin_low_depth != curr_depth || twin_high_depth != curr_depth)
        {
            return;
        }

        TriState twin_low_state = bisector_state_buffer[twin_lowid];
        TriState twin_high_state = bisector_state_buffer[twin_highid];

        if (twin_low_state != TriState::SIMPLIFY_ELEMENT || twin_high_state != TriState::SIMPLIFY_ELEMENT)
        {
            return;
        }

        num_pairs_to_merge = 2;
    }

    // Remove reference from NEXT’s twin to NEXT (points back to curr)
    if (next_neighbors[TWIN] != 0xFFFFFFFF)
    {
        uint next_twin_id = next_neighbors[TWIN];
        uint3 next_twin_neighbors = neighbor_buffer[next_twin_id];
        uint edge_type = find_edge_type(next_id, next_twin_neighbors);
        neighbor_buffer[next_twin_id][edge_type] = curr_id;
    }

    uint base_slot = pc.dispatch.merging_bisector_count.add(num_pairs_to_merge);
    simplification_buffer[base_slot] = curr_id;

    if (num_pairs_to_merge == 2)
    {
        simplification_buffer[base_slot + 1] = twin_highid;

        uint3 twin_low_neighbors = neighbor_buffer[twin_lowid];
        if (twin_low_neighbors[TWIN] != 0xFFFFFFFF)
        {
            uint twin_low_twin_id = twin_low_neighbors[TWIN];
            uint3 twin_low_twin_neighbors = neighbor_buffer[twin_low_twin_id];
            uint edge_type = find_edge_type(twin_lowid, twin_low_twin_neighbors);
            neighbor_buffer[twin_low_twin_id][edge_type] = twin_highid;
        }
    }
}
