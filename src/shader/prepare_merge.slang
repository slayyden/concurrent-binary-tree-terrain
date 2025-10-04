#include "cbt.slang"
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void prepare_merge(
    uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dispatch_next.want_merge_buffer_count.load())
    {
        return;
    }
    let curr_id = pc.scene_next.want_merge_buffer[tid.x];
    let neighbor_buffer = pc.scene_next.neighbors_buffer;
    let heap_id_buffer = pc.scene_next.heapid_buffer;
    let bisector_state_buffer = pc.scene_next.bisector_state_buffer;
    let simplification_buffer = pc.scene_next.merging_bisector_buffer;

    uint curr_heapid = heap_id_buffer[curr_id];
    uint3 curr_neighbors = neighbor_buffer[curr_id];
    uint curr_depth = heap_id_depth(curr_heapid);

    uint next_id = curr_neighbors[NEXT];
    uint next_heapid = heap_id_buffer[next_id];
    uint next_depth = heap_id_depth(next_heapid);
    TriState next_state = bisector_state_buffer[next_id];

    if (next_depth != curr_depth || next_state != TriState::SIMPLIFY_ELEMENT)
    {
        return;
    }

    uint3 next_neighbors = neighbor_buffer[next_id];
    uint num_pairs_to_merge = 1;

    uint twin_highid = next_neighbors[NEXT];
    uint twin_lowid = curr_neighbors[PREV];

    if (twin_lowid != INVALID_INDEX)
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

    // curr id 8A

    // 7 0.0 | 6 0.1
    // -----------------
    // 8 3.1 | 9 3.0
    // Remove reference from NEXT’s twin to NEXT (points back to curr)
    if (next_neighbors[TWIN] != INVALID_INDEX)
    {
        uint next_twin_id = next_neighbors[TWIN];                      // 5
        uint3 next_twin_neighbors = neighbor_buffer[next_twin_id];     // [9, 4, -1]
        uint edge_type = find_edge_type(next_id, next_twin_neighbors); // NEXT
        neighbor_buffer[next_twin_id][edge_type] = curr_id;            // 5 -> 9 = 8
    }

    uint base_slot = pc.dispatch_next.merging_bisector_count.add(num_pairs_to_merge);
    simplification_buffer[base_slot] = curr_id;

    heap_id_buffer[next_id] = 0;
    unset_bit(next_id);

    if (num_pairs_to_merge == 2)
    {
        simplification_buffer[base_slot + 1] = twin_highid;

        uint3 twin_low_neighbors = neighbor_buffer[twin_lowid]; // neighbors of 7 ([8, 6, 2])
        if (twin_low_neighbors[TWIN] != INVALID_INDEX)
        {
            uint twin_low_twin_id = twin_low_neighbors[TWIN];                     // 2
            uint3 twin_low_twin_neighbors = neighbor_buffer[twin_low_twin_id];    // [7, 1, -1]
            uint edge_type = find_edge_type(twin_lowid, twin_low_twin_neighbors); // NEXT
            neighbor_buffer[twin_low_twin_id][edge_type] = twin_highid;
        }
        heap_id_buffer[twin_lowid] = 0;
        unset_bit(twin_lowid);
    }
}
