#include "common.slang"

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void split_element(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dispatch_next.want_split_buffer_count.load())
    {
        return;
    }
    var curr_id = pc.scene_next.want_split_buffer[tid.x];
    let neighbors_buffer = pc.scene_next.neighbors_buffer;
    let base_depth = pc.scene_next.base_depth;
    uint3 curr_neighbors = neighbors_buffer[curr_id];
    let splitting_buffer = pc.scene_next.splitting_buffer;
    let heapid_buffer = pc.scene_next.heapid_buffer;
    let bisector_command_buffer = pc.scene_next.bisector_split_command_buffer;
    let bisector_state_buffer = pc.scene_next.bisector_state_buffer;

    // check if the next neighbor is gonna split us anyway
    uint next = curr_neighbors[NEXT];
    if (next != INVALID_INDEX)
    {
        uint3 next_neighbors = neighbors_buffer[next];
        if (next_neighbors[TWIN] == curr_id && bisector_state_buffer[next] != TriState::UNCHANGED_ELEMENT)
        {
            return;
        }
    }

    // check if the prev neighbor is gonna split us anyway
    uint prev = curr_neighbors[PREV];
    if (prev != INVALID_INDEX)
    {
        uint3 prev_neighbors = neighbors_buffer[prev];
        if (prev_neighbors[TWIN] == curr_id && bisector_state_buffer[prev] != TriState::UNCHANGED_ELEMENT)
        {
            return;
        }
    }

    uint heapid = heapid_buffer[curr_id];       // 0b1000
    uint current_depth = heap_id_depth(heapid); // 3

    uint twin_id = curr_neighbors[TWIN]; // 3
    // check if we have enough memory to allocate compatibility chain elements
    int max_required_memory = (twin_id == INVALID_INDEX)                     ? 2                                     // boundary
                              : (neighbors_buffer[twin_id][TWIN] == curr_id) ? 4                                     // perfect matching
                                                                             : 3 * (current_depth - base_depth) + 2; // triple split for each entry in compatibility chain
    int remaining_memory = pc.dispatch_next.remaining_memory_count.add(-max_required_memory);
    if (remaining_memory < max_required_memory)
    {
        pc.dispatch_next.remaining_memory_count.add(max_required_memory); // undo subtraction
        return;
    }

    // check if another thread has already split this bisector
    uint base_pattern = bisector_command_buffer[curr_id].or(CENTER_SPLIT);
    if (base_pattern != NO_SPLIT)
    {
        pc.dispatch_next.remaining_memory_count.add(max_required_memory);
        return;
    }

    // good to split. start by splitting this bisector
    int used_memory = 2; // one for each split child
    uint target_location = pc.dispatch_next.splitting_buffer_count.add(1);
    splitting_buffer[target_location] = curr_id;

    // iterate over compabibility chain
    while (true)
    {
        // base case: boundary
        if (twin_id == INVALID_INDEX)
        {
            break;
        }

        uint twin_heapid = heapid_buffer[twin_id];
        uint twin_depth = heap_id_depth(twin_heapid);
        uint3 twin_neighbors = neighbors_buffer[twin_id];

        // base case: perfect match
        if (twin_neighbors[TWIN] == curr_id)
        {
            uint twin_prev_command = bisector_command_buffer[twin_id].or(CENTER_SPLIT);
            // check if twin is already split
            // if not, add to splitting buffer and claim memory
            if (twin_prev_command == NO_SPLIT)
            {
                uint alloc_loc = pc.dispatch_next.splitting_buffer_count.add(1);
                splitting_buffer[alloc_loc] = twin_id;
                used_memory += 2;
            }
            break;
        }
        // otherwise: twin needs a side split
        else
        {
            // split the twin
            uint twin_prev_command = (twin_neighbors[NEXT] == curr_id)
                                         ? bisector_command_buffer[twin_id].or(RIGHT_DOUBLE_SPLIT)
                                         : bisector_command_buffer[twin_id].or(LEFT_DOUBLE_SPLIT);

            // twin is already splitting
            // claim 1 slot for the side split and we're done
            if (twin_prev_command != NO_SPLIT)
            {
                used_memory += 1;
                break;
            }
            // twin has not been split at all
            // claim 3 slots for the double split and traverse to the twin
            // to handle its compatibility chain
            else
            {
                uint alloc_loc = pc.dispatch_next.splitting_buffer_count.add(1);
                splitting_buffer[alloc_loc] = twin_id;
                used_memory += 3;

                curr_id = twin_id;
                current_depth = twin_depth;
                twin_id = neighbors_buffer[curr_id][TWIN];
            }
        }

        // return unclaimed memory
        pc.dispatch_next.remaining_memory_count.add(max_required_memory - used_memory);
    }
}
