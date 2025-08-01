#include "common.slang"

[shader("compute")]
void split_element(uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dispatch.want_split_buffer_count.load())
    {
        return;
    }
    var curr_id = pc.scene.want_split_buffer[tid.x];
    let neighbors_buffer = pc.scene.neighbors_buffer;
    let base_depth = pc.scene.base_depth;
    uint3 curr_neighbors = neighbors_buffer[curr_id];
    // var pc.dispatch.remaining_memory_count = pc.dispatch.remaining_memory_count;
    // var pc.dispatch.allocation_counter = pc.dispatch.allocation_counter;
    let allocation_buffer = pc.scene.splitting_buffer;
    let heapid_buffer = pc.scene.heapid_buffer;
    let bisector_command_buffer = pc.scene.bisector_split_command_buffer;
    let bisector_state_buffer = pc.scene.bisector_state_buffer;

    // Check NEXT neighbor
    uint next = curr_neighbors[NEXT];
    if (next != INVALID_INDEX)
    {
        uint3 next_neighbors = neighbors_buffer[next];
        if (next_neighbors[TWIN] == curr_id && bisector_state_buffer[next] != UNCHANGED_ELEMENT)
        {
            return;
        }
    }

    // Check PREV neighbor
    uint prev = curr_neighbors[PREV];
    if (prev != INVALID_INDEX)
    {
        uint3 prev_neighbors = neighbors_buffer[prev];
        if (prev_neighbors[TWIN] == curr_id && bisector_state_buffer[prev] != UNCHANGED_ELEMENT)
        {
            return;
        }
    }

    uint heapid = heapid_buffer[curr_id];
    uint current_depth = heap_id_depth(heapid);

    uint twin_id = curr_neighbors[TWIN];
    uint max_required_memory = (twin_id == INVALID_INDEX) ? 1 : (neighbors_buffer[twin_id][TWIN] == curr_id) ? 2
                                                                                                             : 2 * (current_depth - base_depth) - 1;

    uint remaining_memory = pc.dispatch.remaining_memory_count.add(-int(max_required_memory));

    if (remaining_memory < max_required_memory)
    {
        pc.dispatch.remaining_memory_count.add(int(max_required_memory)); // undo subtraction
        return;
    }

    uint base_pattern = bisector_command_buffer[curr_id].or(CENTER_SPLIT);
    if (base_pattern != NO_SPLIT)
    {
        pc.dispatch.remaining_memory_count.add(int(max_required_memory));
        return;
    }

    uint target_location = pc.dispatch.allocation_counter.add(1);
    allocation_buffer[target_location] = curr_id;

    uint used_memory = 1;

    while (true)
    {
        if (twin_id == INVALID_INDEX)
        {
            break;
        }

        uint twin_heapid = heapid_buffer[twin_id];
        uint twin_depth = heap_id_depth(twin_heapid);
        uint3 twin_neighbors = neighbors_buffer[twin_id];

        if (twin_depth == current_depth)
        {
            uint twin_prev_command = bisector_command_buffer[twin_id].or(CENTER_SPLIT);
            if (twin_prev_command == NO_SPLIT)
            {
                uint alloc_loc = pc.dispatch.allocation_counter.add(1);
                allocation_buffer[alloc_loc] = twin_id;
                used_memory += 1;
            }
            break;
        }
        else
        {
            uint twin_prev_command = (twin_neighbors[NEXT] == curr_id)
                                         ? bisector_command_buffer[twin_id].or(RIGHT_DOUBLE_SPLIT)
                                         : bisector_command_buffer[twin_id].or(LEFT_DOUBLE_SPLIT);

            if (twin_prev_command != NO_SPLIT)
            {
                used_memory += 1;
                break;
            }
            else
            {
                uint alloc_loc = pc.dispatch.allocation_counter.add(1);
                allocation_buffer[alloc_loc] = twin_id;
                used_memory += 2;

                curr_id = twin_id;
                current_depth = twin_depth;
                twin_id = neighbors_buffer[curr_id][TWIN];
            }
        }
    }

    pc.dispatch.remaining_memory_count.add(max((int(max_required_memory) - int(used_memory)), 0));
}
