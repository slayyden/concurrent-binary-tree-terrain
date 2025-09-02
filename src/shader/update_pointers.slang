#include "cbt.slang"
// Lookup table mapping split command + edge index => edge slots

static const uint2 COMMAND_EDGE_LUT[4][3] = {
    // CENTRAL_SPLIT
    { uint2(0, 0), uint2(1, 1), uint2(1, 0) },
    // RIGHT_DOUBLE_SPLIT
    { uint2(2, 0), uint2(1, 1), uint2(1, 2) },
    // LEFT_DOUBLE_SPLIT
    { uint2(0, 0), uint2(2, 1), uint2(1, 0) },
    // TRIPLE_SPLIT
    { uint2(2, 0), uint2(3, 1), uint2(1, 2) }
};

// Given a split command and an edge index (NEXT, PREV, or TWIN)
// Return up to two allocation slots that touch each half of the edge
// If the edge has not been split, these slots will be the same
// The first slot appears on the left when viewing the triangle from the outside
// with the specified edge at the bottom and its opposing vertex at the top
uint2 get_edge_slots(uint command, uint edge_type)
{
    if (command == 0) // NO_SPLIT
        return uint2(0, 0);
    return (COMMAND_EDGE_LUT[command >> 1])[edge_type];
}

// Child edge type LUT
static const uint CHILD_EDGE_TYPE_LUT[4][3][2] = {
    // CENTRAL_SPLIT
    { { TWIN, TWIN }, { TWIN, TWIN }, { NEXT, PREV } },
    // RIGHT_DOUBLE_SPLIT
    { { NEXT, PREV }, { TWIN, TWIN }, { NEXT, TWIN } },
    // LEFT_DOUBLE_SPLIT
    { { TWIN, TWIN }, { NEXT, PREV }, { TWIN, PREV } },
    // TRIPLE_SPLIT
    { { NEXT, PREV }, { NEXT, PREV }, { TWIN, TWIN } }
};

// Given a split command and a parent edge type,
// return the types of the children’s edges corresponding to that edge
uint[2] get_child_edge_types(uint command, uint parent_edge_type)
{
    return CHILD_EDGE_TYPE_LUT[command >> 1][parent_edge_type];
}

// Link sibling edges by updating their neighbor buffer entries
void link_siblings(uint prev_sibling, uint next_sibling)
{
    (pc.scene.neighbors_buffer[prev_sibling])[NEXT] = next_sibling;
    (pc.scene.neighbors_buffer[next_sibling])[PREV] = prev_sibling;
}

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void update_pointers(
    int3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dispatch.splitting_buffer_count.load())
    {
        return;
    }
    let curr_id = pc.scene.splitting_buffer[tid.x];
    let heap_id_buffer = pc.scene.heapid_buffer;
    let allocation_indices_buffer = pc.scene.allocation_indices_buffer;
    let bisector_command_buffer = (uint *)pc.scene.bisector_split_command_buffer;
    let neighbor_buffer = pc.scene.neighbors_buffer;
    let curr_heapid = heap_id_buffer[curr_id];
    let curr_command = bisector_command_buffer[curr_id];

    let neighbors = neighbor_buffer[curr_id];
    let curr_allocation_indices = pc.scene.allocation_indices_buffer[curr_id];

    // Clear current bit and set new ones
    unset_bit(curr_id);
    let num_allocations = countbits(curr_command) + 1;
    for (int i = 0; i < num_allocations; ++i)
    {
        let allocation_index = curr_allocation_indices[i];
        set_bit(allocation_index);
    }

    // Loop over each edge (NEXT, PREV, TWIN)
    for (uint bisector_edge_idx = 0; bisector_edge_idx < 3; ++bisector_edge_idx)
    {
        uint neighbor_index = neighbors[bisector_edge_idx];
        // get allocation slots of this edge in the current bisector
        uint2 bisector_slots = get_edge_slots(curr_command, bisector_edge_idx);
        // get edge types of neighboring bisector's adjacent children
        let edge_types = get_child_edge_types(curr_command, bisector_edge_idx);

        // setup variables as if the neighbor did not exist
        uint4 neighbor_allocation_indices = uint4(INVALID_INDEX);
        // slot of neighbor to write to for first half of edge
        uint neighbor_slot = INVALID_INDEX;
        // slots of neighbor that are adjacent to this bisector edge
        uint2 neighbor_slots = uint2(0, 0);

        // by default, we load into the first slot and the first subedge
        uint slot = curr_allocation_indices[bisector_slots[0]];
        uint edge_type = edge_types[0];

        if (neighbor_index != INVALID_INDEX)
        {
            uint neighbor_command = bisector_command_buffer[neighbor_index];
            uint neighbor_edge = find_edge_type(curr_id, neighbor_buffer[neighbor_index]);

            // non split tris are not updated so we must update them ourselves
            if (neighbor_command == NO_SPLIT)
            {
                // if (curr_id != slot)
                // {
                // (neighbor_buffer[neighbor_index])[neighbor_edge] = slot;
                // }
                (neighbor_buffer[neighbor_index])[neighbor_edge] = slot;
                neighbor_slot = neighbor_index;
                if (slot == INVALID_INDEX)
                {
                    pc.dispatch.debug_data.curr_heapid.add(1);
                }
            }
            else
            {
                neighbor_allocation_indices = allocation_indices_buffer[neighbor_index];
                neighbor_slots = get_edge_slots(neighbor_command, neighbor_edge);
                neighbor_slot = neighbor_allocation_indices[neighbor_slots[1]]; // left and right are flipped from the neighbor's perspective
            }
        }

        neighbor_buffer[slot][edge_type] = neighbor_slot;
        if (neighbor_index != INVALID_INDEX && neighbor_slot == INVALID_INDEX)
        {
            pc.dispatch.debug_data.curr_heapid.add(1);
        }

        if (bisector_slots[0] != bisector_slots[1])
        {
            uint slot2 = curr_allocation_indices[bisector_slots[1]];
            uint edge_type2 = edge_types[1];
            uint neighbor_slot2 = neighbor_allocation_indices[neighbor_slots[0]];
            (neighbor_buffer[slot2])[edge_type2] = neighbor_slot2;
            if (neighbor_index != INVALID_INDEX && neighbor_slot2 == INVALID_INDEX)
            {
                pc.dispatch.debug_data.curr_heapid.add(1);
            }
        }
    }

    // Write intra-bisector pointers and heap IDs
    if (curr_command == CENTER_SPLIT)
    {
        for (int i = 0; i < 2; ++i)
        {
            if (curr_allocation_indices[i] == INVALID_INDEX)
            {
                pc.dispatch.debug_data.curr_heapid.add(1);
            }
        }
        heap_id_buffer[curr_allocation_indices[1]] = 2 * curr_heapid + 0;
        heap_id_buffer[curr_allocation_indices[0]] = 2 * curr_heapid + 1;

        link_siblings(curr_allocation_indices[0], curr_allocation_indices[1]);
    }
    else if (curr_command == RIGHT_DOUBLE_SPLIT)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (curr_allocation_indices[i] == INVALID_INDEX)
            {
                pc.dispatch.debug_data.curr_heapid.add(1);
            }
        }
        link_siblings(curr_allocation_indices[0], curr_allocation_indices[2]);
        (neighbor_buffer[curr_allocation_indices[0]])[TWIN] = curr_allocation_indices[1];
        (neighbor_buffer[curr_allocation_indices[1]])[PREV] = curr_allocation_indices[0];

        heap_id_buffer[curr_allocation_indices[1]] = 2 * curr_heapid;
        heap_id_buffer[curr_allocation_indices[2]] = 4 * curr_heapid + 2;
        heap_id_buffer[curr_allocation_indices[0]] = 4 * curr_heapid + 3;
    }
    else if (curr_command == LEFT_DOUBLE_SPLIT)
    {
        for (int i = 0; i < 3; ++i)
        {
            if (curr_allocation_indices[i] == INVALID_INDEX)
            {
                pc.dispatch.debug_data.curr_heapid.add(1);
            }
        }
        link_siblings(curr_allocation_indices[1], curr_allocation_indices[2]);
        (neighbor_buffer[curr_allocation_indices[2]])[TWIN] = curr_allocation_indices[0];
        (neighbor_buffer[curr_allocation_indices[0]])[NEXT] = curr_allocation_indices[2];

        heap_id_buffer[curr_allocation_indices[0]] = 2 * curr_heapid + 1;
        heap_id_buffer[curr_allocation_indices[2]] = 4 * curr_heapid + 0;
        heap_id_buffer[curr_allocation_indices[1]] = 4 * curr_heapid + 1;
    }
    else
    {
        for (int i = 0; i < 4; ++i)
        {
            if (curr_allocation_indices[i] == INVALID_INDEX)
            {
                pc.dispatch.debug_data.curr_heapid.add(1);
            }
        }
        // TRIPLE_SPLIT
        link_siblings(curr_allocation_indices[1], curr_allocation_indices[3]);
        link_siblings(curr_allocation_indices[0], curr_allocation_indices[2]);
        (neighbor_buffer[curr_allocation_indices[3]])[TWIN] = curr_allocation_indices[0];
        (neighbor_buffer[curr_allocation_indices[0]])[TWIN] = curr_allocation_indices[3];

        heap_id_buffer[curr_allocation_indices[3]] = 4 * curr_heapid + 0;
        heap_id_buffer[curr_allocation_indices[1]] = 4 * curr_heapid + 1;
        heap_id_buffer[curr_allocation_indices[2]] = 4 * curr_heapid + 2;
        heap_id_buffer[curr_allocation_indices[0]] = 4 * curr_heapid + 3;
    }
    // for debugging
    heap_id_buffer[curr_id] = 0xFFFF_FFFF;
}
