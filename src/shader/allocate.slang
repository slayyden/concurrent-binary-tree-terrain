#include "cbt.slang"

[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void allocate(
    uint3 threadId: SV_DispatchThreadID) 
{
    let uniform_buffer = pc.uniform_buffer;
    let scene = uniform_buffer.scene_next;
    let dispatch = uniform_buffer.dispatch_next;
    let tid = threadId.x;

    if (tid >= dispatch.splitting_buffer_count.load())
    {
        return;
    }
    let curr_id = scene.splitting_buffer[tid];
    let command = scene.bisector_split_command_buffer[curr_id].load();

    let num_allocations = countbits(command) + 1;

    let first_bit_index = dispatch.allocation_counter.add(num_allocations);

    var allocation_indices = uint4(INVALID_INDEX);

    for (int i = 0; i < num_allocations; ++i)
    {
        let bit_index = i + first_bit_index;
        let child_id = zero_bit_to_id(bit_index);
        allocation_indices[i] = child_id;
    }
    scene.allocation_indices_buffer[curr_id] = allocation_indices;
    scene.bisector_state_buffer[curr_id] = TriState::BISECT_ELEMENT;
}

