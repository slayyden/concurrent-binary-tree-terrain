#include "cbt.slang"
void merge(
    uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dispatch.merging_bisector_count.load())
    {
        return;
    }
    let curr_id = pc.scene.merging_bisector_buffer[tid.x];
    let neighbor_buffer = pc.scene.neighbors_buffer;
    let heap_id_buffer = pc.scene.heapid_buffer;
    uint curr_heapid = heap_id_buffer[curr_id]; // even guaranteed
    uint3 curr_neighbors = neighbor_buffer[curr_id];

    uint next_id = curr_neighbors[NEXT];
    uint3 next_neighbors = neighbor_buffer[next_id];

    uint3 new_neighbors = uint3(
        curr_neighbors[TWIN], // NEXT
        next_neighbors[TWIN], // PREV
        next_neighbors[NEXT]  // TWIN
    );

    neighbor_buffer[curr_id] = new_neighbors;
    heap_id_buffer[curr_id] = curr_heapid >> 1;
    heap_id_buffer[next_id] = 0;

    unset_bit(next_id);
}
