#include "cbt.slang"
[shader("compute")]
[numthreads(WORKGROUP_SIZE, 1, 1)]
void merge(
    uint3 tid: SV_DispatchThreadID)
{
    if (tid.x >= pc.dispatch_next.merging_bisector_count.load()) return;

    let curr_id = pc.scene_next.merging_bisector_buffer[tid.x];
    let neighbor_buffer = pc.scene_next.neighbors_buffer;
    let heap_id_buffer = pc.scene_next.heapid_buffer;
    uint3 curr_neighbors = neighbor_buffer[curr_id];

    uint next_id = curr_neighbors.x;
    uint3 next_neighbors = neighbor_buffer[next_id];

    uint3 new_neighbors = uint3(
        curr_neighbors[TWIN], // NEXT
        next_neighbors[TWIN], // PREV
        next_neighbors[NEXT], // TWIN
    );

    neighbor_buffer[curr_id] = new_neighbors;
    heap_id_buffer[curr_id] /= 2;
}
