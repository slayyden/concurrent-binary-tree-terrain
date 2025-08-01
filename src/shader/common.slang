struct SceneData
{
    // written once at initialization
    float3 *root_bisector_vertices;
    // our concurrent binary tree
    // cbt: CBT,
    uint *cbt_interior;
    uint *cbt_leaves;

    // classification stage
    uint *classification_buffer;
    uint *bisector_state_buffer;

    // prepare split
    uint *bisector_split_command_buffer;
    uint3 *neighbors_buffer;
    uint *splitting_buffer;
    uint *heapid_buffer;

    // allocate
    uint4 *allocation_indices_buffer;

    // split
    uint *want_split_buffer;

    // prepare merge
    uint *want_merge_buffer;

    // merge
    uint *merging_bisector_buffer;

    // draw
    float3 *vertex_buffer;

    // integers
    uint num_memory_blocks;
    uint base_depth;
    uint cbt_depth;
}

struct DispatchData
{
    Atomic<uint> remaining_memory_count;
    uint2 pad0;
    Atomic<uint> allocation_counter;
    uint2 pad1;
    uint3 want_split_buffer_count;
    uint3 splitting_buffer_count;
    uint3 want_merge_buffer_count;
    Atomic<uint> merging_bisector_count;
    uint2 pad2;
}

struct PushConstant
{
    float4x4 view_project;
    float3 *positions;
    uint32_t *curr_id;
    SceneData *scene;
    DispatchData *dispatch;
}

static const uint BITFIELD_INT_SIZE = 32;
static const uint INVALID_INDEX = 0xFFFF_FFFF;
static const uint num_leaves = 4096;
static const uint num_internal = 2 * num_leaves - 1;

static const uint UNCHANGED_ELEMENT = 0;
static const uint SIMPLIFY = 1;
static const uint SPLIT = 2;

static const uint NO_SPLIT = 0;
static const uint CENTER_SPLIT = 1;
static const uint RIGHT_SPLIT = 1 << 1;
static const uint LEFT_SPLIT = 1 << 2;
static const uint RIGHT_DOUBLE_SPLIT = CENTER_SPLIT | RIGHT_SPLIT;
static const uint LEFT_DOUBLE_SPLIT = CENTER_SPLIT | LEFT_SPLIT;
static const uint TRIPLE_SPLIT = CENTER_SPLIT | RIGHT_SPLIT | LEFT_SPLIT;

static const uint NEXT = 0;
static const uint PREV = 1;
static const uint TWIN = 2;

[[vk::push_constant]]
ConstantBuffer<PushConstant> pc;

uint heap_id_depth(uint heapid)
{
    return firstbithigh(heapid);
}

// Given current ID and neighbor's neighbors, find edge type
uint find_edge_type(uint curr_id, uint3 neighbor_neighbors)
{
    if (neighbor_neighbors[NEXT] == curr_id)
        return NEXT;
    else if (neighbor_neighbors[PREV] == curr_id)
        return PREV;
    else
        return TWIN;
}
