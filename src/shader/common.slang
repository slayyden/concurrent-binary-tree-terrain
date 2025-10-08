struct SceneData
{
    // written once at initialization
    float3 *root_bisector_vertices;
    // our concurrent binary tree
    // cbt: CBT,
    uint *cbt_interior;
    Atomic<uint> *cbt_leaves;

    // classification stage
    TriState *bisector_state_buffer;

    // prepare split
    Atomic<uint> *bisector_split_command_buffer;
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
    uint *curr_id_buffer;

    // swapback
    SwapbackData *swapback_buffer;

    // integers
    uint num_memory_blocks;
    uint base_depth;
    uint cbt_depth;
}
// 116 bytes
struct DrawIndirectCommand
{
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
}

struct DispatchIndirectCommand
{
    uint x;
    uint y;
    uint z;
}

struct DebugData
{
    Atomic<uint> curr_heapid;
    uint curr_id;
    uint neighbor_heapid;
    uint neighbor_id;
    float3 curr_verts[3];
    float3 neighbor_verts[3];
    uint num_equal;
    int2 equal_indices[2];
}

struct DispatchData
{
    DrawIndirectCommand draw_indirect_command;

    DispatchIndirectCommand dispatch_split_command;
    DispatchIndirectCommand dispatch_allocate_command;
    DispatchIndirectCommand dispatch_prepare_merge_command;
    // written at reduce
    DispatchIndirectCommand dispatch_vertex_compute_command;

    // written at
    Atomic<int> remaining_memory_count;
    Atomic<uint> allocation_counter;
    Atomic<uint> want_split_buffer_count;
    Atomic<uint> splitting_buffer_count;
    Atomic<uint> want_merge_buffer_count;
    Atomic<uint> merging_bisector_count;
    // written at reduce
    uint num_allocated_blocks;
    DebugData debug_data;
}

enum RenderingMode : uint32_t {
    DEFAULT,
    ROLLBACK_DEFAULT,
    DBG_WANT_SPLIT,
}

struct PushConstant
{
    float4x4 view_project; 
    SceneData *scene_prev;
    SceneData *scene_next;
    DispatchData *dispatch_prev;
    DispatchData *dispatch_next;
    float3 cam_pos;
    uint8_t pad0;

    float3 lookdir;
<<<<<<< HEAD
    uint8_t pad1;

    uint swapback;
}

struct SwapbackData
{
    float3 vertex_position[3];
    uint split_command;
    uint8_t want_split; // 0 for false, 1 for true
=======
    RenderingMode rendering_mode;
>>>>>>> aba2c23 (reference data from multiple scenes)
}
[[vk::push_constant]]
ConstantBuffer<PushConstant, ScalarDataLayout> pc;

static const uint WORKGROUP_SIZE = 64;

static const uint BITFIELD_INT_SIZE = 32;
static const uint INVALID_INDEX = 0xFFFF_FFFF;
static const uint num_leaves = 4096;
static const uint num_internal = 2 * num_leaves - 1;

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

static const uint SCREEN_WIDTH = 1920;
static const uint SCREEN_HEIGHT = 1080;

enum TriState : int
{
    BACK_FACE_CULLED = -3,
    FRUSTUM_CULLED = -2,
    TOO_SMALL = -1,
    UNCHANGED_ELEMENT = 0,
    BISECT_ELEMENT = 1,
    SIMPLIFY_ELEMENT = 2,
}


uint heap_id_depth(uint heapid)
{
    /*
    uint depth = 0;
    while (heapid > 0u)
    {
        ++depth;
        heapid >>= 1u;
    }
    return depth - 1;*/
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
    {
        if (neighbor_neighbors[TWIN] != curr_id)
        {
            pc.dispatch_next.debug_data.curr_heapid.compareExchange(0, curr_id);
        }
        return TWIN;
    }
}

uint div_ceil(uint x, uint divisor)
{
    return (x + divisor - 1) / divisor;
}

// linearly remap val from [low0, high0] to [low1, high1].
inline T map_range<T>(T val, T low0, T high0, T low1, T high1)
    where T : IArithmetic
{
    // t in [0, 1] if val in [low0, high0]
    let t = (val - low0) / (high0 - low0);
    return low1 + t * (high1 - low1);
}

inline bool swapback(PushConstant pc)
{
    return (pc.swapback != 0);
}

inline float3 get_vertex_buffer(PushConstant pc, uint index)
{
    if (swapback(pc))
    {
        return pc.scene.swapback_buffer[index / 3].vertex_position[index % 3];
    }
    else
    {
        return pc.scene.vertex_buffer[index];
    }
}
