#pragma clang diagnostic ignored "-Wunused-variable"

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

using namespace metal;

struct SceneData_natural;
struct DispatchData_natural;
struct DrawIndirectCommand_natural;
struct DispatchIndirectCommand_natural;

struct _MatrixStorage_float4x4natural
{
    float4 data[4];
};

struct PushConstant_natural
{
    _MatrixStorage_float4x4natural view_project;
    device float3* positions;
    device uint* curr_id;
    device SceneData_natural* scene;
    device DispatchData_natural* dispatch;
    packed_float3 cam_pos;
    packed_float3 lookdir;
};

struct SceneData_natural
{
    device float3* root_bisector_vertices;
    device uint* cbt_interior;
    device uint* cbt_leaves;
    device int* bisector_state_buffer;
    device uint* bisector_split_command_buffer;
    device uint3* neighbors_buffer;
    device uint* splitting_buffer;
    device uint* heapid_buffer;
    device uint4* allocation_indices_buffer;
    device uint* want_split_buffer;
    device uint* want_merge_buffer;
    device uint* merging_bisector_buffer;
    device float3* vertex_buffer;
    device uint* currid_buffer;
    uint num_memory_blocks;
    uint base_depth;
    uint cbt_depth;
    uint debug_counter;
};

struct DrawIndirectCommand_natural
{
    uint vertex_count;
    uint instance_count;
    uint first_vertex;
    uint first_instance;
};

struct DispatchIndirectCommand_natural
{
    uint x;
    uint y;
    uint z;
};

struct DispatchData_natural
{
    DrawIndirectCommand_natural draw_indirect_command;
    DispatchIndirectCommand_natural dispatch_split_command;
    DispatchIndirectCommand_natural dispatch_allocate_command;
    DispatchIndirectCommand_natural dispatch_prepare_merge_command;
    DispatchIndirectCommand_natural dispatch_vertex_compute_command;
    int remaining_memory_count;
    uint allocation_counter;
    uint want_split_buffer_count;
    uint splitting_buffer_count;
    uint want_merge_buffer_count;
    uint merging_bisector_count;
    uint num_allocated_blocks;
    uint debug_counter;
};

kernel void main0(constant PushConstant_natural& pc [[buffer(0)]], uint3 gl_GlobalInvocationID [[thread_position_in_grid]])
{
    bool _7;
    if (gl_GlobalInvocationID.x == 0u)
    {
        _7 = gl_GlobalInvocationID.y == 0u;
    }
    else
    {
        _7 = false;
    }
    if (_7)
    {
        _7 = gl_GlobalInvocationID.z == 0u;
    }
    else
    {
        _7 = false;
    }
    if (_7)
    {
        atomic_store_explicit((device atomic_int*)&pc.dispatch->remaining_memory_count, 131072 - int(pc.scene->cbt_interior[0]), memory_order_relaxed);
        pc.dispatch->num_allocated_blocks = pc.scene->cbt_interior[0];
        pc.dispatch->draw_indirect_command.vertex_count = 3u * pc.scene->cbt_interior[0];
        pc.dispatch->dispatch_vertex_compute_command.x = (pc.scene->cbt_interior[0] + 63u) / 64u;
        pc.dispatch->debug_counter++;
        pc.scene->debug_counter++;
    }
}

