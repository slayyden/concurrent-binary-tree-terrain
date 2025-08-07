#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

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
    threadgroup spvUnsafeArray<uint, 2048> buf1;
    threadgroup spvUnsafeArray<uint, 4096> buf0;
    uint i = 0u;
    for (;;)
    {
        if (!(i < 4096u))
        {
            break;
        }
        uint _365 = gl_GlobalInvocationID.x + i;
        uint _367 = atomic_load_explicit((device atomic_uint*)&pc.scene->cbt_leaves[_365], memory_order_relaxed);
        uint _368 = popcount(_367);
        buf0[_365] = _368;
        pc.scene->cbt_interior[4095u + _365] = _368;
        i += 1024u;
        continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    i = 0u;
    for (;;)
    {
        if (!(i < 2048u))
        {
            break;
        }
        uint _343 = gl_GlobalInvocationID.x + i;
        uint _344 = 2u * _343;
        uint _350 = buf0[_344] + buf0[_344 + 1u];
        buf1[_343] = _350;
        pc.scene->cbt_interior[2047u + _343] = _350;
        i += 1024u;
        continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _111 = 2u * gl_GlobalInvocationID.x;
    uint _119 = _111 + 1u;
    uint _123 = buf1[_111] + buf1[_119];
    buf0[gl_GlobalInvocationID.x] = _123;
    pc.scene->cbt_interior[1023u + gl_GlobalInvocationID.x] = _123;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 512u)
    {
        uint _142 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _142;
        pc.scene->cbt_interior[511u + gl_GlobalInvocationID.x] = _142;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 256u)
    {
        uint _156 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _156;
        pc.scene->cbt_interior[255u + gl_GlobalInvocationID.x] = _156;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 128u)
    {
        uint _171 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _171;
        pc.scene->cbt_interior[127u + gl_GlobalInvocationID.x] = _171;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 64u)
    {
        uint _185 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _185;
        pc.scene->cbt_interior[63u + gl_GlobalInvocationID.x] = _185;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 32u)
    {
        uint _200 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _200;
        pc.scene->cbt_interior[31u + gl_GlobalInvocationID.x] = _200;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 16u)
    {
        uint _214 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _214;
        pc.scene->cbt_interior[15u + gl_GlobalInvocationID.x] = _214;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 8u)
    {
        uint _229 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _229;
        pc.scene->cbt_interior[7u + gl_GlobalInvocationID.x] = _229;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 4u)
    {
        uint _243 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _243;
        pc.scene->cbt_interior[3u + gl_GlobalInvocationID.x] = _243;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 2u)
    {
        uint _257 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _257;
        pc.scene->cbt_interior[1u + gl_GlobalInvocationID.x] = _257;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint final;
    if (gl_GlobalInvocationID.x < 1u)
    {
        uint _272 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _272;
        pc.scene->cbt_interior[gl_GlobalInvocationID.x] = _272;
        final = _272;
    }
    else
    {
        final = 120394819u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _11;
    if (gl_GlobalInvocationID.x == 0u)
    {
        _11 = gl_GlobalInvocationID.y == 0u;
    }
    else
    {
        _11 = false;
    }
    if (_11)
    {
        _11 = gl_GlobalInvocationID.z == 0u;
    }
    else
    {
        _11 = false;
    }
    if (_11)
    {
        atomic_store_explicit((device atomic_int*)&pc.dispatch->remaining_memory_count, 131072 - int(final), memory_order_relaxed);
        pc.dispatch->num_allocated_blocks = final;
        pc.dispatch->draw_indirect_command.vertex_count = 3u * final;
        pc.dispatch->dispatch_vertex_compute_command.x = (final + 63u) / 64u;
    }
}

#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wmissing-prototypes"
#pragma clang diagnostic ignored "-Wmissing-braces"

#include <metal_stdlib>
#include <simd/simd.h>
#include <metal_atomic>

using namespace metal;

template<typename T, size_t Num>
struct spvUnsafeArray
{
    T elements[Num ? Num : 1];
    
    thread T& operator [] (size_t pos) thread
    {
        return elements[pos];
    }
    constexpr const thread T& operator [] (size_t pos) const thread
    {
        return elements[pos];
    }
    
    device T& operator [] (size_t pos) device
    {
        return elements[pos];
    }
    constexpr const device T& operator [] (size_t pos) const device
    {
        return elements[pos];
    }
    
    constexpr const constant T& operator [] (size_t pos) const constant
    {
        return elements[pos];
    }
    
    threadgroup T& operator [] (size_t pos) threadgroup
    {
        return elements[pos];
    }
    constexpr const threadgroup T& operator [] (size_t pos) const threadgroup
    {
        return elements[pos];
    }
};

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
    threadgroup spvUnsafeArray<uint, 2048> buf1;
    threadgroup spvUnsafeArray<uint, 4096> buf0;
    uint i = 0u;
    for (;;)
    {
        if (!(i < 4096u))
        {
            break;
        }
        uint _365 = gl_GlobalInvocationID.x + i;
        uint _367 = atomic_load_explicit((device atomic_uint*)&pc.scene->cbt_leaves[_365], memory_order_relaxed);
        uint _368 = popcount(_367);
        buf0[_365] = _368;
        pc.scene->cbt_interior[4095u + _365] = _368;
        i += 1024u;
        continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    i = 0u;
    for (;;)
    {
        if (!(i < 2048u))
        {
            break;
        }
        uint _343 = gl_GlobalInvocationID.x + i;
        uint _344 = 2u * _343;
        uint _350 = buf0[_344] + buf0[_344 + 1u];
        buf1[_343] = _350;
        pc.scene->cbt_interior[2047u + _343] = _350;
        i += 1024u;
        continue;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint _111 = 2u * gl_GlobalInvocationID.x;
    uint _119 = _111 + 1u;
    uint _123 = buf1[_111] + buf1[_119];
    buf0[gl_GlobalInvocationID.x] = _123;
    pc.scene->cbt_interior[1023u + gl_GlobalInvocationID.x] = _123;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 512u)
    {
        uint _142 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _142;
        pc.scene->cbt_interior[511u + gl_GlobalInvocationID.x] = _142;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 256u)
    {
        uint _156 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _156;
        pc.scene->cbt_interior[255u + gl_GlobalInvocationID.x] = _156;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 128u)
    {
        uint _171 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _171;
        pc.scene->cbt_interior[127u + gl_GlobalInvocationID.x] = _171;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 64u)
    {
        uint _185 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _185;
        pc.scene->cbt_interior[63u + gl_GlobalInvocationID.x] = _185;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 32u)
    {
        uint _200 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _200;
        pc.scene->cbt_interior[31u + gl_GlobalInvocationID.x] = _200;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 16u)
    {
        uint _214 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _214;
        pc.scene->cbt_interior[15u + gl_GlobalInvocationID.x] = _214;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 8u)
    {
        uint _229 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _229;
        pc.scene->cbt_interior[7u + gl_GlobalInvocationID.x] = _229;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 4u)
    {
        uint _243 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _243;
        pc.scene->cbt_interior[3u + gl_GlobalInvocationID.x] = _243;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if (gl_GlobalInvocationID.x < 2u)
    {
        uint _257 = buf0[_111] + buf0[_119];
        buf1[gl_GlobalInvocationID.x] = _257;
        pc.scene->cbt_interior[1u + gl_GlobalInvocationID.x] = _257;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    uint final;
    if (gl_GlobalInvocationID.x < 1u)
    {
        uint _272 = buf1[_111] + buf1[_119];
        buf0[gl_GlobalInvocationID.x] = _272;
        pc.scene->cbt_interior[gl_GlobalInvocationID.x] = _272;
        final = _272;
    }
    else
    {
        final = 120394819u;
    }
    threadgroup_barrier(mem_flags::mem_threadgroup);
    bool _11;
    if (gl_GlobalInvocationID.x == 0u)
    {
        _11 = gl_GlobalInvocationID.y == 0u;
    }
    else
    {
        _11 = false;
    }
    if (_11)
    {
        _11 = gl_GlobalInvocationID.z == 0u;
    }
    else
    {
        _11 = false;
    }
    if (_11)
    {
        atomic_store_explicit((device atomic_int*)&pc.dispatch->remaining_memory_count, 131072 - int(final), memory_order_relaxed);
        pc.dispatch->num_allocated_blocks = final;
        pc.dispatch->draw_indirect_command.vertex_count = 3u * final;
        pc.dispatch->dispatch_vertex_compute_command.x = (final + 63u) / 64u;
    }
}

