
#include "common.slang"

groupshared uint buf0[4096];
groupshared uint buf1[2048];

[shader("compute")]
[numthreads(1024, 1, 1)]
void reduce(uint3 threadId: SV_DispatchThreadID)
{
    let tid = threadId.x;
    let interior_offset = num_internal / 2;
    let interior = pc.scene_next.cbt_interior;
    let leaves = pc.scene_next.cbt_leaves;

    // bottom level: [4095, 8191)
    for (uint i = 0; i < 4096; i += 1024)
    {
        let leaf_index = tid + i;
        let num_allocated = countbits(leaves[leaf_index].load());
        buf0[leaf_index] = num_allocated;
        interior[interior_offset + leaf_index] = num_allocated;
    }

    GroupMemoryBarrier();

    // level n-1 [2047, 4095)
    {
        let interior_offset = num_internal / (1 << 2);
        for (uint i = 0; i < 2048; i += 1024)
        {
            let index_in_level = tid + i;
            let sum = buf0[2 * index_in_level] + buf0[2 * index_in_level + 1];
            buf1[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();
    // level n-2 [1023, 2047)
    {
        let interior_offset = num_internal / (1 << 3);
        let index_in_level = tid;
        let sum = buf1[2 * index_in_level] + buf1[2 * index_in_level + 1];
        buf0[index_in_level] = sum;
        interior[interior_offset + index_in_level] = sum;
    }
    GroupMemoryBarrier();

    // level n-3 [511, 1023)
    {
        let index_in_level = tid;
        if (index_in_level < 512)
        {
            let interior_offset = num_internal / (1 << 4);
            let sum = buf0[2 * index_in_level] + buf0[2 * index_in_level + 1];
            buf1[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();

    // level n-4 [255, 511)
    {
        let index_in_level = tid;
        if (index_in_level < 256)
        {
            let interior_offset = num_internal / (1 << 5);
            let sum = buf1[2 * index_in_level] + buf1[2 * index_in_level + 1];
            buf0[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();

    // level n-5 [127, 255)
    {
        let index_in_level = tid;
        if (index_in_level < 128)
        {
            let interior_offset = num_internal / (1 << 6);
            let sum = buf0[2 * index_in_level] + buf0[2 * index_in_level + 1];
            buf1[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();

    // level n-6 [63, 127)
    {
        let index_in_level = tid;
        if (index_in_level < 64)
        {
            let interior_offset = num_internal / (1 << 7);
            let sum = buf1[2 * index_in_level] + buf1[2 * index_in_level + 1];
            buf0[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();

    // level n-7 [31, 63)
    {
        let index_in_level = tid;
        if (index_in_level < 32)
        {
            let interior_offset = num_internal / (1 << 8);
            let sum = buf0[2 * index_in_level] + buf0[2 * index_in_level + 1];
            buf1[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();

    // level n-8 [15, 31)
    {
        let index_in_level = tid;
        if (index_in_level < 16)
        {
            let interior_offset = num_internal / (1 << 9);
            let sum = buf1[2 * index_in_level] + buf1[2 * index_in_level + 1];
            buf0[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();

    // level n-9 [7, 15)
    {
        let index_in_level = tid;
        if (index_in_level < 8)
        {
            let interior_offset = num_internal / (1 << 10);
            let sum = buf0[2 * index_in_level] + buf0[2 * index_in_level + 1];
            buf1[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();

    // level n-10 [3, 7)
    {
        let index_in_level = tid;
        if (index_in_level < 4)
        {
            let interior_offset = num_internal / (1 << 11);
            let sum = buf1[2 * index_in_level] + buf1[2 * index_in_level + 1];
            buf0[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();

    // level n-11 [1, 3)
    {
        let index_in_level = tid;
        if (index_in_level < 2)
        {
            let interior_offset = num_internal / (1 << 12);
            let sum = buf0[2 * index_in_level] + buf0[2 * index_in_level + 1];
            buf1[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();

    // level n-12 [0, 1)
    {
        let index_in_level = tid;
        if (index_in_level < 1)
        {
            let interior_offset = num_internal / (1 << 13);
            let sum = buf1[2 * index_in_level] + buf1[2 * index_in_level + 1];
            buf0[index_in_level] = sum;
            interior[interior_offset + index_in_level] = sum;
        }
    }
    GroupMemoryBarrier();
    if (threadId.x == 0 && threadId.y == 0 && threadId.z == 0)
    {
        let num_bisectors : uint = buf0[0];
        pc.dispatch_next.remaining_memory_count.store((int)(1 << 17) - (int)num_bisectors);
        pc.dispatch_next.num_allocated_blocks = num_bisectors;
        pc.dispatch_next.draw_indirect_command.vertex_count = 3 * num_bisectors;
        pc.dispatch_next.dispatch_vertex_compute_command.x = div_ceil(num_bisectors, WORKGROUP_SIZE);
    }
    return;
}
