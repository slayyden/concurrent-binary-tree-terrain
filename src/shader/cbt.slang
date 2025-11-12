#pragma once
#include "common.slang"

uint one_bit_to_id(uint index)
{
    let uniform_buffer = pc.uniform_buffer;
    let scene = uniform_buffer.scene_next;
    let interior = scene.cbt_interior;
    let leaves = scene.cbt_leaves;

    if (index >= interior[0])
    {
        return INVALID_INDEX;
    }
    let interior_base_offset = num_internal / 2;

    // get the index of the leaf u64 in the leaf array
    // base case: interior_idx points to the root
    var interior_idx = 0;
    var index = index;
    while (true)
    {
        let lesser_child_idx = interior_idx * 2 + 1;
        // check if in lesser subtree
        if (index < interior[lesser_child_idx])
        {
            interior_idx = lesser_child_idx; // goto root of lesser subtree
        }
        else
        {
            interior_idx = lesser_child_idx + 1; // goto root of greater subtree
            index -= interior[lesser_child_idx]; // eliminate lesser subtree sum
        }

        // go until interior_idx is at the deepest level of the interior
        if (interior_idx >= interior_base_offset)
        {
            break;
        }
    }
    let base_idx = interior_idx - interior_base_offset;
    let one_count = index;
    // 0010
    // 0004      0006
    // 0001 0003 0002 0004
    // 0010 0111 1001 1111

    // find index 4
    // self.interior.len() / 2 = 3
    // itr lci idx ii
    //   0   1   4  2

    let leaf_idx = base_idx * BITFIELD_INT_SIZE;
    // get the index of the bit in the leaf
    let leaf = leaves[base_idx].load();
    var bit_idx = firstbitlow(leaf);
    for (int i = 0; i < one_count; ++i)
    {
        if (bit_idx >= 31)
            break;
        let leaf_shifted = leaf >> (bit_idx + 1); // add 1 to skip over the next bit
        bit_idx += firstbitlow(leaf_shifted) + 1; // add 1 to skip over the previous bit
    }
    return leaf_idx + bit_idx;
}
uint zero_bit_to_id(
    uint index, // index of the bit in the bitfield
)
{
    let uniform_buffer = pc.uniform_buffer;
    let scene = uniform_buffer.scene_next;
    let interior = scene.cbt_interior;
    let leaves = scene.cbt_leaves;
    var cap : uint = 1 << 17;
    if (index >= cap - interior[0])
    {
        uniform_buffer.dispatch_next.debug_data.curr_heapid = INVALID_INDEX;
        return INVALID_INDEX;
    }
    let interior_base_offset = num_internal / 2;
    cap = cap / 2;

    // get the index of the leaf u64 in the leaf array
    // let(base_idx, zero_count) = {
    // base case: interior_idx points to the root
    var interior_idx = 0;
    var index = index;
    while (true)
    {
        let lesser_child_idx = interior_idx * 2 + 1;
        // check if in lesser subtree
        if (index < cap - interior[lesser_child_idx])
        {
            interior_idx = lesser_child_idx; // goto root of lesser subtree
        }
        else
        {
            interior_idx = lesser_child_idx + 1;       // goto root of greater subtree
            index -= cap - interior[lesser_child_idx]; // eliminate lesser subtree sum
        }
        cap /= 2;

        // go until interior_idx is at the deepest level of the interior
        if (interior_idx >= interior_base_offset)
        {
            break;
        }
    }
    let base_idx = interior_idx - interior_base_offset;
    let zero_count = index;

    let leaf_idx = base_idx * BITFIELD_INT_SIZE;
    // get the index of the bit in the leaf
    let leaf = leaves[base_idx].load();
    var bit_idx = firstbitlow(~leaf);
    for (int i = 0; i < zero_count; ++i)
    {
        if (bit_idx >= 31)
            break;
        let leaf_shifted = leaf >> (bit_idx + 1);  // add 1 to skip over the next bit
        bit_idx += firstbitlow(~leaf_shifted) + 1; // add 1 to skip over the previous bit
    }

    return leaf_idx + bit_idx;
};

void set_bit(uint index)
{
    // let leaves = pc.scene_next.cbt_leaves;
    let leaves = pc.uniform_buffer.scene_next.cbt_leaves;
    uint leaf_idx = index / BITFIELD_INT_SIZE;
    uint bit_idx = index % BITFIELD_INT_SIZE;

    uint bit = 1u << bit_idx;
    leaves[leaf_idx].or(bit);
}

void unset_bit(uint index)
{
    // let leaves = pc.scene_next.cbt_leaves;
    let leaves = pc.uniform_buffer.scene_next.cbt_leaves;
    uint leaf_idx = index / BITFIELD_INT_SIZE;
    uint bit_idx = index % BITFIELD_INT_SIZE;

    uint bit = 1u << bit_idx;
    leaves[leaf_idx].and(~bit);
}

float tri_area(float3 verts[3], float VdotN) {
    var v0_projected = mul(float4(verts[0], 1), pc.uniform_buffer.view_project);
    var v1_projected = mul(float4(verts[1], 1), pc.uniform_buffer.view_project);
    var v2_projected = mul(float4(verts[2], 1), pc.uniform_buffer.view_project);
    v0_projected.xy /= v0_projected.w;
    v1_projected.xy /= v1_projected.w;
    v2_projected.xy /= v2_projected.w;
    v0_projected.xy = map_range(v0_projected.xy, float2(-1, -1), float2(1, 1), float2(0, 0), float2(SCREEN_WIDTH, SCREEN_HEIGHT));
    v1_projected.xy = map_range(v1_projected.xy, float2(-1, -1), float2(1, 1), float2(0, 0), float2(SCREEN_WIDTH, SCREEN_HEIGHT));
    v2_projected.xy = map_range(v2_projected.xy, float2(-1, -1), float2(1, 1), float2(0, 0), float2(SCREEN_WIDTH, SCREEN_HEIGHT));

    let edge0 = v1_projected.xy - v0_projected.xy;
    let edge1 = v2_projected.xy - v0_projected.xy;
    let det = edge0.x * edge1.y - edge0.y * edge1.x;
    var area = (0.5 * det) / (SCREEN_HEIGHT * SCREEN_WIDTH); // area as percentage of screen

    // We over estimate the area at grazing angles
    let areaOverestimation = lerp(2.0, 1.0, pow(abs(VdotN), 0.2));
    area *= areaOverestimation;
    return abs(area);
}
