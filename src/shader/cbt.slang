#pragma once
#include "common.slang"

uint one_bit_to_id(uint index)
{
    let interior = pc.scene.cbt_interior;
    let leaves = pc.scene.cbt_leaves;
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

        // BOOKMARK

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
    let leaf = leaves[base_idx];
    var bit_idx = firstbitlow(leaf);
    for (int i = 0; i < one_count; ++i)
    {
        let leaf = leaf >> bit_idx + 1;   // add 1 to skip over the next bit
        bit_idx += firstbitlow(leaf) + 1; // add 1 to skip over the previous bit
    }
    return leaf_idx + bit_idx;
}
uint zero_bit_to_id(
    uint index, // index of the bit in the bitfield
)
{
    let interior = pc.scene.cbt_interior;
    let leaves = pc.scene.cbt_leaves;
    var cap : uint = 1 << 17;
    if (index >= cap - interior[0])
    {
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
    let leaf = leaves[base_idx];
    var bit_idx = firstbitlow(~leaf);
    for (int i = 0; i < zero_count; ++i)
    {
        let leaf = leaf >> bit_idx + 1;    // add 1 to skip over the next bit
        bit_idx += firstbitlow(~leaf) + 1; // add 1 to skip over the previous bit
    }

    return leaf_idx + bit_idx;
};

void set_bit(uint index)
{
    let leaves = pc.scene.cbt_leaves;
    const uint BITFIELD_INT_SIZE = 32; // or whatever your constant is
    uint leaf_idx = index / BITFIELD_INT_SIZE;
    uint bit_idx = index % BITFIELD_INT_SIZE;

    uint bit = 1u << bit_idx;
    leaves[leaf_idx].or(bit);
}

void unset_bit(uint index)
{
    let leaves = pc.scene.cbt_leaves;
    const uint BITFIELD_INT_SIZE = 32; // same as above
    uint leaf_idx = index / BITFIELD_INT_SIZE;
    uint bit_idx = index % BITFIELD_INT_SIZE;

    uint bit = 1u << bit_idx;
    leaves[leaf_idx].and(~bit);
}
