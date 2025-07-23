use std::u32;

use ash::{Device, vk};

pub fn byteslice<T: Sized>(p: &T) -> &[u8] {
    unsafe {
        ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
    }
}
// finds the index of the device memory type that matches the memory requirements and flags
// this implicitly finds a heap as a memory type contains a heap index as well
pub fn find_memorytype_index(
    memory_req: &vk::MemoryRequirements,
    memory_prop: &vk::PhysicalDeviceMemoryProperties,
    flags: vk::MemoryPropertyFlags,
) -> Option<u32> {
    memory_prop.memory_types[..memory_prop.memory_type_count as usize]
        .iter()
        .enumerate()
        .find(|(index, memory_type)| {
            (1 << index) & memory_req.memory_type_bits != 0
                && memory_type.property_flags & flags == flags
        })
        .map(|(index, _memory_type)| index as u32)
}
pub struct AllocatedBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk::DeviceMemory,
}

impl AllocatedBuffer {
    pub fn new(
        device: &ash::Device,
        size: u64,
        mem_props: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_type: vk::MemoryPropertyFlags,
    ) -> Self {
        unsafe {
            // create buffer handle
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage)
                .sharing_mode(sharing_mode);
            let buffer = device
                .create_buffer(&buffer_create_info, None)
                .expect("Create buffer.");

            // get allocation information
            let req = device.get_buffer_memory_requirements(buffer);
            let mut mem_allocate_flags = vk::MemoryAllocateFlagsInfo::default()
                .flags(vk::MemoryAllocateFlags::DEVICE_ADDRESS);
            let mem_info = vk::MemoryAllocateInfo::default()
                .allocation_size(req.size)
                .memory_type_index(
                    find_memorytype_index(&req, &mem_props, memory_type)
                        .expect("Unable to find suitable memory for vertex buffer."),
                )
                .push_next(&mut mem_allocate_flags);

            // allocate memory
            let buffer_memory = device
                .allocate_memory(&mem_info, None)
                .expect("Failed to allocate vertex buffer memory.");
            device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Could not bind vertex buffer to its memory.");

            Self {
                buffer: buffer,
                allocation: buffer_memory,
            }
        }
    }

    pub fn destroy(self, device: &ash::Device) {
        unsafe {
            device.free_memory(self.allocation, None);
            device.destroy_buffer(self.buffer, None);
        }
    }
}

pub struct CBT {
    depth: u64, // number of edges from the root to a furthest leaf
    interior: Vec<u32>,
    leaves: Vec<u64>,
}

impl CBT {
    pub fn new(depth: u64) -> CBT {
        let bitfield_length = (1 << depth) as u64;
        let num_leaves = bitfield_length / 64;
        let num_internal = 2 * num_leaves - 1;

        let leaves = vec![0 as u64; num_leaves as usize];
        let interior = vec![0 as u32; num_internal as usize];

        return CBT {
            depth: depth,
            interior: interior,
            // occupancy bitfield in little endian bit order
            // ie. between u64: smaller idx -> larger idx
            //     within  u64:         lsb -> msb
            // WARNING: this is the OPPOSITE directionality as expected from bit shifting operations
            leaves: leaves,
        };
    }

    pub fn reduce(&mut self) {
        let interior_offset = self.interior.len() / 2;

        for i in 0..(interior_offset + 1) {
            let num_ones = self.leaves[i].count_ones();
            self.interior[interior_offset + i] = num_ones;
        }

        // index_of_last_level = depth
        // 64 (depth) -> 32 (depth - 1) -> 16 (depth - 2) -> 8 (depth - 3) -> 4 (depth - 4) -> 2 (depth - 5) -> 1 (depth - 6)
        // all these levels have been filled
        let deepest_filled_level = self.depth - 6;

        for level in (0..deepest_filled_level).rev() {
            let level_start = (1 << level) - 1;
            let level_end = (1 << (level + 1)) - 1;
            for i in level_start..level_end {
                self.interior[i] = self.interior[2 * i + 1] + self.interior[2 * i + 2];
            }
        }
    }

    pub fn zero_bit_to_id(
        &self,
        index: u32, // index of the bit in the bitfield
    ) -> u32 {
        let cap = (self.leaves.len() * 64) as u32;
        if index >= cap - self.interior[0] {
            return u32::MAX;
        }
        let mut cap = cap / 2;
        let interior_base_offset = (self.interior.len() / 2) as u32;

        // get the index of the leaf u64 in the leaf array
        let (base_idx, zero_count) = {
            // base case: interior_idx points to the root
            let mut interior_idx = 0 as u32;
            let mut index = index;
            loop {
                let lesser_child_idx = interior_idx * 2 + 1;
                // check if in lesser subtree
                if index < cap - self.interior[lesser_child_idx as usize] {
                    interior_idx = lesser_child_idx; // goto root of lesser subtree
                } else {
                    interior_idx = lesser_child_idx + 1; // goto root of greater subtree
                    index -= cap - self.interior[lesser_child_idx as usize]; // eliminate lesser subtree sum
                }
                cap /= 2;

                // go until interior_idx is at the deepest level of the interior
                if interior_idx >= interior_base_offset {
                    break (interior_idx - interior_base_offset, index);
                }
            }
        };

        let leaf_idx = base_idx * 64;
        // get the index of the bit in the leaf
        let bit_idx = {
            let leaf = self.leaves[base_idx as usize];
            let mut bit_idx = leaf.trailing_ones();
            for _ in 0..zero_count {
                let leaf = leaf >> bit_idx + 1; // add 1 to skip over the next bit
                bit_idx += leaf.trailing_ones() + 1; // add 1 to skip over the previous bit
            }
            bit_idx
        };

        return leaf_idx + bit_idx;

        // return 0;
    }

    pub fn one_bit_to_id(
        &self,
        index: u32, // index of the bit in the bitfield
    ) -> u32 {
        if index >= self.interior[0] {
            return u32::MAX;
        }
        let interior_base_offset = (self.interior.len() / 2) as u32;

        // get the index of the leaf u64 in the leaf array
        let (base_idx, one_count) = {
            // base case: interior_idx points to the root
            let mut interior_idx = 0 as u32;
            let mut index = index;
            loop {
                let lesser_child_idx = interior_idx * 2 + 1;
                // check if in lesser subtree
                if index < self.interior[lesser_child_idx as usize] {
                    interior_idx = lesser_child_idx; // goto root of lesser subtree
                } else {
                    interior_idx = lesser_child_idx + 1; // goto root of greater subtree
                    index -= self.interior[lesser_child_idx as usize]; // eliminate lesser subtree sum
                }

                // BOOKMARK

                // go until interior_idx is at the deepest level of the interior
                if interior_idx >= interior_base_offset {
                    break (interior_idx - interior_base_offset, index);
                }
            }
        };
        // 0010
        // 0004      0006
        // 0001 0003 0002 0004
        // 0010 0111 1001 1111

        // find index 4
        // self.interior.len() / 2 = 3
        // itr lci idx ii
        //   0   1   4  2

        let leaf_idx = base_idx * 64;
        // get the index of the bit in the leaf
        let bit_idx = {
            let leaf = self.leaves[base_idx as usize];
            let mut bit_idx = leaf.trailing_zeros();
            for _ in 0..one_count {
                let leaf = leaf >> bit_idx + 1; // add 1 to skip over the next bit
                bit_idx += leaf.trailing_zeros() + 1; // add 1 to skip over the previous bit
            }
            bit_idx
        };

        return leaf_idx + bit_idx;
    }

    pub fn set_bit(&mut self, index: u32) {
        let leaf_idx = index / 64;
        let bit_idx = index % 64;

        let bit = (1 << bit_idx) as u64;
        self.leaves[leaf_idx as usize] |= bit;
    }

    pub fn unset_bit(&mut self, index: u32) {
        let leaf_idx = index / 64;
        let bit_idx = index % 64;

        let bit = (1 << bit_idx) as u64;
        self.leaves[leaf_idx as usize] &= !bit;
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce1() {
        let mut cbt = CBT::new(8);

        // 0010
        // 0004      0006
        // 0001 0003 0002 0004
        // 0010 0111 1001 1111

        cbt.leaves[0] = 0b0010;
        cbt.leaves[1] = 0b0111;
        cbt.leaves[2] = 0b1001;
        cbt.leaves[3] = 0b1111;

        println!("leaves: {:?}", cbt.leaves);

        cbt.reduce();
        assert_eq!(cbt.interior, vec![10, 4, 6, 1, 3, 2, 4]);

        assert_eq!(cbt.one_bit_to_id(0), 64 * 0 + 1);
        assert_eq!(cbt.one_bit_to_id(1), 64 * 1 + 0);
        assert_eq!(cbt.one_bit_to_id(2), 64 * 1 + 1);
        assert_eq!(cbt.one_bit_to_id(3), 64 * 1 + 2);
        assert_eq!(cbt.one_bit_to_id(4), 64 * 2 + 0);
        assert_eq!(cbt.one_bit_to_id(5), 64 * 2 + 3);
        assert_eq!(cbt.one_bit_to_id(6), 64 * 3 + 0);
        assert_eq!(cbt.one_bit_to_id(7), 64 * 3 + 1);
        assert_eq!(cbt.one_bit_to_id(8), 64 * 3 + 2);
        assert_eq!(cbt.one_bit_to_id(9), 64 * 3 + 3);
        assert_eq!(cbt.one_bit_to_id(10), u32::MAX);

        cbt.leaves = cbt
            .leaves
            .iter_mut()
            .map(|x| *x | 0xFFFF_FFFF_FFFF_FFF0)
            .collect();
        cbt.reduce();

        assert_eq!(cbt.zero_bit_to_id(0), 64 * 0 + 0);
        assert_eq!(cbt.zero_bit_to_id(1), 64 * 0 + 2);
        assert_eq!(cbt.zero_bit_to_id(2), 64 * 0 + 3);

        assert_eq!(cbt.zero_bit_to_id(3), 64 * 1 + 3);

        assert_eq!(cbt.zero_bit_to_id(4), 64 * 2 + 1);
        assert_eq!(cbt.zero_bit_to_id(5), 64 * 2 + 2);
        assert_eq!(cbt.zero_bit_to_id(6), u32::MAX);
    }

    #[test]
    fn reduce2() {
        let mut cbt = CBT::new(9);

        // 12
        // 8                   4
        // 4         4         1         3
        // 3    1    2    2    0    1    2    1
        // 1011 0001 0101 1010 0000 0100 0110 0001
        cbt.leaves[0] = 0b1011;
        cbt.leaves[1] = 0b0001;
        cbt.leaves[2] = 0b0101;
        cbt.leaves[3] = 0b1010;
        cbt.leaves[4] = 0b0000;
        cbt.leaves[5] = 0b0100;
        cbt.leaves[6] = 0b0110;
        cbt.leaves[7] = 0b0001;

        println!("leaves: {:?}", cbt.leaves);

        cbt.reduce();
        assert_eq!(
            cbt.interior,
            vec![12, 8, 4, 4, 4, 1, 3, 3, 1, 2, 2, 0, 1, 2, 1]
        );

        // 1011
        assert_eq!(cbt.one_bit_to_id(0), 64 * 0 + 0);
        assert_eq!(cbt.one_bit_to_id(1), 64 * 0 + 1);
        assert_eq!(cbt.one_bit_to_id(2), 64 * 0 + 3);
        // 0001
        assert_eq!(cbt.one_bit_to_id(3), 64 * 1 + 0);
        // 0101
        assert_eq!(cbt.one_bit_to_id(4), 64 * 2 + 0);
        assert_eq!(cbt.one_bit_to_id(5), 64 * 2 + 2);
        // 1010
        assert_eq!(cbt.one_bit_to_id(6), 64 * 3 + 1);
        assert_eq!(cbt.one_bit_to_id(7), 64 * 3 + 3);
        // 0000
        // 0100
        assert_eq!(cbt.one_bit_to_id(8), 64 * 5 + 2);
        // 0110
        assert_eq!(cbt.one_bit_to_id(9), 64 * 6 + 1);
        assert_eq!(cbt.one_bit_to_id(10), 64 * 6 + 2);
        // 0001
        assert_eq!(cbt.one_bit_to_id(11), 64 * 7 + 0);
        // end
        assert_eq!(cbt.one_bit_to_id(12), u32::MAX);

        cbt.leaves = cbt
            .leaves
            .iter_mut()
            .map(|x| *x | 0xFFFF_FFFF_FFFF_FFF0)
            .collect();

        cbt.reduce();
        // 1011
        assert_eq!(cbt.zero_bit_to_id(0), 64 * 0 + 2);
        // 0001
        assert_eq!(cbt.zero_bit_to_id(1), 64 * 1 + 1);
        assert_eq!(cbt.zero_bit_to_id(2), 64 * 1 + 2);
        assert_eq!(cbt.zero_bit_to_id(3), 64 * 1 + 3);
        // 0101
        assert_eq!(cbt.zero_bit_to_id(4), 64 * 2 + 1);
        assert_eq!(cbt.zero_bit_to_id(5), 64 * 2 + 3);
        // 1010
        assert_eq!(cbt.zero_bit_to_id(6), 64 * 3 + 0);
        assert_eq!(cbt.zero_bit_to_id(7), 64 * 3 + 2);
        // 0000
        assert_eq!(cbt.zero_bit_to_id(8), 64 * 4 + 0);
        assert_eq!(cbt.zero_bit_to_id(9), 64 * 4 + 1);
        assert_eq!(cbt.zero_bit_to_id(10), 64 * 4 + 2);
        assert_eq!(cbt.zero_bit_to_id(11), 64 * 4 + 3);
        // 0100
        assert_eq!(cbt.zero_bit_to_id(12), 64 * 5 + 0);
        assert_eq!(cbt.zero_bit_to_id(13), 64 * 5 + 1);
        assert_eq!(cbt.zero_bit_to_id(14), 64 * 5 + 3);
        // 0110
        assert_eq!(cbt.zero_bit_to_id(15), 64 * 6 + 0);
        assert_eq!(cbt.zero_bit_to_id(16), 64 * 6 + 3);
        // 0001
        assert_eq!(cbt.zero_bit_to_id(17), 64 * 7 + 1);
        assert_eq!(cbt.zero_bit_to_id(18), 64 * 7 + 2);
        assert_eq!(cbt.zero_bit_to_id(19), 64 * 7 + 3);
        // end
        assert_eq!(cbt.zero_bit_to_id(20), u32::MAX);
    }
}
