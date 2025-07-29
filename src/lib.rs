use std::{
    cmp::max,
    ops::{Add, AddAssign, Div},
    sync::atomic::{AtomicU32, AtomicUsize, Ordering},
    u32,
};

use glam::*;

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

const NO_SPLIT: u32 = 0;
const CENTER_SPLIT: u32 = 1;
const RIGHT_SPLIT: u32 = 1 << 1;
const LEFT_SPLIT: u32 = 1 << 2;
const RIGHT_DOUBLE_SPLIT: u32 = CENTER_SPLIT | RIGHT_SPLIT;
const LEFT_DOUBLE_SPLIT: u32 = CENTER_SPLIT | LEFT_SPLIT;
const TRIPLE_SPLIT: u32 = CENTER_SPLIT | RIGHT_SPLIT | LEFT_SPLIT;
const INVALID_POINTER: u32 = u32::MAX;

const NEXT: usize = 0;
const PREV: usize = 1;
const TWIN: usize = 2;

const UNCHANGED_ELEMENT: u32 = 0;

// use pure AoS for now
pub struct Bisector {
    heapid: u32,
    state: u32,
    command: AtomicU32,
    allocation_slots: [u32; 4],
    // next, prev, twin
}

pub fn heap_id_depth(heapid: u32) -> u32 {
    return 32 - heapid.leading_zeros();
}
impl Bisector {
    pub fn split_element(
        &mut self,
        curr_id: u32,                   // global thread index
        bisector_data: &mut [Bisector], // list of bisectors (global ordering)
        neighbors_buffer: &[[u32; 3]],  // list of neighbors (global ordering)
        base_depth: u32,                // base depth of the cbt
        memory_count: AtomicU32,        // memory pool
        allocation_count: AtomicU32,    // how many bisectors need to allocate memory
        allocation_buffer: &mut [u32],  // stores heapids of bisectors that need to allocate memory
        heapid_buffer: &mut [u32],      // stores heapids of bisectors
    ) {
        let curr_neighbors = neighbors_buffer[curr_id as usize];

        // check if we should release control to next neighbor
        let next = curr_neighbors[NEXT];
        if next != INVALID_POINTER {
            let next_neighbors = neighbors_buffer[next as usize];
            if next_neighbors[TWIN] == curr_id
                && bisector_data[next as usize].state != UNCHANGED_ELEMENT
            {
                return;
            }
        }

        // check if we should release control to prev neighbor
        let prev = curr_neighbors[PREV];
        if prev != INVALID_POINTER {
            let prev_neighbors = neighbors_buffer[prev as usize];
            if prev_neighbors[TWIN] == curr_id
                && bisector_data[prev as usize].state != UNCHANGED_ELEMENT
            {
                return;
            }
        }

        let current_depth = heap_id_depth(curr_id);

        let twin_id = curr_neighbors[TWIN];
        let max_required_memory =
           // boundary
            if
            twin_id == INVALID_POINTER {
            1
            // perfect matching
        } else if neighbors_buffer[twin_id as usize][TWIN] == curr_id {
            2
            // worst case: we must traverse up the tree
        } else {
            2 * (current_depth - base_depth) - 1
        };

        let remaining_memory = memory_count.fetch_sub(max_required_memory, Ordering::Relaxed);

        if remaining_memory < max_required_memory {
            memory_count.fetch_add(max_required_memory, Ordering::Relaxed);
            return;
        }

        let base_pattern = self.command.fetch_or(CENTER_SPLIT, Ordering::Relaxed);
        // another thread got to this bisector first
        if base_pattern != NO_SPLIT {
            memory_count.fetch_add(max_required_memory, Ordering::Relaxed);
            return;
        }

        let target_location = allocation_count.fetch_add(1, Ordering::Relaxed);
        allocation_buffer[target_location as usize] = curr_id;

        let mut used_memory = 1;

        // let mut previous_command = base_pattern;
        let mut twin_id = twin_id;
        let mut curr_id = curr_id;
        let mut current_depth = current_depth;
        // this is a recursive algorithm implemented as a loop
        loop {
            // base case: refinement edge is a boundary
            if twin_id == INVALID_POINTER {
                break;
            }

            // setup neighbor data for loop iteration
            let twin_heapid = heapid_buffer[twin_id as usize];
            let twin_bisector_data = &bisector_data[twin_id as usize];
            let twin_depth = heap_id_depth(twin_heapid);
            let twin_neighbors = neighbors_buffer[twin_id as usize];

            // base case: we have a twin!
            if twin_depth == current_depth {
                let twin_previous_command = bisector_data[twin_id as usize]
                    .command
                    .fetch_or(CENTER_SPLIT, Ordering::Relaxed);

                // twin needs to allocate for its split
                if twin_previous_command == NO_SPLIT {
                    let target_location = allocation_count.fetch_add(1, Ordering::Relaxed);
                    allocation_buffer[target_location as usize] = twin_id;
                    used_memory += 1;
                }
                break;
            }
            // twin must be split as part of compatibility chain
            else {
                // add another subidivision to the twin
                // we need to splits because we are cutting into the twin from the side
                // and save the twin's previous command
                let twin_previous_command = if twin_neighbors[NEXT] == curr_id {
                    twin_bisector_data
                        .command
                        .fetch_or(RIGHT_DOUBLE_SPLIT, Ordering::Relaxed)
                } else {
                    // twin_neighbors[PREV] = curr_id
                    twin_bisector_data
                        .command
                        .fetch_or(LEFT_DOUBLE_SPLIT, Ordering::Relaxed)
                };
                // why is twin_neighbors[TWIN] != curr_id?

                // twin was already split
                // so we allocate memory for the one extra split the current bisector requested
                if twin_previous_command != NO_SPLIT {
                    used_memory += 1;
                    break;
                } else {
                    // we need twin needs to allocate memory for at least 2 splits
                    let target_location = allocation_count.fetch_add(1, Ordering::Relaxed);
                    allocation_buffer[target_location as usize] = twin_id;

                    used_memory += 2;

                    // loop continues
                    // we propagate the twin's split to its twin
                    curr_id = twin_id;
                    current_depth = twin_depth;
                    twin_id = neighbors_buffer[curr_id as usize][TWIN];
                }
            }
        }

        memory_count.fetch_add(
            max((max_required_memory as i32) - (used_memory as i32), 0) as u32,
            Ordering::Relaxed,
        );
    }

    pub fn heapid_to_vertices(
        heapid: u32,
        base_level: u32,
        final_level: u32,
        root_bisectors: &[[Vec3; 3]], // list of 3 triangles each with 3 verts
    ) -> [Vec3; 3] {
        let mask = (0xFFFF_FFFF as u32) << (base_level + 1);
        let root_bisector_index = heapid & !mask;

        let split_chain = heapid >> base_level;
        let mut curr_bisector = root_bisectors[root_bisector_index as usize];
        let mut peak_idx: i32 = 2; // index of vertex that is the peak

        for i in 0..(final_level - base_level) {
            let bit = (split_chain >> i) & 0b0000_0000_0000_0001;

            let replacement_slot = if bit != 0 { peak_idx + 1 } else { peak_idx - 1 } % 3;
            let refinement_edge = [
                curr_bisector[(peak_idx + 1 % 3) as usize],
                curr_bisector[(peak_idx + 2 % 3) as usize],
            ];
            let new_vertex = refinement_edge[0].midpoint(refinement_edge[1]);

            curr_bisector[replacement_slot as usize] = new_vertex;
            peak_idx = replacement_slot;
        }

        return curr_bisector;
    }
}

struct Face {
    v0: u32,
    num_verts: u32,
}

struct HalfedgeMesh {
    verts: Vec<Vec3>,

    faces: Vec<Face>,
    indices: Vec<u32>,

    // halfedge data
    next: Vec<u32>,
    prev: Vec<u32>,
    twin: Vec<u32>,
}

impl HalfedgeMesh {
    pub fn new(
        verts: Vec<Vec3>,
        faces: Vec<Face>,
        indices: Vec<u32>,
        next: Vec<u32>,
        prev: Vec<u32>,
        twin: Vec<u32>,
    ) -> Self {
        Self {
            verts: verts,
            next: next,
            prev: prev,
            twin: twin,
            faces: faces,
            indices: indices,
        }
    }
    pub fn create_root_bisectors(&self) -> Vec<[Vec3; 3]> {
        let mut ret = Vec::<[Vec3; 3]>::with_capacity(self.next.len());
        for face in self.faces.iter() {
            // all bisectors share the midpoint of the face as v2
            let mut face_midpoint = Vec3::new(0.0, 0.0, 0.0);
            for vert_idx in face.v0..(face.v0 + face.num_verts) {
                face_midpoint += self.verts[vert_idx as usize];
            }
            let face_midpoint = face_midpoint / Vec3::splat(face.num_verts as f32);

            // get first 2 verts for each bisector
            for vert_idx in face.v0..(face.v0 + face.num_verts - 1) {
                let v0 = self.verts[vert_idx as usize];
                let v1 = self.verts[(vert_idx + 1) as usize];

                let root_bisector = [v0, v1, face_midpoint];
                ret.push(root_bisector);
            }
        }

        return ret;
    }
}

const COMMAND_EDGE_LUT: [[[u8;
        2 /* number of slots going to an edge */];
        3 /* number of edges */];
        4 /*number of potential commmands */]
= [
    // CENTRAL_SPLIT
    [[0, u8::MAX], [1, u8::MAX], [1, 0]],
    // RIGHT_DOUBLE_SPLIT
    [[2, 0], [1, u8::MAX], [1, 2]],
    // LEFT_DOUBLE_SPLIT
    [[0, u8::MAX], [2, 1], [1, 0]],
    // TRIPLE_SPLIT
    [[2, 0], [3, 1], [1, 2]],
];

// given a split command and an edge index (NEXT, PREV, or TWIN)
// return up to two allocation slots that touch the edge
// the first slot appears on the left when viewing the triangle from the outside
// with the specified edge at the bottom and its opposing vertex at the top
pub fn get_edge_slots(command: u32, edge_type: usize) -> [u8; 2] {
    debug_assert!(
        command == CENTER_SPLIT
            || command == RIGHT_DOUBLE_SPLIT
            || command == LEFT_DOUBLE_SPLIT
            || command == TRIPLE_SPLIT,
    );
    debug_assert!(edge_type == NEXT || edge_type == PREV || edge_type == TWIN);

    return COMMAND_EDGE_LUT[(command / 2) as usize][edge_type as usize];
}

const NEXT_U8: u8 = 0;
const PREV_U8: u8 = 1;
const TWIN_U8: u8 = 2;
const CHILD_EDGE_TYPE_LUT: [[[u8; 2]; 3]; 4] = [
    // CENTRAL_SPLIT
    [[TWIN_U8, TWIN_U8], [TWIN_U8, TWIN_U8], [NEXT_U8, PREV_U8]],
    // RIGHT_DOUBLE_SPLIT
    [[NEXT_U8, PREV_U8], [TWIN_U8, TWIN_U8], [NEXT_U8, TWIN_U8]],
    // LEFT_DOUBLE_SPLIT
    [[TWIN_U8, TWIN_U8], [NEXT_U8, PREV_U8], [TWIN_U8, NEXT_U8]],
    // TRIPLE_SPLIT
    [[NEXT_U8, PREV_U8], [NEXT_U8, PREV_U8], [TWIN_U8, TWIN_U8]],
];
pub fn get_child_edge_types(command: u32, parent_edge_type: usize) -> [u8; 2] {
    return CHILD_EDGE_TYPE_LUT[(command / 2) as usize][parent_edge_type as usize];
}

struct SplitEdge {
    slot: u32,           // allocation slot
    edge_type: u8,       // whether it is NEXT, PREV, or TWIN
    neighbor_index: u32, // index of neighbor allocation slot
}

// convention: when looking at the bisector from the outside,
// with the newest vertex (of the child) on top, splitedge x is left of splitedge y
struct EdgeData {
    x: SplitEdge,
    y: SplitEdge,
}
pub fn update_pointers(
    curr_index: u32,
    bisector_data: Vec<Bisector>,
    neighbor_buffer: &mut Vec<[u32; 3]>,
) {
    let curr_bisector = &bisector_data[curr_index as usize];
    let curr_command = curr_bisector.command.load(Ordering::Relaxed);
    debug_assert!(curr_command != NO_SPLIT);
    let neighbors = neighbor_buffer[curr_index as usize];
    for (bisector_edge_idx, neighbor_index) in neighbors.into_iter().enumerate() {
        if neighbor_index == u32::MAX {
            continue;
        }
        let neighbor = &bisector_data[neighbor_index as usize];
        let neighbor_command = neighbor.command.load(Ordering::Relaxed);

        if neighbor_command == NO_SPLIT {
            todo!()
        }

        let neighbor_neighbors = neighbor_buffer[neighbor_index as usize];

        let neighbor_edge = if neighbor_neighbors[NEXT] == curr_index {
            NEXT
        } else if neighbor_neighbors[PREV] == curr_index {
            PREV
        } else {
            debug_assert!(neighbor_neighbors[TWIN] == curr_index);
            TWIN
        };
        let bisector_slots = get_edge_slots(curr_command, bisector_edge_idx);
        let neighbor_slots = get_edge_slots(neighbor_command, neighbor_edge);
        let edge_types = get_child_edge_types(curr_command, bisector_edge_idx);
        let curr_edge_data = EdgeData {
            // neighbor slots are reversed curr X must map to neighbor Y and vice versa
            //        / \
            //       / | \
            //      / Y|X \
            //     +---|---+
            //      \ X|Y /
            //       \ | /
            //        \ /
            x: SplitEdge {
                slot: curr_bisector.allocation_slots[bisector_slots[0] as usize],
                edge_type: edge_types[0],
                neighbor_index: neighbor.allocation_slots[neighbor_slots[1] as usize],
            },
            y: SplitEdge {
                slot: curr_bisector.allocation_slots[bisector_slots[1] as usize],
                edge_type: edge_types[1],
                neighbor_index: neighbor.allocation_slots[neighbor_slots[0] as usize],
            },
        };

        // update neighbor buffer
        neighbor_buffer[curr_edge_data.x.slot as usize][curr_edge_data.x.edge_type as usize] =
            curr_edge_data.x.neighbor_index;
        neighbor_buffer[curr_edge_data.y.slot as usize][curr_edge_data.y.edge_type as usize] =
            curr_edge_data.y.neighbor_index;
    }

    // update pointers from bisector children to other bisector children
    if curr_command == CENTER_SPLIT {
        neighbor_buffer[curr_bisector.allocation_slots[1] as usize][NEXT] =
            curr_bisector.allocation_slots[0];
        neighbor_buffer[curr_bisector.allocation_slots[0] as usize][PREV] =
            curr_bisector.allocation_slots[1];
    } else if curr_command == RIGHT_DOUBLE_SPLIT {
    } else if curr_command == LEFT_DOUBLE_SPLIT {
    } else {
        debug_assert!(curr_command == TRIPLE_SPLIT);
    }
}
