#![allow(unsafe_op_in_unsafe_fn)]

use std::{
    cmp::max,
    iter::Map,
    marker::PhantomData,
    ptr::null_mut,
    slice,
    sync::atomic::{AtomicU32, Ordering},
    u32,
};

use glam::{Mat4, Vec3, Vec4};

use ash::{
    Device,
    vk::{self, Extent2D},
};

pub fn byteslice<T: Sized>(p: &T) -> &[u8] {
    unsafe {
        ::core::slice::from_raw_parts((p as *const T) as *const u8, ::core::mem::size_of::<T>())
    }
}

pub fn cast_slice<T: Sized>(p: &[T]) -> &[u8] {
    unsafe {
        ::core::slice::from_raw_parts(
            p.as_ptr() as *const u8,
            ::core::mem::size_of::<T>() * p.len(),
        )
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

pub fn record_submit_commandbuffer<F: FnOnce(&Device, vk::CommandBuffer)>(
    device: &Device,
    command_buffer: vk::CommandBuffer,
    command_buffer_reuse_fence: vk::Fence,
    submit_queue: vk::Queue,
    wait_mask: &[vk::PipelineStageFlags],
    wait_semaphores: &[vk::Semaphore],
    signal_semaphores: &[vk::Semaphore],
    f: F,
) {
    unsafe {
        device
            .wait_for_fences(&[command_buffer_reuse_fence], true, u64::MAX)
            .expect("Wait for fence failed.");

        device
            .reset_fences(&[command_buffer_reuse_fence])
            .expect("Reset fences failed");

        device
            .reset_command_buffer(
                command_buffer,
                vk::CommandBufferResetFlags::RELEASE_RESOURCES,
            )
            .expect("Reset command buffer failed.");

        let command_buffer_begin_info = vk::CommandBufferBeginInfo::default()
            .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT);

        device
            .begin_command_buffer(command_buffer, &command_buffer_begin_info)
            .expect("Begin commandbuffer.");

        f(device, command_buffer);

        device
            .end_command_buffer(command_buffer)
            .expect("End command buffer.");

        let command_buffers = vec![command_buffer];

        let submit_info = vk::SubmitInfo::default()
            .wait_semaphores(wait_semaphores)
            .wait_dst_stage_mask(wait_mask)
            .command_buffers(&command_buffers)
            .signal_semaphores(signal_semaphores);

        device
            .queue_submit(submit_queue, &[submit_info], command_buffer_reuse_fence)
            .expect("Queue submit failed.");
    }
}
pub struct AllocatedBuffer<T> {
    buffer: vk::Buffer,
    allocation: vk::DeviceMemory,
    address: vk::DeviceAddress,
    num_elems: u64,
    _marker: PhantomData<T>,
}

impl<T> AllocatedBuffer<T> {
    fn map_memory(&self, device: &ash::Device) -> *mut T {
        unsafe {
            device
                .map_memory(
                    self.allocation,
                    0,
                    size_of::<T>() as u64 * self.num_elems,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Map Memory.") as *mut T
        }
    }
}

pub struct MappedBuffer<T> {
    allocated_buffer: AllocatedBuffer<T>,
    ptr: *mut T,
}

pub trait GPUBuffer<T> {
    fn new(
        device: &ash::Device,
        num_elems: u64,
        mem_props: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_type: vk::MemoryPropertyFlags,
    ) -> Self;

    fn new_with_data(
        device: &ash::Device,
        data: &[T],
        mem_props: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_type: vk::MemoryPropertyFlags,
        command_buffer: vk::CommandBuffer,
        command_buffer_reuse_fence: vk::Fence,
        queue: vk::Queue,
    ) -> Self;

    fn device_address(&self) -> vk::DeviceAddress;
    fn destroy(&mut self, device: &ash::Device);

    fn allocation(&self) -> vk::DeviceMemory;
    fn buffer(&self) -> vk::Buffer;

    fn size_in_bytes(&self) -> u64;
}

pub trait GPUMappedBuffer<T> {
    unsafe fn mapped_slice(&self) -> &mut [T];
    fn copy_from_slice(&self, src: &[T]);
}

impl<T: Copy> GPUMappedBuffer<T> for MappedBuffer<T> {
    unsafe fn mapped_slice(&self) -> &mut [T] {
        slice::from_raw_parts_mut(self.ptr, self.allocated_buffer.num_elems as usize)
    }

    fn copy_from_slice(&self, src: &[T]) {
        let mapped_slice;
        unsafe {
            mapped_slice = self.mapped_slice();
        }
        mapped_slice.copy_from_slice(src);
    }
}

impl<T: Copy> GPUBuffer<T> for AllocatedBuffer<T> {
    fn new(
        device: &ash::Device,
        num_elems: u64,
        mem_props: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_type: vk::MemoryPropertyFlags,
    ) -> Self {
        let size = size_of::<T>() as u64 * num_elems;
        unsafe {
            // create buffer handle
            let buffer_create_info = vk::BufferCreateInfo::default()
                .size(size)
                .usage(usage | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS)
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
                .expect("Failed to allocate buffer memory.");
            device
                .bind_buffer_memory(buffer, buffer_memory, 0)
                .expect("Could not bind buffer to its memory.");

            let address = device
                .get_buffer_device_address(&vk::BufferDeviceAddressInfo::default().buffer(buffer));

            Self {
                buffer: buffer,
                allocation: buffer_memory,
                address: address,
                num_elems: num_elems,
                _marker: PhantomData::default(),
            }
        }
    }

    fn new_with_data(
        device: &ash::Device,
        data: &[T],
        mem_props: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_type: vk::MemoryPropertyFlags,
        command_buffer: vk::CommandBuffer,
        command_buffer_reuse_fence: vk::Fence,
        queue: vk::Queue,
    ) -> Self {
        let num_elems = data.len() as u64;
        let buffer: AllocatedBuffer<T> = GPUBuffer::<T>::new(
            device,
            num_elems,
            mem_props,
            usage | vk::BufferUsageFlags::TRANSFER_DST,
            sharing_mode,
            memory_type,
        );
        let mut staging_buffer: MappedBuffer<T> = GPUBuffer::<T>::new(
            &device,
            num_elems,
            mem_props,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::SharingMode::EXCLUSIVE,
            vk::MemoryPropertyFlags::HOST_VISIBLE,
        );
        staging_buffer.copy_from_slice(data);
        unsafe {
            record_submit_commandbuffer(
                &device,
                command_buffer,
                command_buffer_reuse_fence,
                queue,
                &[],
                &[],
                &[],
                |device, command_buffer| {
                    let copy_region =
                        vk::BufferCopy::default().size(staging_buffer.size_in_bytes());
                    device.cmd_copy_buffer(
                        command_buffer,
                        staging_buffer.buffer(),
                        buffer.buffer,
                        &[copy_region],
                    );
                },
            );
            device.device_wait_idle().expect("Wait idle.");
            staging_buffer.destroy(device);
        }

        return buffer;
    }

    fn device_address(&self) -> vk::DeviceAddress {
        self.address
    }

    fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.free_memory(self.allocation, None);
            device.destroy_buffer(self.buffer, None);
        }
    }

    fn allocation(&self) -> vk::DeviceMemory {
        self.allocation
    }

    fn buffer(&self) -> vk::Buffer {
        self.buffer
    }

    fn size_in_bytes(&self) -> u64 {
        size_of::<T>() as u64 * self.num_elems
    }
}

impl<T: Copy> GPUBuffer<T> for MappedBuffer<T> {
    fn allocation(&self) -> vk::DeviceMemory {
        self.allocated_buffer.allocation
    }

    fn buffer(&self) -> vk::Buffer {
        self.allocated_buffer.buffer
    }

    fn size_in_bytes(&self) -> u64 {
        self.allocated_buffer.size_in_bytes()
    }

    fn destroy(&mut self, device: &ash::Device) {
        unsafe {
            device.unmap_memory(self.allocated_buffer.allocation);
        }
        self.ptr = null_mut::<T>();
        self.allocated_buffer.destroy(device);
    }

    fn new(
        device: &ash::Device,
        num_elems: u64,
        mem_props: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_type: vk::MemoryPropertyFlags,
    ) -> Self {
        let allocated_buffer: AllocatedBuffer<T> = GPUBuffer::<T>::new(
            device,
            num_elems,
            mem_props,
            usage,
            sharing_mode,
            memory_type | vk::MemoryPropertyFlags::HOST_VISIBLE,
        );
        let ptr = allocated_buffer.map_memory(device);

        Self {
            allocated_buffer: allocated_buffer,
            ptr: ptr,
        }
    }

    fn new_with_data(
        device: &ash::Device,
        data: &[T],
        mem_props: vk::PhysicalDeviceMemoryProperties,
        usage: vk::BufferUsageFlags,
        sharing_mode: vk::SharingMode,
        memory_type: vk::MemoryPropertyFlags,
        command_buffer: vk::CommandBuffer,
        command_buffer_reuse_fence: vk::Fence,
        queue: vk::Queue,
    ) -> Self {
        let allocated_buffer: AllocatedBuffer<T> = GPUBuffer::<T>::new_with_data(
            device,
            data,
            mem_props,
            usage,
            sharing_mode,
            memory_type | vk::MemoryPropertyFlags::HOST_VISIBLE,
            command_buffer,
            command_buffer_reuse_fence,
            queue,
        );
        let ptr = allocated_buffer.map_memory(device);
        Self {
            allocated_buffer: allocated_buffer,
            ptr: ptr,
        }
    }
    fn device_address(&self) -> vk::DeviceAddress {
        self.allocated_buffer.device_address()
    }
}
pub struct CBT {
    pub depth: u32, // number of edges from the root to a furthest leaf
    pub interior: Vec<u32>,
    pub leaves: Vec<AtomicU32>,
}

pub const BITFIELD_INT_SIZE: u32 = 32;
impl CBT {
    pub fn new(depth: u32) -> CBT {
        let bitfield_length = 1 << depth;
        let num_leaves = bitfield_length / (BITFIELD_INT_SIZE as u64);
        let num_internal = 2 * num_leaves - 1;

        let interior = vec![0 as u32; num_internal as usize];

        let mut leaves = Vec::<AtomicU32>::new();
        leaves.resize_with(num_leaves as usize, || AtomicU32::new(0));

        return CBT {
            depth: depth,
            interior: interior,
            // occupancy bitfield in little endian bit order
            // ie. between u32: smaller idx -> larger idx
            //     within  u32:         lsb -> msb
            // WARNING: this is the OPPOSITE directionality as expected from bit shifting operations
            leaves: leaves,
        };
    }

    pub fn reduce(&mut self) {
        let interior_offset = self.interior.len() / 2;

        for i in 0..(interior_offset + 1) {
            let num_ones = self.leaves[i].load(Ordering::Relaxed).count_ones();
            self.interior[interior_offset + i] = num_ones;
        }

        // index_of_last_level = depth
        // 32 (depth) -> 16 (depth - 1) -> 8 (depth - 2) -> 4 (depth - 3) -> 2 (depth - 4) -> 1 (depth - 5)
        // all these levels have been filled
        let deepest_filled_level = self.depth - BITFIELD_INT_SIZE.ilog2();

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
        let cap = self.leaves.len() as u32 * BITFIELD_INT_SIZE;
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

        let leaf_idx = base_idx * BITFIELD_INT_SIZE;
        // get the index of the bit in the leaf
        let bit_idx = {
            let leaf = self.leaves[base_idx as usize].load(Ordering::Relaxed);
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

        let leaf_idx = base_idx * BITFIELD_INT_SIZE;
        // get the index of the bit in the leaf
        let bit_idx = {
            let leaf = self.leaves[base_idx as usize].load(Ordering::Relaxed);
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
        let leaf_idx = index / BITFIELD_INT_SIZE;
        let bit_idx = index % BITFIELD_INT_SIZE;

        let bit = (1 << bit_idx) as u32;
        self.leaves[leaf_idx as usize].fetch_or(bit, Ordering::Relaxed);
    }

    pub fn unset_bit(&mut self, index: u32) {
        let leaf_idx = index / BITFIELD_INT_SIZE;
        let bit_idx = index % BITFIELD_INT_SIZE;

        let bit = (1 << bit_idx) as u32;
        self.leaves[leaf_idx as usize].fetch_and(!bit, Ordering::Relaxed);
    }
}

pub const UNCHANGED_ELEMENT: u32 = 0;
pub const SIMPLIFY: u32 = 1;
pub const SPLIT: u32 = 2;

pub const NO_SPLIT: u32 = 0;
pub const CENTER_SPLIT: u32 = 1;
pub const RIGHT_SPLIT: u32 = 1 << 1;
pub const LEFT_SPLIT: u32 = 1 << 2;
pub const RIGHT_DOUBLE_SPLIT: u32 = CENTER_SPLIT | RIGHT_SPLIT;
pub const LEFT_DOUBLE_SPLIT: u32 = CENTER_SPLIT | LEFT_SPLIT;
pub const TRIPLE_SPLIT: u32 = CENTER_SPLIT | RIGHT_SPLIT | LEFT_SPLIT;
pub const INVALID_POINTER: u32 = u32::MAX;

pub const NEXT: usize = 0;
pub const PREV: usize = 1;
pub const TWIN: usize = 2;

pub fn heap_id_depth(heapid: u32) -> u32 {
    find_msb(heapid)
}
pub fn find_msb(x: u32) -> u32 {
    assert!(x != 0);
    return 31 - x.leading_zeros();
}

// TODO: look up in the want split buffer
pub fn split_element(
    curr_id: u32,                  // memory block index
    neighbors_buffer: &[[u32; 3]], // list of neighbors (global ordering)
    base_depth: u32,               // base depth of the cbt
    memory_count: &AtomicU32,      // memory pool
    allocation_count: &AtomicU32,  // how many bisectors need to allocate memory
    allocation_buffer: &mut [u32], // stores heapids of bisectors that need to allocate memory
    heapid_buffer: &[u32],         // stores heapids of bisectors
    bisector_command_buffer: &mut [AtomicU32],
    bisector_state_buffer: &[u32],
) {
    let curr_neighbors = neighbors_buffer[curr_id as usize];

    // check if we should release control to next neighbor
    let next = curr_neighbors[NEXT];
    if next != INVALID_POINTER {
        let next_neighbors = neighbors_buffer[next as usize];
        if next_neighbors[TWIN] == curr_id
            && bisector_state_buffer[next as usize] != UNCHANGED_ELEMENT
        {
            return;
        }
    }

    // check if we should release control to prev neighbor
    let prev = curr_neighbors[PREV];
    if prev != INVALID_POINTER {
        let prev_neighbors = neighbors_buffer[prev as usize];
        if prev_neighbors[TWIN] == curr_id
            && bisector_state_buffer[prev as usize] != UNCHANGED_ELEMENT
        {
            return;
        }
    }

    // println!("curr_id: {:b}", curr_id);
    let heapid = heapid_buffer[curr_id as usize];
    let current_depth = heap_id_depth(heapid);

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

    let base_pattern =
        bisector_command_buffer[curr_id as usize].fetch_or(CENTER_SPLIT, Ordering::Relaxed);
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
        let twin_depth = heap_id_depth(twin_heapid);
        let twin_neighbors = neighbors_buffer[twin_id as usize];

        // base case: we have a twin!
        if twin_depth == current_depth {
            let twin_previous_command =
                bisector_command_buffer[twin_id as usize].fetch_or(CENTER_SPLIT, Ordering::Relaxed);

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
                bisector_command_buffer[twin_id as usize]
                    .fetch_or(RIGHT_DOUBLE_SPLIT, Ordering::Relaxed)
            } else {
                // twin_neighbors[PREV] = curr_id
                bisector_command_buffer[twin_id as usize]
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
    root_bisectors: &[[Vec3; 3]], // list of 3 triangles each with 3 verts
) -> [Vec3; 3] {
    // heapid: 0b000...01 root_bisector_index split_code
    // the most significant 1 is a "tag" that allows us to compute the length of the split code
    // root_bisector_index takes `base_level` bits
    // split_code takes up a number of bits equal to the number of tree edges from the bisector to its root bisector

    assert!(heapid != 0);
    let depth = heap_id_depth(heapid);
    let num_split_code_bits = depth - base_level;

    let heapid_without_tag = !(1 << depth) & heapid;
    let root_bisector_index = heapid_without_tag >> num_split_code_bits;
    let split_code = !(0xFFFF_FFFF << num_split_code_bits) & heapid;

    let mut curr_bisector = root_bisectors[root_bisector_index as usize];
    let mut peak_idx: u32 = 2; // index of vertex that is the peak
    for i in (0..(depth - base_level)).rev() {
        let bit = (split_code >> i) & 0b0000_0000_0000_0001;

        let replacement_slot = (if bit == 0 { peak_idx + 2 } else { peak_idx + 1 }) % 3;
        let refinement_edge = [
            curr_bisector[((peak_idx + 1) % 3) as usize],
            curr_bisector[((peak_idx + 2) % 3) as usize],
        ];
        let new_vertex = refinement_edge[0].midpoint(refinement_edge[1]);

        curr_bisector[replacement_slot as usize] = new_vertex;
        peak_idx = replacement_slot;
    }

    return curr_bisector;
}

pub struct Face {
    pub v0: u32,
    pub num_verts: u32,
}

pub struct HalfedgeMesh {
    pub verts: Vec<Vec3>,
    pub faces: Vec<Face>,
    pub indices: Vec<u32>,
    pub neighbors: Vec<[u32; 3]>,
}

impl HalfedgeMesh {
    pub fn compute_root_bisector_vertices(&self) -> Vec<[Vec3; 3]> {
        let mut ret = Vec::<[Vec3; 3]>::with_capacity(self.neighbors.len());
        for face in self.faces.iter() {
            // all bisectors share the midpoint of the face as v2
            let mut face_midpoint = Vec3::new(0.0, 0.0, 0.0);
            for vert_idx in face.v0..(face.v0 + face.num_verts) {
                face_midpoint += self.verts[self.indices[vert_idx as usize] as usize];
            }
            let face_midpoint = face_midpoint / Vec3::splat(face.num_verts as f32);

            // get first 2 verts for each bisector
            for i in 0..face.num_verts {
                let v0 = self.verts[self.indices[(face.v0 + i) as usize] as usize];
                let v1 = self.verts
                    [self.indices[(face.v0 + ((i + 1) % face.num_verts)) as usize] as usize];

                let root_bisector = [v0, v1, face_midpoint];
                ret.push(root_bisector);
            }
        }
        return ret;
    }
}

pub fn allocate(
    curr_id: u32,
    memory_counter: &AtomicU32,
    cbt: &CBT,
    bisector_command_buffer: &[AtomicU32],
    allocation_indices_buffer: &mut [[u32; 4]],
    bisector_state_buffer: &mut [u32],
) {
    let command = bisector_command_buffer[curr_id as usize].load(Ordering::Relaxed);

    let num_allocations = command.count_ones() + 1;
    let first_bit_index = memory_counter.fetch_add(num_allocations, Ordering::Relaxed);

    let mut allocation_indices: [u32; 4] = [u32::MAX, u32::MAX, u32::MAX, u32::MAX];

    for (i, bit_index) in (first_bit_index..(first_bit_index + num_allocations)).enumerate() {
        let child_id = cbt.zero_bit_to_id(bit_index);
        allocation_indices[i] = child_id;
    }
    allocation_indices_buffer[curr_id as usize] = allocation_indices;
    // cleanup
    bisector_state_buffer[curr_id as usize] = SPLIT;
}

const COMMAND_EDGE_LUT: [[[u8;
        2 /* number of slots going to an edge */];
        3 /* number of edges */];
        4 /*number of potential commmands */]
= [
    // CENTRAL_SPLIT
    [[0, 0], [1, 1], [1, 0]],
    // RIGHT_DOUBLE_SPLIT
    [[2, 0], [1, 1], [1, 2]],
    // LEFT_DOUBLE_SPLIT
    [[0, 0], [2, 1], [1, 0]],
    // TRIPLE_SPLIT
    [[2, 0], [3, 1], [1, 2]],
];

// given a split command and an edge index (NEXT, PREV, or TWIN)
// return up to the allocation slots that touch each half of the edge
// if the edge has not been split, these slots will be the same
// the first slot appears on the left when viewing the triangle from the outside
// with the specified edge at the bottom and its opposing vertex at the top
pub fn get_edge_slots(command: u32, edge_type: usize) -> [u8; 2] {
    debug_assert!(
        command == NO_SPLIT
            || command == CENTER_SPLIT
            || command == RIGHT_DOUBLE_SPLIT
            || command == LEFT_DOUBLE_SPLIT
            || command == TRIPLE_SPLIT,
    );
    debug_assert!(edge_type == NEXT || edge_type == PREV || edge_type == TWIN);
    if command == NO_SPLIT {
        return [0, 0];
    }
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
    [[TWIN_U8, TWIN_U8], [NEXT_U8, PREV_U8], [TWIN_U8, PREV_U8]],
    // TRIPLE_SPLIT
    [[NEXT_U8, PREV_U8], [NEXT_U8, PREV_U8], [TWIN_U8, TWIN_U8]],
];
// given a split command of the parent and an edge type (NEXT, PREV, or TWIN)
// return the types of edges of the children corresponding to an edge
// eg. the twin edge of a parent undergoing a central split
// becomes the next edge of one child and the prev edge of another
pub fn get_child_edge_types(command: u32, parent_edge_type: usize) -> [u8; 2] {
    return CHILD_EDGE_TYPE_LUT[(command / 2) as usize][parent_edge_type as usize];
}

pub fn link_siblings(prev_sibling: u32, next_sibling: u32, neighbor_buffer: &mut [[u32; 3]]) {
    neighbor_buffer[prev_sibling as usize][NEXT] = next_sibling;
    neighbor_buffer[next_sibling as usize][PREV] = prev_sibling;
}

fn find_edge_type(curr_id: u32, neighbor_neighbors: [u32; 3]) -> usize {
    if neighbor_neighbors[NEXT] == curr_id {
        NEXT
    } else if neighbor_neighbors[PREV] == curr_id {
        PREV
    } else {
        if neighbor_neighbors[TWIN] != curr_id {
            println!("FUCK");
            println!("curr_id: {:?}", curr_id);
            println!("neighbor_neighbors: {:?}", neighbor_neighbors);
        }
        debug_assert!(neighbor_neighbors[TWIN] == curr_id);
        TWIN
    }
}

pub fn update_pointers(
    curr_id: u32,
    neighbor_buffer: &mut [[u32; 3]],
    heap_id_buffer: &mut [u32],
    bisector_command_buffer: &[AtomicU32],
    allocation_indices_buffer: &[[u32; 4]],
    cbt: &mut CBT,
) {
    let curr_heapid = heap_id_buffer[curr_id as usize];
    let curr_command = bisector_command_buffer[curr_id as usize].load(Ordering::Relaxed);
    debug_assert!(curr_command != NO_SPLIT);
    let neighbors = neighbor_buffer[curr_id as usize];

    let curr_allocation_indices = allocation_indices_buffer[curr_id as usize];

    // iterate over EDGES of the parent bisector that is being split
    for (bisector_edge_idx, neighbor_index) in neighbors.into_iter().enumerate() {
        // boundary edge, write child pointers and we're done

        let bisector_slots = get_edge_slots(curr_command, bisector_edge_idx); // 0
        let edge_types = get_child_edge_types(curr_command, bisector_edge_idx); // TWIN

        // write external pointers for first allocated slot
        let slot = curr_allocation_indices[bisector_slots[0] as usize] as usize; // 1
        let edge_type = edge_types[0] as usize; // TWIN
        // defaults if the neighbor does not exist
        let mut neighbor_allocation_indices = [u32::MAX, u32::MAX, u32::MAX, u32::MAX];
        let mut neighbor_slot = u32::MAX;
        let mut neighbor_slots = [0, 0];
        // the neighbor exists. find the slot in the neighbor adjacent to the first slot in the current bisector
        if neighbor_index != u32::MAX {
            let neighbor_command =
                bisector_command_buffer[neighbor_index as usize].load(Ordering::Relaxed);
            let neighbor_edge = find_edge_type(curr_id, neighbor_buffer[neighbor_index as usize]);
            if neighbor_command == NO_SPLIT {
                // non-split tris are not dispatched, so we update them as well
                // neighbor_buffer[neighbor_index] == curr_index by construction
                if curr_id != slot as u32 {
                    neighbor_buffer[neighbor_index as usize][neighbor_edge] = slot as u32;
                }
                neighbor_slot = neighbor_index;
            } else {
                neighbor_allocation_indices = allocation_indices_buffer[neighbor_index as usize];
                neighbor_slots = get_edge_slots(neighbor_command, neighbor_edge);
                neighbor_slot = neighbor_allocation_indices[neighbor_slots[1] as usize]; // left and right are flipped from the neighbor's perspective
            }
        }

        neighbor_buffer[slot][edge_type] = neighbor_slot;

        // this edge was split, write pointers to the second child
        if bisector_slots[0] != bisector_slots[1] {
            let slot = curr_allocation_indices[bisector_slots[1] as usize] as usize;
            let edge_type = edge_types[1] as usize;
            let neighbor_slot = neighbor_allocation_indices[neighbor_slots[0] as usize];

            neighbor_buffer[slot][edge_type] = neighbor_slot;
        }
    }

    // update pointers from bisector children to other bisector children
    // also writes new heapids
    // best to draw a picture for this
    // TODO: reduce branched global stores
    if curr_command == CENTER_SPLIT {
        link_siblings(
            curr_allocation_indices[0],
            curr_allocation_indices[1],
            neighbor_buffer,
        );

        // LEFT child must have the LOWER heapid
        // left child
        heap_id_buffer[curr_allocation_indices[1] as usize] = 2 * curr_heapid + 0;
        // right child
        heap_id_buffer[curr_allocation_indices[0] as usize] = 2 * curr_heapid + 1;
    } else if curr_command == RIGHT_DOUBLE_SPLIT {
        link_siblings(
            curr_allocation_indices[0],
            curr_allocation_indices[2],
            neighbor_buffer,
        );
        neighbor_buffer[curr_allocation_indices[0] as usize][TWIN] = curr_allocation_indices[1];
        neighbor_buffer[curr_allocation_indices[1] as usize][PREV] = curr_allocation_indices[0];

        // L
        heap_id_buffer[curr_allocation_indices[1] as usize] = 2 * curr_heapid;
        // RL
        heap_id_buffer[curr_allocation_indices[2] as usize] = 4 * curr_heapid + 2;
        // RR
        heap_id_buffer[curr_allocation_indices[0] as usize] = 4 * curr_heapid + 3;
    } else if curr_command == LEFT_DOUBLE_SPLIT {
        link_siblings(
            curr_allocation_indices[1],
            curr_allocation_indices[2],
            neighbor_buffer,
        );
        neighbor_buffer[curr_allocation_indices[2] as usize][TWIN] = curr_allocation_indices[0];
        neighbor_buffer[curr_allocation_indices[0] as usize][NEXT] = curr_allocation_indices[2];

        // R
        heap_id_buffer[curr_allocation_indices[0] as usize] = 2 * curr_heapid + 1;
        // LL
        heap_id_buffer[curr_allocation_indices[2] as usize] = 4 * curr_heapid;
        // LR
        heap_id_buffer[curr_allocation_indices[1] as usize] = 4 * curr_heapid + 1;
    } else {
        debug_assert!(curr_command == TRIPLE_SPLIT);
        link_siblings(
            curr_allocation_indices[1],
            curr_allocation_indices[3],
            neighbor_buffer,
        );
        link_siblings(
            curr_allocation_indices[0],
            curr_allocation_indices[2],
            neighbor_buffer,
        );
        neighbor_buffer[curr_allocation_indices[3] as usize][TWIN] = curr_allocation_indices[0];
        neighbor_buffer[curr_allocation_indices[0] as usize][TWIN] = curr_allocation_indices[3];

        // LL
        heap_id_buffer[curr_allocation_indices[3] as usize] = 4 * curr_heapid + 0;
        // LR
        heap_id_buffer[curr_allocation_indices[1] as usize] = 4 * curr_heapid + 1;
        // RL
        heap_id_buffer[curr_allocation_indices[2] as usize] = 4 * curr_heapid + 2;
        // RR
        heap_id_buffer[curr_allocation_indices[0] as usize] = 4 * curr_heapid + 3;
    }
    cbt.unset_bit(curr_id);
    for i in curr_allocation_indices {
        if i != u32::MAX {
            cbt.set_bit(i);
        }
    }
}

pub fn prepare_merge(
    curr_id: u32,
    neighbor_buffer: &mut [[u32; 3]],
    heap_id_buffer: &[u32],
    bisector_state_buffer: &[u32],
    simplification_counter: &AtomicU32,
    simplification_buffer: &mut [u32],
) {
    // Get the bisector and heapid
    println!("curr_id: {:?}", curr_id);
    let curr_heapid = heap_id_buffer[curr_id as usize]; // (garaunteed to be even)
    let curr_neighbors = neighbor_buffer[curr_id as usize];
    let curr_depth = heap_id_depth(curr_heapid);

    // Grab the pair neighbor (it has to exist)
    let next_id = curr_neighbors[NEXT];
    let next_heapid = heap_id_buffer[next_id as usize];
    let next_depth = heap_id_depth(next_heapid);
    let next_state = bisector_state_buffer[next_id as usize];
    // If they are not at the same depth or the pair is not to be simplified, we're done
    if next_depth != curr_depth || next_state != SIMPLIFY {
        return;
    }
    let next_neighbors = neighbor_buffer[next_id as usize];

    let mut num_pairs_to_merge = 1 as u32;
    // We need to identify our twin pair
    let twin_highid = next_neighbors[NEXT];
    let twin_lowid = curr_neighbors[PREV];
    if twin_lowid != u32::MAX {
        // Grab the two bisectors
        let twin_low_heapid = heap_id_buffer[twin_lowid as usize];
        let twin_high_heapid = heap_id_buffer[twin_highid as usize];

        // The current bisector is not the greatest element of the neighborhood, he will be handeled by twinLowBisect if needed
        if curr_heapid < twin_high_heapid {
            return;
        }

        // Compute the depth of both neighbors
        let twin_low_depth = heap_id_depth(twin_low_heapid);
        let twin_high_depth = heap_id_depth(twin_high_heapid);

        // If all four elements are not on the same
        if twin_low_depth != curr_depth || twin_high_depth != curr_depth {
            return;
        }
        debug_assert!(twin_high_heapid > twin_low_heapid);

        // Grab the two bisectors
        let twin_low_bisector_state = bisector_state_buffer[twin_lowid as usize];
        let twin_high_bisector_state = bisector_state_buffer[twin_highid as usize];

        // This element should not be doing the simplifications if:
        // - One of the four elements doesn't have the same depth
        // - One of the four elements isn't flagged for simplification
        if twin_low_bisector_state != SIMPLIFY || twin_high_bisector_state != SIMPLIFY {
            return;
        }
        num_pairs_to_merge = 2;
    }

    // remove reference to next
    if next_neighbors[TWIN] != u32::MAX {
        let next_twin_neighbors = neighbor_buffer[next_neighbors[TWIN] as usize];
        let edge_type = find_edge_type(next_id, next_twin_neighbors);
        neighbor_buffer[next_neighbors[TWIN] as usize][edge_type] = curr_id;
    }

    // enqueue 1 thread for each pair we need to merge
    let base_slot = simplification_counter.fetch_add(num_pairs_to_merge, Ordering::Relaxed);
    simplification_buffer[base_slot as usize] = curr_id;
    if num_pairs_to_merge == 2 {
        simplification_buffer[(base_slot + 1) as usize] = twin_highid;

        // remove reference to twin_low
        let twin_low_neighbors = neighbor_buffer[twin_lowid as usize];

        if twin_low_neighbors[TWIN] != u32::MAX {
            let twin_low_twin_neighbors = neighbor_buffer[twin_low_neighbors[TWIN] as usize];
            let edge_type = find_edge_type(twin_lowid, twin_low_twin_neighbors);
            neighbor_buffer[twin_low_neighbors[TWIN] as usize][edge_type] = twin_highid;
        }
    }
}

pub fn merge(
    curr_id: u32,
    neighbor_buffer: &mut [[u32; 3]],
    heap_id_buffer: &mut [u32],
    cbt: &mut CBT,
) {
    let curr_heapid = heap_id_buffer[curr_id as usize]; // (garaunteed to be even)
    let curr_neighbors = neighbor_buffer[curr_id as usize];

    let next_id = curr_neighbors[NEXT];
    let next_neighbors = neighbor_buffer[next_id as usize];

    let new_neighbors = [
        curr_neighbors[TWIN], // NEXT
        next_neighbors[TWIN], // PREV
        next_neighbors[NEXT], // TWIN
    ];

    neighbor_buffer[curr_id as usize] = new_neighbors;
    heap_id_buffer[curr_id as usize] = curr_heapid >> 1;
    heap_id_buffer[next_id as usize] = 0;

    cbt.unset_bit(next_id);
}

#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct SceneDataGPU {
    // written once at initialization
    pub root_bisector_vertices: vk::DeviceAddress,
    // our concurrent binary tree
    // cbt: CBT,
    pub cbt_interior: vk::DeviceAddress,
    pub cbt_leaves: vk::DeviceAddress,

    // classification stage
    pub bisector_state_buffer: vk::DeviceAddress,

    // prepare split
    pub bisector_split_command_buffer: vk::DeviceAddress,
    pub neighbors_buffer: vk::DeviceAddress,
    pub splitting_buffer: vk::DeviceAddress,
    pub heapid_buffer: vk::DeviceAddress,

    // allocate
    pub allocation_indices_buffer: vk::DeviceAddress,

    // split
    pub want_split_buffer: vk::DeviceAddress,

    // prepare merge
    pub want_merge_buffer: vk::DeviceAddress,

    // merge
    pub merging_bisector_buffer: vk::DeviceAddress,

    // draw
    pub vertex_buffer: vk::DeviceAddress,

    // integers
    pub num_memory_blocks: u32,
    pub base_depth: u32,
    pub cbt_depth: u32,
    pub debug_counter: u32,
}

pub struct SceneCPUHandles {
    // written once at initialization
    pub root_bisector_vertices: AllocatedBuffer<[Vec3; 3]>,
    // our concurrent binary tree
    // cbt: CBT,
    pub cbt_interior: MappedBuffer<u32>,
    pub cbt_leaves: MappedBuffer<u32>,

    // classification stage
    // pub classification_pipeline: vk::Pipeline,
    pub classify_pipeline: vk::Pipeline,
    pub bisector_state_buffer: AllocatedBuffer<u32>,

    // prepare split
    pub dispatch_split_pipeline: vk::Pipeline,
    pub bisector_split_command_buffer: AllocatedBuffer<u32>,
    pub neighbors_buffer: AllocatedBuffer<[u32; 3]>,
    pub splitting_buffer: AllocatedBuffer<u32>,
    pub heapid_buffer: AllocatedBuffer<u32>,

    // allocate
    pub dispatch_allocate_pipeline: vk::Pipeline,
    pub allocation_indices_buffer: AllocatedBuffer<[u32; 4]>,

    // split
    pub want_split_buffer: AllocatedBuffer<u32>,

    // prepare merge
    pub dispatch_prepare_merge_pipeline: vk::Pipeline,
    pub want_merge_buffer: AllocatedBuffer<u32>,

    // merge
    pub merging_bisector_buffer: AllocatedBuffer<u32>,
    pub vertex_compute_pipeline: vk::Pipeline,
    // draw
    pub vertex_buffer: AllocatedBuffer<[Vec3; 3]>,

    // reset
    pub reset_pipeline: vk::Pipeline,

    pub split_element_pipeline: vk::Pipeline,
    pub allocate_pipeline: vk::Pipeline,
    pub update_pointers_pipeline: vk::Pipeline,
    pub prepare_merge_pipeline: vk::Pipeline,
    pub merge_pipeline: vk::Pipeline,
    pub reduce_pipeline: vk::Pipeline,
    pub post_reduce_pipeline: vk::Pipeline,
}
#[repr(C)]
#[derive(Clone, Copy, Debug)]
pub struct DispatchDataGPU {
    // written to by reduce
    pub draw_indirect_command: vk::DrawIndirectCommand,

    // before split
    pub dispatch_split_command: vk::DispatchIndirectCommand,
    // before allocate
    pub dispatch_allocate_command: vk::DispatchIndirectCommand,
    // before prepare_merge
    pub dispatch_prepare_merge_command: vk::DispatchIndirectCommand,
    // written to by reduce
    pub dispatch_vertex_compute_command: vk::DispatchIndirectCommand,

    pub remaining_memory_count: u32,
    pub allocation_counter: u32,
    pub want_split_buffer_count: u32,
    pub splitting_buffer_count: u32,
    pub want_merge_buffer_count: u32,
    pub merging_bisector_count: u32,
    pub num_allocated_blocks: u32,

    // DEBUG
    pub debug_counter: u32,
}
pub struct PipelineData {
    // written once at initialization
    pub root_bisector_vertices: Vec<[Vec3; 3]>,
    pub num_memory_blocks: u32,
    pub base_depth: u32,

    // our concurrent binary tree
    pub cbt: CBT,

    // SoA for bisector struct
    pub neighbors_buffer: Vec<[u32; 3]>,
    pub heapid_buffer: Vec<u32>,
    pub allocation_indices_buffer: Vec<[u32; 4]>,
    // UNCHANGED, SPLIT, OR MERGE
    pub bisector_state_buffer: Vec<u32>,
    // has to be u32 for atomic reasons :(
    pub bisector_split_command_buffer: Vec<AtomicU32>,

    // intermediary buffers for various stages
    // use this later
    pub classification_buffer: Vec<u32>,

    pub remaining_memory_count: AtomicU32,
    pub allocation_counter: AtomicU32,
    pub want_split_buffer_count: AtomicU32,
    pub want_split_buffer: Vec<u32>,

    pub splitting_buffer_count: AtomicU32,
    pub splitting_buffer: Vec<u32>,

    pub want_merge_buffer_count: AtomicU32,
    pub want_merge_buffer: Vec<u32>,

    pub merging_bisector_count: AtomicU32,
    pub merging_bisector_buffer: Vec<u32>,

    pub vertex_buffer: Vec<[Vec3; 3]>,
}

pub fn linear_dispatch(x: u32) -> vk::DispatchIndirectCommand {
    vk::DispatchIndirectCommand { x: x, y: 1, z: 1 }
}

impl PipelineData {
    pub fn new(halfedge_mesh: HalfedgeMesh, depth: u32) -> Self {
        let root_bisector_vertices = halfedge_mesh.compute_root_bisector_vertices();
        let num_root_bisectors = root_bisector_vertices.len();
        let base_depth = if num_root_bisectors.count_ones() == 1 {
            // number of root bisectors is a power of 2
            num_root_bisectors.ilog2()
        } else {
            num_root_bisectors.ilog2() + 1
        };

        let mut cbt = CBT::new(depth);
        let num_memory_blocks = cbt.leaves.len() * BITFIELD_INT_SIZE as usize;

        let mut heapid_buffer = vec![0; num_memory_blocks];
        for i in 0..root_bisector_vertices.len() {
            heapid_buffer[i] = (i | (1 << base_depth)) as u32;
        }

        let mut neighbors_buffer = vec![[0, 0, 0]; num_memory_blocks];
        for i in 0..root_bisector_vertices.len() {
            neighbors_buffer[i] = halfedge_mesh.neighbors[i];
        }

        for i in 0..root_bisector_vertices.len() {
            cbt.set_bit(i as u32);
        }
        cbt.reduce();

        let allocation_indices_buffer =
            vec![[u32::MAX, u32::MAX, u32::MAX, u32::MAX]; num_memory_blocks];

        let mut bisector_split_command_buffer = Vec::<AtomicU32>::new();
        bisector_split_command_buffer.resize_with(num_memory_blocks, || AtomicU32::new(0));

        const UNINITIALIZED_TRIANGLE_VERTS: [Vec3; 3] = [
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
            Vec3::new(0.0, 0.0, 0.0),
        ];
        let mut vertex_buffer = vec![UNINITIALIZED_TRIANGLE_VERTS; num_memory_blocks];
        for i in 0..root_bisector_vertices.len() {
            vertex_buffer[i] = root_bisector_vertices[i];
        }

        Self {
            root_bisector_vertices: root_bisector_vertices,
            num_memory_blocks: num_memory_blocks as u32,
            base_depth: base_depth,

            cbt: cbt,

            heapid_buffer: heapid_buffer,
            neighbors_buffer: neighbors_buffer,
            allocation_indices_buffer: allocation_indices_buffer,
            bisector_split_command_buffer: bisector_split_command_buffer,

            classification_buffer: vec![0; num_memory_blocks],
            bisector_state_buffer: vec![0; num_memory_blocks],

            remaining_memory_count: AtomicU32::new(
                num_memory_blocks as u32 * BITFIELD_INT_SIZE - num_root_bisectors as u32,
            ),
            allocation_counter: AtomicU32::new(0),
            want_split_buffer_count: AtomicU32::new(0),
            want_split_buffer: vec![0; num_memory_blocks],

            splitting_buffer_count: AtomicU32::new(0),
            splitting_buffer: vec![0; num_memory_blocks],

            want_merge_buffer_count: AtomicU32::new(0),
            want_merge_buffer: vec![0; num_memory_blocks],

            merging_bisector_count: AtomicU32::new(0),
            merging_bisector_buffer: vec![0; num_memory_blocks],

            vertex_buffer: vertex_buffer,
        }
    }

    pub fn iterate(&mut self) {
        // DOES NOT CLASSIFY

        // write split commands
        for i in 0..self.want_split_buffer_count.load(Ordering::Relaxed) {
            // println!("loop i: {:?}", i);
            let curr_id = self.want_split_buffer[i as usize];
            // println!("loop curr_id: {:?}", curr_id);
            split_element(
                curr_id,
                &self.neighbors_buffer[..],
                self.base_depth,
                &self.remaining_memory_count,
                &self.splitting_buffer_count,
                &mut self.splitting_buffer[..],
                &self.heapid_buffer[..],
                &mut self.bisector_split_command_buffer[..],
                &self.bisector_state_buffer[..],
            );
        }

        // allocation
        for i in 0..self.splitting_buffer_count.load(Ordering::Relaxed) {
            let curr_id = self.splitting_buffer[i as usize];
            allocate(
                curr_id,
                &self.allocation_counter,
                &self.cbt,
                &self.bisector_split_command_buffer[..],
                &mut self.allocation_indices_buffer,
                &mut self.bisector_state_buffer,
            );
        }

        // split and update pointers
        for i in 0..self.splitting_buffer_count.load(Ordering::Relaxed) {
            let curr_id = self.splitting_buffer[i as usize];
            update_pointers(
                curr_id,
                &mut self.neighbors_buffer,
                &mut self.heapid_buffer,
                &self.bisector_split_command_buffer[..],
                &self.allocation_indices_buffer[..],
                &mut self.cbt,
            );
        }

        // prepare merge
        for i in 0..self.want_merge_buffer_count.load(Ordering::Relaxed) {
            let curr_id = self.want_merge_buffer[i as usize];
            prepare_merge(
                curr_id,
                &mut self.neighbors_buffer[..],
                &self.heapid_buffer[..],
                &self.bisector_state_buffer[..],
                &self.merging_bisector_count,
                &mut self.merging_bisector_buffer[..],
            );
        }

        // merge and update pointers
        for i in 0..self.merging_bisector_count.load(Ordering::Relaxed) {
            let curr_id = self.merging_bisector_buffer[i as usize];
            merge(
                curr_id,
                &mut self.neighbors_buffer[..],
                &mut self.heapid_buffer[..],
                &mut self.cbt,
            );
        }

        // cbt update
        self.cbt.reduce();
        self.remaining_memory_count.store(
            self.num_memory_blocks * BITFIELD_INT_SIZE - self.cbt.interior[0],
            Ordering::Relaxed,
        );

        for i in 0..self.cbt.interior[0] {
            let curr_id = self.cbt.one_bit_to_id(i);
            let curr_heapid = self.heapid_buffer[curr_id as usize];
            self.vertex_buffer[i as usize] = heapid_to_vertices(
                curr_heapid,
                self.base_depth,
                &self.root_bisector_vertices[..],
            );
        }
    }

    pub fn reset(&mut self) {
        self.splitting_buffer_count.store(0, Ordering::Relaxed);
        self.allocation_counter.store(0, Ordering::Relaxed);
        self.merging_bisector_count.store(0, Ordering::Relaxed);
        self.want_merge_buffer_count.store(0, Ordering::Relaxed);
        self.want_split_buffer_count.store(0, Ordering::Relaxed);

        self.bisector_split_command_buffer = self
            .bisector_split_command_buffer
            .iter_mut()
            .map(|_| AtomicU32::new(0))
            .collect();

        self.bisector_state_buffer = self.bisector_state_buffer.iter_mut().map(|_| 0).collect();
    }
}
#[derive(Debug)]
pub struct CameraState {
    pub pos: Vec3,
    // pitch (0, pi)
    pub pitch: f32,
    // yaw (-pi, pi)
    pub yaw: f32,

    // fov in radians
    pub fovy: f32,
    resolution: vk::Extent2D,
    aspect: f32,

    pub near: f32,
}

impl CameraState {
    pub fn new(
        pos: Vec3,
        pitch: f32,
        yaw: f32,
        fovy: f32,
        resolution: vk::Extent2D,
        near: f32,
    ) -> Self {
        let aspect = (resolution.width as f32) / (resolution.height as f32);
        Self {
            pos: pos,
            pitch: pitch,
            yaw: yaw,
            fovy: fovy,
            resolution: resolution,
            aspect: aspect,
            near: near,
        }
    }
    pub fn update_resolution(&mut self, resolution: Extent2D) {
        self.resolution = resolution;
        self.aspect = (resolution.width as f32) / (resolution.height as f32);
    }
    pub fn resolution(&self) -> vk::Extent2D {
        self.resolution
    }
    pub fn aspect(&self) -> f32 {
        self.aspect
    }
    pub fn lookdir(&self) -> Vec3 {
        let down = Vec4::new(0.0, 0.0, -1.0, 1.0);
        let pitched = Mat4::from_rotation_x(self.pitch) * down;
        let yawed = Mat4::from_rotation_z(self.yaw) * pitched;
        let ret = Vec3::new(yawed.x, yawed.y, yawed.z);
        debug_assert!(ret.is_normalized());
        return ret;
    }
    pub fn view_matrix(&self) -> Mat4 {
        let down_global = Vec3::NEG_Z;
        let forward = self.lookdir();
        let side = down_global.cross(forward).normalize();
        let down = forward.cross(side);

        Mat4::from_cols(
            Vec4::new(side.x, down.x, forward.x, 0.0),
            Vec4::new(side.y, down.y, forward.y, 0.0),
            Vec4::new(side.z, down.z, forward.z, 0.0),
            Vec4::new(
                -self.pos.dot(side),
                -self.pos.dot(down),
                -self.pos.dot(forward),
                1.0,
            ),
        )
    }

    pub fn projection_matrix(&self) -> Mat4 {
        let focal_length = 1.0 / f32::tan(self.fovy * 0.5);
        let x = focal_length / self.aspect;
        let y = focal_length;

        Mat4::from_cols(
            Vec4::new(x, 0.0, 0.0, 0.0),
            Vec4::new(0.0, y, 0.0, 0.0),
            Vec4::new(0.0, 0.0, 0.0, 1.0),
            Vec4::new(0.0, 0.0, self.near, 0.0),
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn reduce1() {
        let mut cbt = CBT::new(7);

        // 0010
        // 0004      0006
        // 0001 0003 0002 0004
        // 0010 0111 1001 1111

        cbt.leaves[0] = AtomicU32::new(0b0010);
        cbt.leaves[1] = AtomicU32::new(0b0111);
        cbt.leaves[2] = AtomicU32::new(0b1001);
        cbt.leaves[3] = AtomicU32::new(0b1111);

        println!("leaves: {:?}", cbt.leaves);

        cbt.reduce();
        assert_eq!(cbt.interior, vec![10, 4, 6, 1, 3, 2, 4]);

        assert_eq!(cbt.one_bit_to_id(0), 32 * 0 + 1);
        assert_eq!(cbt.one_bit_to_id(1), 32 * 1 + 0);
        assert_eq!(cbt.one_bit_to_id(2), 32 * 1 + 1);
        assert_eq!(cbt.one_bit_to_id(3), 32 * 1 + 2);
        assert_eq!(cbt.one_bit_to_id(4), 32 * 2 + 0);
        assert_eq!(cbt.one_bit_to_id(5), 32 * 2 + 3);
        assert_eq!(cbt.one_bit_to_id(6), 32 * 3 + 0);
        assert_eq!(cbt.one_bit_to_id(7), 32 * 3 + 1);
        assert_eq!(cbt.one_bit_to_id(8), 32 * 3 + 2);
        assert_eq!(cbt.one_bit_to_id(9), 32 * 3 + 3);
        assert_eq!(cbt.one_bit_to_id(10), u32::MAX);

        for leaf in cbt.leaves.iter_mut() {
            leaf.fetch_or(0xFFFF_FFF0, Ordering::Relaxed);
        }

        cbt.reduce();

        assert_eq!(cbt.zero_bit_to_id(0), 32 * 0 + 0);
        assert_eq!(cbt.zero_bit_to_id(1), 32 * 0 + 2);
        assert_eq!(cbt.zero_bit_to_id(2), 32 * 0 + 3);

        assert_eq!(cbt.zero_bit_to_id(3), 32 * 1 + 3);

        assert_eq!(cbt.zero_bit_to_id(4), 32 * 2 + 1);
        assert_eq!(cbt.zero_bit_to_id(5), 32 * 2 + 2);
        assert_eq!(cbt.zero_bit_to_id(6), u32::MAX);
    }

    #[test]
    fn reduce2() {
        let mut cbt = CBT::new(8);

        // 12
        // 8                   4
        // 4         4         1         3
        // 3    1    2    2    0    1    2    1
        // 1011 0001 0101 1010 0000 0100 0110 0001
        cbt.leaves[0] = AtomicU32::new(0b1011);
        cbt.leaves[1] = AtomicU32::new(0b0001);
        cbt.leaves[2] = AtomicU32::new(0b0101);
        cbt.leaves[3] = AtomicU32::new(0b1010);
        cbt.leaves[4] = AtomicU32::new(0b0000);
        cbt.leaves[5] = AtomicU32::new(0b0100);
        cbt.leaves[6] = AtomicU32::new(0b0110);
        cbt.leaves[7] = AtomicU32::new(0b0001);

        println!("leaves: {:?}", cbt.leaves);

        cbt.reduce();
        assert_eq!(
            cbt.interior,
            vec![12, 8, 4, 4, 4, 1, 3, 3, 1, 2, 2, 0, 1, 2, 1]
        );

        // 1011
        assert_eq!(cbt.one_bit_to_id(0), 32 * 0 + 0);
        assert_eq!(cbt.one_bit_to_id(1), 32 * 0 + 1);
        assert_eq!(cbt.one_bit_to_id(2), 32 * 0 + 3);
        // 0001
        assert_eq!(cbt.one_bit_to_id(3), 32 * 1 + 0);
        // 0101
        assert_eq!(cbt.one_bit_to_id(4), 32 * 2 + 0);
        assert_eq!(cbt.one_bit_to_id(5), 32 * 2 + 2);
        // 1010
        assert_eq!(cbt.one_bit_to_id(6), 32 * 3 + 1);
        assert_eq!(cbt.one_bit_to_id(7), 32 * 3 + 3);
        // 0000
        // 0100
        assert_eq!(cbt.one_bit_to_id(8), 32 * 5 + 2);
        // 0110
        assert_eq!(cbt.one_bit_to_id(9), 32 * 6 + 1);
        assert_eq!(cbt.one_bit_to_id(10), 32 * 6 + 2);
        // 0001
        assert_eq!(cbt.one_bit_to_id(11), 32 * 7 + 0);
        // end
        assert_eq!(cbt.one_bit_to_id(12), u32::MAX);

        for leaf in cbt.leaves.iter_mut() {
            leaf.fetch_or(0xFFFF_FFF0, Ordering::Relaxed);
        }
        cbt.reduce();
        // 1011
        assert_eq!(cbt.zero_bit_to_id(0), 32 * 0 + 2);
        // 0001
        assert_eq!(cbt.zero_bit_to_id(1), 32 * 1 + 1);
        assert_eq!(cbt.zero_bit_to_id(2), 32 * 1 + 2);
        assert_eq!(cbt.zero_bit_to_id(3), 32 * 1 + 3);
        // 0101
        assert_eq!(cbt.zero_bit_to_id(4), 32 * 2 + 1);
        assert_eq!(cbt.zero_bit_to_id(5), 32 * 2 + 3);
        // 1010
        assert_eq!(cbt.zero_bit_to_id(6), 32 * 3 + 0);
        assert_eq!(cbt.zero_bit_to_id(7), 32 * 3 + 2);
        // 0000
        assert_eq!(cbt.zero_bit_to_id(8), 32 * 4 + 0);
        assert_eq!(cbt.zero_bit_to_id(9), 32 * 4 + 1);
        assert_eq!(cbt.zero_bit_to_id(10), 32 * 4 + 2);
        assert_eq!(cbt.zero_bit_to_id(11), 32 * 4 + 3);
        // 0100
        assert_eq!(cbt.zero_bit_to_id(12), 32 * 5 + 0);
        assert_eq!(cbt.zero_bit_to_id(13), 32 * 5 + 1);
        assert_eq!(cbt.zero_bit_to_id(14), 32 * 5 + 3);
        // 0110
        assert_eq!(cbt.zero_bit_to_id(15), 32 * 6 + 0);
        assert_eq!(cbt.zero_bit_to_id(16), 32 * 6 + 3);
        // 0001
        assert_eq!(cbt.zero_bit_to_id(17), 32 * 7 + 1);
        assert_eq!(cbt.zero_bit_to_id(18), 32 * 7 + 2);
        assert_eq!(cbt.zero_bit_to_id(19), 32 * 7 + 3);
        // end
        assert_eq!(cbt.zero_bit_to_id(20), u32::MAX);
    }

    #[test]
    fn test_iterate1() {
        let halfedge_mesh = HalfedgeMesh {
            verts: vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
            ],
            indices: vec![2, 0, 1, 0, 2, 3],
            faces: vec![
                Face {
                    v0: 0,
                    num_verts: 3,
                },
                Face {
                    v0: 3,
                    num_verts: 3,
                },
            ],
            neighbors: vec![
                [1, 2, 3],
                [2, 0, u32::MAX],
                [0, 1, u32::MAX],
                [4, 5, 0],
                [5, 3, u32::MAX],
                [3, 4, u32::MAX],
            ],
        };
        let mut pipeline_data = PipelineData::new(halfedge_mesh, 8);

        assert_eq!(pipeline_data.cbt.leaves.len(), 8);
        assert_eq!(
            pipeline_data.cbt.leaves[0].load(Ordering::Relaxed),
            0x0000_003F // there are 6 root bisectors
        );
        assert_eq!(pipeline_data.base_depth, 3);
        assert_eq!(pipeline_data.heapid_buffer[0], 0b1000);
        assert_eq!(pipeline_data.heapid_buffer[1], 0b1001);
        assert_eq!(pipeline_data.heapid_buffer[2], 0b1010);
        assert_eq!(pipeline_data.heapid_buffer[3], 0b1011);
        assert_eq!(pipeline_data.heapid_buffer[4], 0b1100);
        assert_eq!(pipeline_data.heapid_buffer[5], 0b1101);

        // let's try to split the bisector 1
        pipeline_data
            .want_split_buffer_count
            .fetch_add(1, Ordering::Relaxed);
        pipeline_data.want_split_buffer[0] = 1; // bisector 1 wants to split

        pipeline_data.iterate();

        println!(
            "bitfield: {:b}",
            pipeline_data.cbt.leaves[0].load(Ordering::Relaxed)
        );
        assert_eq!(
            pipeline_data.allocation_indices_buffer[1],
            [6, 7, u32::MAX, u32::MAX]
        );
        assert_eq!(pipeline_data.cbt.interior[0], 7); // we added 1 more bisector
        assert_eq!(pipeline_data.heapid_buffer[6], 0b1001_1);
        assert_eq!(pipeline_data.heapid_buffer[7], 0b1001_0);
        assert_eq!(pipeline_data.neighbors_buffer[0], [7, 2, 3]);
        assert_eq!(pipeline_data.neighbors_buffer[2], [0, 6, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[3], [4, 5, 0]);
        assert_eq!(pipeline_data.neighbors_buffer[4], [5, 3, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[5], [3, 4, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[6], [7, u32::MAX, 2]);
        assert_eq!(pipeline_data.neighbors_buffer[7], [u32::MAX, 6, 0]);

        println!("Segment 1 completed\n");
        pipeline_data.reset();

        pipeline_data
            .want_split_buffer_count
            .fetch_add(1, Ordering::Relaxed);
        pipeline_data.want_split_buffer[0] = 7; // bisector 6 wants to split now

        pipeline_data.iterate();

        assert_eq!(pipeline_data.cbt.interior[0], 11);
        assert_eq!(pipeline_data.heapid_buffer[1], 0b1001_01);
        assert_eq!(pipeline_data.heapid_buffer[8], 0b1001_00);
        assert_eq!(pipeline_data.heapid_buffer[9], 0b1000_11);
        assert_eq!(pipeline_data.heapid_buffer[10], 0b1000_0);
        assert_eq!(pipeline_data.heapid_buffer[11], 0b1000_10);
        assert_eq!(pipeline_data.heapid_buffer[12], 0b1011_1);
        assert_eq!(pipeline_data.heapid_buffer[13], 0b1011_0);

        assert_eq!(pipeline_data.neighbors_buffer[2], [10, 6, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[4], [5, 12, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[5], [13, 4, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[6], [8, u32::MAX, 2]);
        assert_eq!(pipeline_data.neighbors_buffer[1], [8, 11, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[8], [9, 1, 6]);
        assert_eq!(pipeline_data.neighbors_buffer[9], [11, 8, 10]);
        assert_eq!(pipeline_data.neighbors_buffer[10], [12, 9, 2]);
        assert_eq!(pipeline_data.neighbors_buffer[11], [1, 9, 13]);
        assert_eq!(pipeline_data.neighbors_buffer[12], [13, 10, 4]);
        assert_eq!(pipeline_data.neighbors_buffer[13], [11, 12, 5]);

        println!("Segment 2 completed\n");
        pipeline_data.reset();

        pipeline_data.bisector_state_buffer[1] = SIMPLIFY;
        pipeline_data.bisector_state_buffer[8] = SIMPLIFY;
        pipeline_data.bisector_state_buffer[9] = SIMPLIFY;
        pipeline_data.bisector_state_buffer[11] = SIMPLIFY;

        pipeline_data
            .want_merge_buffer_count
            .fetch_add(2, Ordering::Relaxed);
        pipeline_data.want_merge_buffer[0] = 1;
        pipeline_data.want_merge_buffer[1] = 9;

        pipeline_data.iterate();

        assert_eq!(pipeline_data.cbt.interior[0], 9);
        assert_eq!(pipeline_data.heapid_buffer[1], 0b1001_0);
        assert_eq!(pipeline_data.heapid_buffer[9], 0b1000_1);

        assert_eq!(pipeline_data.neighbors_buffer[1], [u32::MAX, 6, 9]);
        assert_eq!(pipeline_data.neighbors_buffer[6], [1, u32::MAX, 2]);
        assert_eq!(pipeline_data.neighbors_buffer[9], [10, 13, 1]);
        assert_eq!(pipeline_data.neighbors_buffer[10], [12, 9, 2]);
        assert_eq!(pipeline_data.neighbors_buffer[13], [9, 12, 5]);

        println!("Segment 3 completed\n");

        pipeline_data.reset();

        pipeline_data.bisector_state_buffer[1] = SIMPLIFY;
        pipeline_data.bisector_state_buffer[6] = SIMPLIFY;

        pipeline_data
            .want_merge_buffer_count
            .fetch_add(1, Ordering::Relaxed);
        pipeline_data.want_merge_buffer[0] = 6; // enqueue ODD heapids for simplification

        pipeline_data.iterate();
        assert_eq!(pipeline_data.cbt.interior[0], 8);
        assert_eq!(pipeline_data.heapid_buffer[6], 0b1001_);

        assert_eq!(pipeline_data.neighbors_buffer[6], [2, 9, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[2], [10, 6, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[9], [10, 13, 6]);
    }

    #[test]
    fn test_iterate2() {
        let halfedge_mesh = HalfedgeMesh {
            verts: vec![
                Vec3::new(1.0, 0.0, 0.0),
                Vec3::new(0.0, 1.0, 0.0),
                Vec3::new(-1.0, 0.0, 0.0),
                Vec3::new(0.0, -1.0, 0.0),
            ],
            indices: vec![2, 0, 1, 0, 2, 3],
            faces: vec![
                Face {
                    v0: 0,
                    num_verts: 3,
                },
                Face {
                    v0: 3,
                    num_verts: 3,
                },
            ],
            neighbors: vec![
                [1, 2, 3],
                [2, 0, u32::MAX],
                [0, 1, u32::MAX],
                [4, 5, 0],
                [5, 3, u32::MAX],
                [3, 4, u32::MAX],
            ],
        };
        let mut pipeline_data = PipelineData::new(halfedge_mesh, 8);

        assert_eq!(pipeline_data.cbt.leaves.len(), 8);
        assert_eq!(
            pipeline_data.cbt.leaves[0].load(Ordering::Relaxed),
            0x0000_003F // there are 6 root bisectors
        );
        assert_eq!(pipeline_data.base_depth, 3);
        assert_eq!(pipeline_data.heapid_buffer[0], 0b1000);
        assert_eq!(pipeline_data.heapid_buffer[1], 0b1001);
        assert_eq!(pipeline_data.heapid_buffer[2], 0b1010);
        assert_eq!(pipeline_data.heapid_buffer[3], 0b1011);
        assert_eq!(pipeline_data.heapid_buffer[4], 0b1100);
        assert_eq!(pipeline_data.heapid_buffer[5], 0b1101);

        // let's try to split the bisector 1
        pipeline_data
            .want_split_buffer_count
            .fetch_add(1, Ordering::Relaxed);
        pipeline_data.want_split_buffer[0] = 1; // bisector 1 wants to split

        pipeline_data.iterate();

        println!(
            "bitfield: {:b}",
            pipeline_data.cbt.leaves[0].load(Ordering::Relaxed)
        );
        assert_eq!(
            pipeline_data.allocation_indices_buffer[1],
            [6, 7, u32::MAX, u32::MAX]
        );
        assert_eq!(pipeline_data.cbt.interior[0], 7); // we added 1 more bisector
        assert_eq!(pipeline_data.heapid_buffer[6], 0b1001_1);
        assert_eq!(pipeline_data.heapid_buffer[7], 0b1001_0);
        assert_eq!(pipeline_data.neighbors_buffer[0], [7, 2, 3]);
        assert_eq!(pipeline_data.neighbors_buffer[2], [0, 6, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[3], [4, 5, 0]);
        assert_eq!(pipeline_data.neighbors_buffer[4], [5, 3, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[5], [3, 4, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[6], [7, u32::MAX, 2]);
        assert_eq!(pipeline_data.neighbors_buffer[7], [u32::MAX, 6, 0]);

        println!("Segment 1 completed\n");
        pipeline_data.reset();

        pipeline_data
            .want_split_buffer_count
            .fetch_add(1, Ordering::Relaxed);
        pipeline_data.want_split_buffer[0] = 7; // bisector 6 wants to split now

        pipeline_data.iterate();

        assert_eq!(pipeline_data.cbt.interior[0], 11);
        assert_eq!(pipeline_data.heapid_buffer[1], 0b1001_01);
        assert_eq!(pipeline_data.heapid_buffer[8], 0b1001_00);
        assert_eq!(pipeline_data.heapid_buffer[9], 0b1000_11);
        assert_eq!(pipeline_data.heapid_buffer[10], 0b1000_0);
        assert_eq!(pipeline_data.heapid_buffer[11], 0b1000_10);
        assert_eq!(pipeline_data.heapid_buffer[12], 0b1011_1);
        assert_eq!(pipeline_data.heapid_buffer[13], 0b1011_0);

        assert_eq!(pipeline_data.neighbors_buffer[2], [10, 6, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[4], [5, 12, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[5], [13, 4, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[6], [8, u32::MAX, 2]);
        assert_eq!(pipeline_data.neighbors_buffer[1], [8, 11, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[8], [9, 1, 6]);
        assert_eq!(pipeline_data.neighbors_buffer[9], [11, 8, 10]);
        assert_eq!(pipeline_data.neighbors_buffer[10], [12, 9, 2]);
        assert_eq!(pipeline_data.neighbors_buffer[11], [1, 9, 13]);
        assert_eq!(pipeline_data.neighbors_buffer[12], [13, 10, 4]);
        assert_eq!(pipeline_data.neighbors_buffer[13], [11, 12, 5]);

        println!("Segment 2 completed\n");
        pipeline_data.reset();
        pipeline_data
            .want_split_buffer_count
            .fetch_add(1, Ordering::Relaxed);
        pipeline_data.want_split_buffer[0] = 9;
        pipeline_data.iterate();

        assert_eq!(pipeline_data.neighbors_buffer[0], [3, 15, 11]);
        assert_eq!(pipeline_data.neighbors_buffer[1], [8, 11, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[3], [14, 0, 8]);
        assert_eq!(pipeline_data.neighbors_buffer[4], [5, 12, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[5], [13, 4, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[6], [8, u32::MAX, 17]);
        assert_eq!(pipeline_data.neighbors_buffer[7], [15, 18, 12]);
        assert_eq!(pipeline_data.neighbors_buffer[8], [3, 1, 6]);
        assert_eq!(pipeline_data.neighbors_buffer[11], [1, 0, 13]);
        assert_eq!(pipeline_data.neighbors_buffer[12], [13, 7, 4]);
        assert_eq!(pipeline_data.neighbors_buffer[13], [11, 12, 5]);
        assert_eq!(pipeline_data.neighbors_buffer[14], [15, 3, 16]);
        assert_eq!(pipeline_data.neighbors_buffer[15], [0, 14, 7]);
        assert_eq!(pipeline_data.neighbors_buffer[16], [18, 14, 17]);
        assert_eq!(pipeline_data.neighbors_buffer[17], [u32::MAX, 16, 6]);
        assert_eq!(pipeline_data.neighbors_buffer[18], [7, 16, u32::MAX]);

        println!("Segment 3 completed\n");

        pipeline_data.reset();
        pipeline_data
            .want_split_buffer_count
            .fetch_add(1, Ordering::Relaxed);
        pipeline_data.want_split_buffer[0] = 14;
        pipeline_data.iterate();

        assert_eq!(pipeline_data.neighbors_buffer[2], [9, 20, 15]);
        assert_eq!(pipeline_data.neighbors_buffer[9], [19, 2, 3]);
        assert_eq!(pipeline_data.neighbors_buffer[10], [20, 23, 18]);
        assert_eq!(pipeline_data.neighbors_buffer[15], [0, 2, 7]);
        assert_eq!(pipeline_data.neighbors_buffer[18], [7, 10, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[19], [20, 9, 22]);
        assert_eq!(pipeline_data.neighbors_buffer[20], [2, 19, 10]);
        assert_eq!(pipeline_data.neighbors_buffer[21], [23, 25, u32::MAX]);
        assert_eq!(pipeline_data.neighbors_buffer[22], [23, 19, 24]);
        assert_eq!(pipeline_data.neighbors_buffer[23], [10, 22, 21]);
        assert_eq!(pipeline_data.neighbors_buffer[24], [25, 22, 8]);
        assert_eq!(pipeline_data.neighbors_buffer[25], [21, 24, u32::MAX]);
    }
}
