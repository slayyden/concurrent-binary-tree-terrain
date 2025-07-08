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
