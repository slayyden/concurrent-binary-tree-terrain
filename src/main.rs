// use ash::Instance;
use ash::{
    Device, Entry,
    ext::{custom_border_color, debug_utils},
    khr::{surface, swapchain},
    util::{Align, read_spv},
    vk::{self, PipelineCreateFlags, PipelineShaderStageCreateFlags, ShaderStageFlags},
};
use dirt_jam::*;
use glam::{Mat4, Vec3};
use std::sync::atomic::{AtomicU32, Ordering};
use std::{
    cmp::{max, min},
    error::Error,
    ffi::CStr,
    io::Cursor,
    os::raw::c_char,
    u64,
};
use std::{f32::consts::PI, ffi::c_void};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::Key,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

#[repr(C)]
struct Vertex {
    position: [f32; 3],
    color: [f32; 3],
}

#[repr(C)]
struct PushConstants {
    view_project: Mat4,
    positions: vk::DeviceAddress,
    curr_ids: vk::DeviceAddress,

    scene: vk::DeviceAddress,
    dispatch: vk::DeviceAddress,
}

struct AtomicTest {
    test: AtomicU32,
}

struct State {
    window: Window,
    instance: ash::Instance,
    device: ash::Device,
    setup_commands_reuse_fence: vk::Fence,
    setup_command_buffer: vk::CommandBuffer,
    draw_command_buffers: [vk::CommandBuffer; 3],
    present_queue: vk::Queue,
    pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    // the we can get any number of images above the minimum, length is unknown at comptime
    present_image_views: Vec<vk::ImageView>,
    present_images: Vec<vk::Image>,
    depth_image_view: vk::ImageView,
    swapchain: vk::SwapchainKHR,
    swapchain_loader: swapchain::Device,

    frame_index: u64,
    present_complete_semaphore: [vk::Semaphore; 3],
    render_complete_semaphore: [vk::Semaphore; 3],
    frame_pace_semaphore: vk::Semaphore,

    resolution: vk::Extent2D,
    command_reuse_fences: [vk::Fence; 3],

    vertex_buffer: AllocatedBuffer,
    curr_ids_buffer: AllocatedBuffer,
    curr_ids_staging_buffer: AllocatedBuffer,
    curr_ids_staging_buffer_ptr: *mut c_void,

    vertex_staging_buffer: AllocatedBuffer,
    vertex_staging_buffer_ptr: *mut c_void,
    algorithm_data: PipelineData,
    iteration_counter: u32,

    scene_buffer_handles: SceneCPUHandles,
    scene_buffer: AllocatedBuffer,
    dispatch_buffer: AllocatedBuffer,

    camera: CameraState,
}

impl State {
    pub fn render(&mut self) {
        unsafe {
            let dev = &self.device;

            let viewport = vk::Viewport {
                x: 0.0,
                y: 0.0,
                width: self.resolution.width as f32,
                height: self.resolution.height as f32,
                min_depth: 0.0,
                max_depth: 1.0,
            };
            let scissor = vk::Rect2D::default().extent(self.resolution);

            let semaphore_index = (self.frame_index % 3) as usize;
            let frame_fence_index = (self.frame_index % 3) as usize;
            let frame_cmdbuf_index = (self.frame_index % 3) as usize;

            let cmdbuf = &self.draw_command_buffers[frame_cmdbuf_index];

            dev.wait_for_fences(
                &[self.command_reuse_fences[frame_fence_index]],
                true,
                u64::MAX,
            )
            .expect("Wait for fences.");
            dev.reset_fences(&[self.command_reuse_fences[frame_fence_index]])
                .expect("Reset fences.");

            let (present_index, _) = self
                .swapchain_loader
                .acquire_next_image(
                    self.swapchain,
                    u64::MAX,
                    self.present_complete_semaphore[semaphore_index],
                    vk::Fence::null(),
                )
                .unwrap();

            let color_attachments = [vk::RenderingAttachmentInfo::default()
                .clear_value(vk::ClearValue {
                    color: vk::ClearColorValue {
                        float32: [0.0, 0.0, 0.0, 1.0] as [f32; 4],
                    },
                })
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image_view(self.present_image_views[present_index as usize])];

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.depth_image_view)
                .image_layout(vk::ImageLayout::DEPTH_ATTACHMENT_OPTIMAL)
                .clear_value(vk::ClearValue {
                    depth_stencil: vk::ClearDepthStencilValue {
                        depth: 1.0,
                        stencil: 0,
                    },
                });

            let rendering_info = vk::RenderingInfo::default()
                .color_attachments(&color_attachments)
                .depth_attachment(&depth_attachment)
                .layer_count(1)
                .render_area(vk::Rect2D::default().extent(self.resolution));

            // command buffer and rendering
            dev.begin_command_buffer(*cmdbuf, &vk::CommandBufferBeginInfo::default())
                .expect("Could not begin command buffer.");

            let timeline_semaphores = [self.frame_pace_semaphore];
            let timeline_semaphore_wait_values = [self.frame_index];
            dev.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(&timeline_semaphores)
                    .values(&timeline_semaphore_wait_values),
                u64::MAX,
            )
            .expect("Timeline semaphore wait");

            // transitioning
            let pre_barrier = vk::ImageMemoryBarrier::default()
                .image(self.present_images[present_index as usize])
                .dst_access_mask(vk::AccessFlags::COLOR_ATTACHMENT_WRITE)
                .old_layout(vk::ImageLayout::UNDEFINED)
                .new_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS)
                        .level_count(vk::REMAINING_MIP_LEVELS),
                );
            dev.cmd_pipeline_barrier(
                *cmdbuf,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[pre_barrier],
            );

            dev.cmd_begin_rendering(*cmdbuf, &rendering_info);

            let push_constants = PushConstants {
                view_project: Mat4::perspective_rh(PI / 2.0, 16.0 / 9.0, 0.01, 100.0)
                    * self.camera.view_matrix(),
                positions: dev.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(self.vertex_buffer.buffer),
                ),
                curr_ids: dev.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(self.curr_ids_buffer.buffer),
                ),

                scene: dev.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(self.scene_buffer.buffer),
                ),
                dispatch: dev.get_buffer_device_address(
                    &vk::BufferDeviceAddressInfo::default().buffer(self.dispatch_buffer.buffer),
                ),
            };

            // push constants
            dev.cmd_push_constants(
                *cmdbuf,
                self.pipeline_layout,
                vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
                0,
                byteslice(&push_constants),
            );

            // RENDERING CORE
            dev.cmd_set_viewport(*cmdbuf, 0, &[viewport]);
            dev.cmd_set_scissor(*cmdbuf, 0, &[scissor]);
            dev.cmd_bind_pipeline(*cmdbuf, vk::PipelineBindPoint::GRAPHICS, self.pipeline);
            dev.cmd_clear_attachments(
                *cmdbuf,
                &[
                    vk::ClearAttachment::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .clear_value(vk::ClearValue {
                            color: vk::ClearColorValue {
                                float32: [0.0, 0.0, 0.0, 1.0],
                            },
                        }),
                    vk::ClearAttachment::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .clear_value(vk::ClearValue {
                            depth_stencil: vk::ClearDepthStencilValue {
                                depth: 1.0,
                                stencil: 0,
                            },
                        }),
                ],
                &[
                    vk::ClearRect::default()
                        .rect(vk::Rect2D::default().extent(self.resolution))
                        .base_array_layer(0)
                        .layer_count(1),
                    vk::ClearRect::default()
                        .rect(vk::Rect2D::default().extent(self.resolution))
                        .base_array_layer(0)
                        .layer_count(1),
                ],
            );
            dev.cmd_draw(*cmdbuf, self.algorithm_data.cbt.interior[0] * 3, 1, 0, 0);

            dev.cmd_end_rendering(*cmdbuf);

            // transitioning out
            let post_barrier = vk::ImageMemoryBarrier::default()
                .image(self.present_images[present_index as usize])
                .src_access_mask(
                    vk::AccessFlags::COLOR_ATTACHMENT_READ
                        | vk::AccessFlags::COLOR_ATTACHMENT_WRITE,
                )
                .old_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .new_layout(vk::ImageLayout::PRESENT_SRC_KHR)
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::COLOR)
                        .layer_count(vk::REMAINING_ARRAY_LAYERS)
                        .level_count(vk::REMAINING_MIP_LEVELS),
                );

            dev.cmd_pipeline_barrier(
                *cmdbuf,
                vk::PipelineStageFlags::COLOR_ATTACHMENT_OUTPUT,
                vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                vk::DependencyFlags::empty(),
                &[],
                &[],
                &[post_barrier],
            );

            // done!
            dev.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(self.frame_pace_semaphore)
                    .value(self.frame_index + 1),
            )
            .expect("Timeline semaphore signal");
            dev.end_command_buffer(*cmdbuf)
                .expect("End command buffer.");

            {
                let command_buffers = vec![*cmdbuf];
                let wait_semaphores = [self.present_complete_semaphore[semaphore_index]];
                let signal_semaphores = [self.render_complete_semaphore[semaphore_index]];

                let submit_info = vk::SubmitInfo::default()
                    .wait_semaphores(&wait_semaphores)
                    .wait_dst_stage_mask(&[vk::PipelineStageFlags::ALL_GRAPHICS])
                    .command_buffers(&command_buffers)
                    .signal_semaphores(&signal_semaphores);

                dev.queue_submit(
                    self.present_queue,
                    &[submit_info],
                    self.command_reuse_fences[frame_fence_index],
                )
                .expect("Queue Submit");
            }

            {
                let wait_semaphores = [self.render_complete_semaphore[semaphore_index]];
                let swapchain = [self.swapchain];
                let image_indices = [present_index];
                let present_info = vk::PresentInfoKHR::default()
                    .wait_semaphores(&wait_semaphores)
                    .swapchains(&swapchain)
                    .image_indices(&image_indices);
                self.swapchain_loader
                    .queue_present(self.present_queue, &present_info)
                    .unwrap();
            }

            // WARNING: WE CANNOT RETURN EARLY BECAUSE OF THIS
            self.frame_index += 1;
        }
    }
}

struct App {
    state: Option<State>,
}
impl App {
    pub fn new() -> Self {
        Self { state: None }
    }
}

impl ApplicationHandler for App {
    fn resumed(&mut self, event_loop: &ActiveEventLoop) {
        println!("resumed");
        // create window
        let window = event_loop
            .create_window(Window::default_attributes())
            .unwrap();

        // --------------------------------------------------------------------
        // initialize vulkan
        let entry = Entry::linked();

        // get required extensions to display to surface
        let mut extension_names =
            ash_window::enumerate_required_extensions(window.display_handle().unwrap().as_raw())
                .unwrap()
                .to_vec();
        extension_names.push(debug_utils::NAME.as_ptr());
        // metal-specific extensions
        #[cfg(any(target_os = "macos", target_os = "ios"))]
        {
            extension_names.push(ash::khr::portability_enumeration::NAME.as_ptr());
            extension_names.push(ash::khr::get_physical_device_properties2::NAME.as_ptr());
        }

        let layer_names = [c"VK_LAYER_KHRONOS_validation"];
        let layer_names_raw: Vec<*const c_char> = layer_names
            .iter()
            .map(|raw_name| raw_name.as_ptr())
            .collect();

        // flags for creating the vulkan instance
        let create_flags = if cfg!(any(target_os = "macos", target_os = "ios")) {
            vk::InstanceCreateFlags::ENUMERATE_PORTABILITY_KHR
        } else {
            vk::InstanceCreateFlags::default()
        };

        let application_info = vk::ApplicationInfo::default().api_version(vk::API_VERSION_1_3);

        let create_info = vk::InstanceCreateInfo::default()
            .enabled_extension_names(&extension_names)
            .enabled_layer_names(&layer_names_raw)
            .flags(create_flags)
            .application_info(&application_info);

        // create vulkan instance
        unsafe {
            let instance = entry
                .create_instance(&create_info, None)
                .expect("Instance Creation Error");

            let surface = ash_window::create_surface(
                &entry,
                &instance,
                window.display_handle().unwrap().as_raw(),
                window.window_handle().unwrap().as_raw(),
                None,
            )
            .unwrap();

            let pdevices = instance
                .enumerate_physical_devices()
                .expect("Physical Device Error");
            let surface_loader = surface::Instance::new(&entry, &instance);
            println!("Num physical devices: {:?}", pdevices.len());
            // get physical device and queue family
            let (pdevice, queue_family_index) = pdevices
                .iter()
                .find_map(|pdevice| {
                    instance
                        .get_physical_device_queue_family_properties(*pdevice)
                        .iter()
                        .enumerate()
                        .find_map(|(i, queue_family_info)| {
                            let supports_graphics = queue_family_info
                                .queue_flags
                                .contains(vk::QueueFlags::GRAPHICS);
                            let supports_surface = surface_loader
                                .get_physical_device_surface_support(*pdevice, i as u32, surface)
                                .unwrap();
                            if supports_graphics && supports_surface {
                                Some((*pdevice, i as u32))
                            } else {
                                None
                            }
                        })
                })
                .expect("Couldn't find suitable device.");

            let mut device_properties = vk::PhysicalDeviceProperties2::default();
            instance.get_physical_device_properties2(pdevice, &mut device_properties);
            println!(
                "Version: {:?}",
                vk::api_version_minor(device_properties.properties.api_version)
            );
            let device_extension_names_raw = [
                swapchain::NAME.as_ptr(),
                #[cfg(any(target_os = "macos", target_os = "ios"))]
                ash::khr::portability_subset::NAME.as_ptr(),
            ];

            let priorities = [1.0];
            let queue_info = vk::DeviceQueueCreateInfo::default()
                .queue_family_index(queue_family_index)
                .queue_priorities(&priorities);
            let mut vulkan_11_features =
                vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
            let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .timeline_semaphore(true)
                .scalar_block_layout(true);
            let mut vulkan_13_features =
                vk::PhysicalDeviceVulkan13Features::default().dynamic_rendering(true);
            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names_raw)
                .push_next(&mut vulkan_11_features)
                .push_next(&mut vulkan_12_features)
                .push_next(&mut vulkan_13_features);

            let device: ash::Device = instance
                .create_device(pdevice, &device_create_info, None)
                .unwrap();

            let present_queue = device.get_device_queue(queue_family_index, 0);

            let surface_format = surface_loader
                .get_physical_device_surface_formats(pdevice, surface)
                .unwrap()[0];

            let surface_capabilities = surface_loader
                .get_physical_device_surface_capabilities(pdevice, surface)
                .unwrap();

            // triple buffer if we can
            let desired_image_count = max(3, surface_capabilities.min_image_count);

            // TODO: move elsewhere
            let window_width = 1920;
            let window_height = 1080;
            let surface_resolution = match surface_capabilities.current_extent.width {
                u32::MAX => vk::Extent2D {
                    width: window_width,
                    height: window_height,
                },
                _ => surface_capabilities.current_extent,
            };

            let present_modes = surface_loader
                .get_physical_device_surface_present_modes(pdevice, surface)
                .unwrap();
            let present_mode: vk::PresentModeKHR = present_modes
                .into_iter()
                .find(|&mode| mode == vk::PresentModeKHR::MAILBOX)
                .unwrap_or(vk::PresentModeKHR::FIFO);

            let swapchain_loader = swapchain::Device::new(&instance, &device);

            let swapchain_create_info = vk::SwapchainCreateInfoKHR::default()
                .surface(surface)
                .min_image_count(desired_image_count)
                .image_color_space(surface_format.color_space)
                .image_format(surface_format.format)
                .image_extent(surface_resolution)
                .image_usage(vk::ImageUsageFlags::COLOR_ATTACHMENT)
                .image_sharing_mode(vk::SharingMode::EXCLUSIVE)
                .pre_transform(vk::SurfaceTransformFlagsKHR::IDENTITY)
                .composite_alpha(vk::CompositeAlphaFlagsKHR::OPAQUE)
                .present_mode(present_mode)
                .clipped(true)
                .image_array_layers(1);

            let swapchain = swapchain_loader
                .create_swapchain(&swapchain_create_info, None)
                .unwrap();

            let pool_create_info = vk::CommandPoolCreateInfo::default()
                .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER)
                .queue_family_index(queue_family_index);

            let pool = device.create_command_pool(&pool_create_info, None).unwrap();

            let command_buffer_allocate_info = vk::CommandBufferAllocateInfo::default()
                .command_buffer_count(4)
                .command_pool(pool)
                .level(vk::CommandBufferLevel::PRIMARY);

            let command_buffers = device
                .allocate_command_buffers(&command_buffer_allocate_info)
                .unwrap();

            let setup_command_buffer = command_buffers[0];
            let draw_command_buffers = [command_buffers[1], command_buffers[2], command_buffers[3]];

            let present_images = swapchain_loader.get_swapchain_images(swapchain).unwrap();
            let present_image_views: Vec<vk::ImageView> = present_images
                .iter()
                .map(|&image| {
                    let create_view_info = vk::ImageViewCreateInfo::default()
                        .view_type(vk::ImageViewType::TYPE_2D)
                        .format(surface_format.format)
                        .components(vk::ComponentMapping {
                            r: vk::ComponentSwizzle::R,
                            g: vk::ComponentSwizzle::G,
                            b: vk::ComponentSwizzle::B,
                            a: vk::ComponentSwizzle::A,
                        })
                        .subresource_range(vk::ImageSubresourceRange {
                            aspect_mask: vk::ImageAspectFlags::COLOR,
                            base_mip_level: 0,
                            level_count: 1,
                            base_array_layer: 0,
                            layer_count: 1,
                        })
                        .image(image);
                    device.create_image_view(&create_view_info, None).unwrap()
                })
                .collect();

            let device_memory_properties = instance.get_physical_device_memory_properties(pdevice);
            let depth_format = vk::Format::D16_UNORM;
            let depth_image_create_info = vk::ImageCreateInfo::default()
                .image_type(vk::ImageType::TYPE_2D)
                .format(depth_format)
                .extent(surface_resolution.into())
                .mip_levels(1)
                .array_layers(1)
                .samples(vk::SampleCountFlags::TYPE_1)
                .tiling(vk::ImageTiling::OPTIMAL)
                .usage(vk::ImageUsageFlags::DEPTH_STENCIL_ATTACHMENT)
                .sharing_mode(vk::SharingMode::EXCLUSIVE);

            let depth_image = device.create_image(&depth_image_create_info, None).unwrap();
            let depth_image_memory_req = device.get_image_memory_requirements(depth_image);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index);

            let depth_image_memory = device
                .allocate_memory(&depth_image_allocate_info, None)
                .unwrap();

            device
                .bind_image_memory(depth_image, depth_image_memory, 0)
                .expect("Unable to bind depth image memory.");

            let fence_signalled_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);
            let fence_unsignalled_info = vk::FenceCreateInfo::default();

            let command_reuse_fences = [
                device.create_fence(&fence_signalled_info, None).unwrap(),
                device.create_fence(&fence_signalled_info, None).unwrap(),
                device.create_fence(&fence_signalled_info, None).unwrap(),
            ];

            let setup_commands_reuse_fence = device
                .create_fence(&fence_signalled_info, None)
                .expect("Create fence failed.");

            // ----------------------------------------------------------------
            // semaphores and fences
            let semaphore_create_info = vk::SemaphoreCreateInfo::default();

            let present_complete_semaphore: [vk::Semaphore; 3] = std::array::from_fn(|_| {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            });

            let render_complete_semaphore: [vk::Semaphore; 3] = std::array::from_fn(|_| {
                device
                    .create_semaphore(&semaphore_create_info, None)
                    .unwrap()
            });

            let mut frame_pace_semaphore_info = vk::SemaphoreTypeCreateInfo::default()
                .semaphore_type(vk::SemaphoreType::TIMELINE)
                .initial_value(0);
            let frame_pace_semaphore = device
                .create_semaphore(
                    &vk::SemaphoreCreateInfo::default().push_next(&mut frame_pace_semaphore_info),
                    None,
                )
                .unwrap();

            // ----------------------------------------------------------------
            // create buffers

            let halfedge_mesh = HalfedgeMesh {
                verts: vec![
                    Vec3::new(1.0, 0.0, 1.0),
                    Vec3::new(0.0, 1.0, 1.0),
                    Vec3::new(-1.0, 0.0, 1.0),
                    Vec3::new(0.0, -1.0, 1.0),
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
            let algorithm_data = PipelineData::new(halfedge_mesh, 17);
            debug_assert!(algorithm_data.cbt.leaves.len() == 4096);

            let vertex_buffer_size =
                (size_of::<[Vec3; 3]>() * algorithm_data.vertex_buffer.len()) as u64;

            let vertex_staging_buffer = AllocatedBuffer::new(
                &device,
                vertex_buffer_size,
                device_memory_properties,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            );

            let vertex_staging_buffer_ptr = device
                .map_memory(
                    vertex_staging_buffer.allocation,
                    0,
                    vertex_buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map device memory.");

            let mut staging_buffer_slice = Align::new(
                vertex_staging_buffer_ptr,
                align_of::<f32>() as u64,
                vertex_buffer_size,
            );
            staging_buffer_slice.copy_from_slice(algorithm_data.vertex_buffer.as_slice());
            // device.unmap_memory(vertex_staging_buffer.allocation);

            let vertex_buffer = AllocatedBuffer::new(
                &device,
                vertex_buffer_size,
                device_memory_properties,
                vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            let curr_ids_buffer_size = (size_of::<u32>() * 64) as u64;

            let curr_ids_staging_buffer = AllocatedBuffer::new(
                &device,
                curr_ids_buffer_size,
                device_memory_properties,
                vk::BufferUsageFlags::TRANSFER_SRC,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::HOST_VISIBLE,
            );

            let curr_ids_staging_buffer_ptr = device
                .map_memory(
                    curr_ids_staging_buffer.allocation,
                    0,
                    curr_ids_buffer_size,
                    vk::MemoryMapFlags::empty(),
                )
                .expect("Failed to map device memory.");

            let mut curr_ids_staging_buffer_slice = Align::new(
                curr_ids_staging_buffer_ptr,
                align_of::<u32>() as u64,
                curr_ids_buffer_size,
            );
            let initial_staging_buffer: Vec<u32> = (0..algorithm_data.root_bisector_vertices.len())
                .map(|x| x as u32)
                .collect();
            curr_ids_staging_buffer_slice.copy_from_slice(initial_staging_buffer.as_slice());

            let curr_ids_buffer = AllocatedBuffer::new(
                &device,
                curr_ids_buffer_size,
                device_memory_properties,
                vk::BufferUsageFlags::TRANSFER_DST
                    | vk::BufferUsageFlags::STORAGE_BUFFER
                    | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS,
                vk::SharingMode::EXCLUSIVE,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            );

            // initialization command buffer
            record_submit_commandbuffer(
                &device,
                setup_command_buffer,
                setup_commands_reuse_fence,
                present_queue,
                &[],
                &[],
                &[],
                |device, setup_command_buffer| {
                    let layout_transition_barriers = vk::ImageMemoryBarrier::default()
                        .image(depth_image)
                        .dst_access_mask(
                            vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_READ
                                | vk::AccessFlags::DEPTH_STENCIL_ATTACHMENT_WRITE,
                        )
                        .new_layout(vk::ImageLayout::DEPTH_STENCIL_ATTACHMENT_OPTIMAL)
                        .old_layout(vk::ImageLayout::UNDEFINED)
                        .subresource_range(
                            vk::ImageSubresourceRange::default()
                                .aspect_mask(vk::ImageAspectFlags::DEPTH)
                                .layer_count(1)
                                .level_count(1),
                        );

                    device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        &[layout_transition_barriers],
                    );

                    let vertex_copy = vk::BufferCopy::default().size(vertex_buffer_size);
                    device.cmd_copy_buffer(
                        setup_command_buffer,
                        vertex_staging_buffer.buffer,
                        vertex_buffer.buffer,
                        &[vertex_copy],
                    );

                    let ids_copy = vk::BufferCopy::default().size(curr_ids_buffer_size);
                    device.cmd_copy_buffer(
                        setup_command_buffer,
                        curr_ids_staging_buffer.buffer,
                        curr_ids_buffer.buffer,
                        &[ids_copy],
                    );
                },
            );

            device.device_wait_idle().expect("Wait idle.");
            // vertex_staging_buffer.destroy(&device);

            let depth_image_view_info = vk::ImageViewCreateInfo::default()
                .subresource_range(
                    vk::ImageSubresourceRange::default()
                        .aspect_mask(vk::ImageAspectFlags::DEPTH)
                        .level_count(1)
                        .layer_count(1),
                )
                .view_type(vk::ImageViewType::TYPE_2D)
                .format(depth_format)
                .image(depth_image);

            let depth_image_view = device
                .create_image_view(&depth_image_view_info, None)
                .unwrap();

            // ----------------------------------------------------------------
            // create rendering pipeline

            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            // let blend_attachment_state = vk::PipelineColorBlendAttachmentState::default();
            let color_attachment_state = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(&color_attachment_state);

            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_compare_op(vk::CompareOp::LESS); // TODO: REVERT
            let viewport_state = vk::PipelineViewportStateCreateInfo::default()
                .viewport_count(1)
                .scissor_count(1);
            let multisample_state = vk::PipelineMultisampleStateCreateInfo::default()
                .rasterization_samples(vk::SampleCountFlags::TYPE_1);
            let dynamic_state = vk::PipelineDynamicStateCreateInfo::default()
                .dynamic_states(&[vk::DynamicState::VIEWPORT, vk::DynamicState::SCISSOR]);
            let vertex_input_state = vk::PipelineVertexInputStateCreateInfo::default();

            let mut vertex_spv_file = Cursor::new(&include_bytes!("./shader/vert.spv")[..]);
            let vertex_code =
                read_spv(&mut vertex_spv_file).expect("Failed to read vertex shader spv file.");
            let vertex_shader_info = vk::ShaderModuleCreateInfo::default().code(&vertex_code);
            let mut fragment_spv_file = Cursor::new(&include_bytes!("./shader/frag.spv")[..]);
            let fragment_code =
                read_spv(&mut fragment_spv_file).expect("Failed to read fragment shader spv file.");
            let fragment_shader_info = vk::ShaderModuleCreateInfo::default().code(&fragment_code);
            let vertex_shader_module = device
                .create_shader_module(&vertex_shader_info, None)
                .expect("Vertex shader module error.");
            let fragment_shader_module = device
                .create_shader_module(&fragment_shader_info, None)
                .expect("Fragment shader module error.");

            let shader_stages = [
                vk::PipelineShaderStageCreateInfo::default()
                    .name(c"main")
                    .stage(vk::ShaderStageFlags::VERTEX)
                    .module(vertex_shader_module),
                vk::PipelineShaderStageCreateInfo::default()
                    .name(c"main")
                    .stage(vk::ShaderStageFlags::FRAGMENT)
                    .module(fragment_shader_module),
            ];

            let color_rendering_format = [surface_format.format];

            let mut pipeline_create = vk::PipelineRenderingCreateInfo::default()
                .color_attachment_formats(&color_rendering_format)
                .depth_attachment_format(depth_format);

            let push_constant_ranges = [vk::PushConstantRange::default()
                .stage_flags(vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE)
                .size(size_of::<PushConstants>() as u32)];
            let pipeline_layout_create_info =
                vk::PipelineLayoutCreateInfo::default().push_constant_ranges(&push_constant_ranges);

            let pipeline_layout = device
                .create_pipeline_layout(&pipeline_layout_create_info, None)
                .expect("Failed to create pipeline layout.");

            let graphics_pipeline_create_info = vk::GraphicsPipelineCreateInfo::default()
                .stages(&shader_stages)
                .layout(pipeline_layout)
                .input_assembly_state(&input_assembly_state)
                .vertex_input_state(&vertex_input_state)
                .rasterization_state(&rasterization_state)
                .color_blend_state(&color_blend_state)
                .multisample_state(&multisample_state)
                .viewport_state(&viewport_state)
                .depth_stencil_state(&depth_stencil_state)
                .dynamic_state(&dynamic_state)
                .push_next(&mut pipeline_create);

            let pipeline = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphics_pipeline_create_info],
                    None,
                )
                .expect("Could not create graphics pipeline")[0];

            let new_buffer = |initial_data, size, usage| {
                AllocatedBuffer::new_with_data(
                    &device,
                    size,
                    device_memory_properties,
                    usage
                        | vk::BufferUsageFlags::SHADER_DEVICE_ADDRESS
                        | vk::BufferUsageFlags::TRANSFER_DST,
                    vk::SharingMode::EXCLUSIVE,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    setup_command_buffer,
                    setup_commands_reuse_fence,
                    present_queue,
                    initial_data,
                )
            };

            pub fn create_compute_pipeline<'a>(
                device: &ash::Device,
                pipeline_layout: vk::PipelineLayout,
                spv_file: &mut Cursor<&[u8]>,
                name: &'a CStr,
            ) -> vk::Pipeline {
                let bytecode = read_spv(spv_file).expect("Failed to read compute shader spv file.");
                let shader_module_create_info =
                    vk::ShaderModuleCreateInfo::default().code(&bytecode);
                unsafe {
                    let shader_module = device
                        .create_shader_module(&shader_module_create_info, None)
                        .expect("Compute shader module error.");
                    let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
                        .stage(
                            vk::PipelineShaderStageCreateInfo::default()
                                .stage(ShaderStageFlags::COMPUTE)
                                .module(shader_module)
                                .name(name),
                        )
                        .layout(pipeline_layout);
                    return device
                        .create_compute_pipelines(
                            vk::PipelineCache::null(),
                            &[pipeline_create_info],
                            None,
                        )
                        .expect("Could not create compute pipeline")[0];
                }
            };

            let mut allocate_bytecode = Cursor::new(&include_bytes!("./shader/allocate.spv")[..]);
            let allocate_pipeline =
                create_compute_pipeline(&device, pipeline_layout, &mut allocate_bytecode, c"main");
            // For cbt_reduce.spv (entry reduce)
            let mut reduce_bytecode = Cursor::new(&include_bytes!("./shader/reduce.spv")[..]);
            let reduce_pipeline =
                create_compute_pipeline(&device, pipeline_layout, &mut reduce_bytecode, c"main");

            // For split_element.spv (entry split_element)
            let mut split_element_bytecode =
                Cursor::new(&include_bytes!("./shader/split_element.spv")[..]);
            let split_element_pipeline = create_compute_pipeline(
                &device,
                pipeline_layout,
                &mut split_element_bytecode,
                c"main",
            );

            // For update_pointers.spv (entry update_pointers)
            let mut update_pointers_bytecode =
                Cursor::new(&include_bytes!("./shader/update_pointers.spv")[..]);
            let update_pointers_pipeline = create_compute_pipeline(
                &device,
                pipeline_layout,
                &mut update_pointers_bytecode,
                c"main",
            );

            // For prepare_merge.spv (entry prepare_merge)
            let mut prepare_merge_bytecode =
                Cursor::new(&include_bytes!("./shader/prepare_merge.spv")[..]);
            let prepare_merge_pipeline = create_compute_pipeline(
                &device,
                pipeline_layout,
                &mut prepare_merge_bytecode,
                c"main",
            );

            // For merge.spv (entry merge)
            let mut merge_bytecode = Cursor::new(&include_bytes!("./shader/merge.spv")[..]);
            let merge_pipeline =
                create_compute_pipeline(&device, pipeline_layout, &mut merge_bytecode, c"main");
            let new_buffer_sized =
                |initial_data, usage| new_buffer(initial_data, initial_data.len() as u64, usage);

            let scene_buffer_handles = SceneCPUHandles {
                root_bisector_vertices: new_buffer_sized(
                    cast_slice(&algorithm_data.root_bisector_vertices),
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                ),

                cbt_interior: new_buffer_sized(
                    cast_slice(&algorithm_data.cbt.interior[..]),
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                ),

                cbt_leaves: new_buffer_sized(
                    cast_slice(&algorithm_data.cbt.leaves[..]),
                    vk::BufferUsageFlags::UNIFORM_BUFFER,
                ),

                classification_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.classification_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),

                bisector_state_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.bisector_state_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),

                bisector_split_command_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.bisector_split_command_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),

                neighbors_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.neighbors_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),

                splitting_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.splitting_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),

                heapid_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.heapid_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),

                allocation_indices_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.allocation_indices_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),

                want_split_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.want_split_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
                want_merge_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.want_merge_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
                merging_bisector_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.merging_bisector_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),
                vertex_buffer: new_buffer_sized(
                    cast_slice(&algorithm_data.vertex_buffer[..]),
                    vk::BufferUsageFlags::STORAGE_BUFFER,
                ),

                allocate_pipeline: allocate_pipeline,
                reduce_pipeline: reduce_pipeline,
                split_element_pipeline: split_element_pipeline,
                update_pointers_pipeline: update_pointers_pipeline,
                prepare_merge_pipeline: prepare_merge_pipeline,
                merge_pipeline: merge_pipeline,
            };

            let scene = SceneDataGPU {
                // written once at initialization
                root_bisector_vertices: scene_buffer_handles
                    .root_bisector_vertices
                    .device_address(&device),
                // our concurrent binary tree
                // cbt: CBT,
                cbt_interior: scene_buffer_handles.cbt_interior.device_address(&device),
                cbt_leaves: scene_buffer_handles.cbt_leaves.device_address(&device),

                // classification stage
                classification_buffer: scene_buffer_handles
                    .bisector_state_buffer
                    .device_address(&device),
                bisector_state_buffer: scene_buffer_handles
                    .bisector_state_buffer
                    .device_address(&device),

                // prepare split
                bisector_split_command_buffer: scene_buffer_handles
                    .bisector_split_command_buffer
                    .device_address(&device),
                neighbors_buffer: scene_buffer_handles
                    .neighbors_buffer
                    .device_address(&device),
                splitting_buffer: scene_buffer_handles
                    .splitting_buffer
                    .device_address(&device),
                heapid_buffer: scene_buffer_handles.heapid_buffer.device_address(&device),

                // allocate
                allocation_indices_buffer: scene_buffer_handles
                    .heapid_buffer
                    .device_address(&device),

                // split
                want_split_buffer: scene_buffer_handles.heapid_buffer.device_address(&device),

                // prepare merge
                want_merge_buffer: scene_buffer_handles.heapid_buffer.device_address(&device),

                // merge
                merging_bisector_buffer: scene_buffer_handles.heapid_buffer.device_address(&device),

                // draw
                vertex_buffer: scene_buffer_handles.vertex_buffer.device_address(&device),

                // integers
                num_memory_blocks: algorithm_data.cbt.leaves.len() as u32 * BITFIELD_INT_SIZE,
                base_depth: algorithm_data.base_depth,
                cbt_depth: algorithm_data.cbt.depth,
            };
            let scene_buffer = new_buffer(
                byteslice(&scene),
                size_of::<SceneDataGPU>() as u64,
                vk::BufferUsageFlags::UNIFORM_BUFFER,
            );

            let dispatch = DispatchSizeGPU {
                remaining_memory_count: linear_dispatch(
                    algorithm_data
                        .remaining_memory_count
                        .load(Ordering::Relaxed),
                ),
                allocation_counter: linear_dispatch(0),
                want_split_buffer_count: linear_dispatch(0),
                splitting_buffer_count: linear_dispatch(0),
                want_merge_buffer_count: linear_dispatch(0),
                merging_bisector_count: linear_dispatch(0),
            };
            let dispatch_buffer = new_buffer(
                byteslice(&dispatch),
                size_of::<DispatchSizeGPU>() as u64,
                vk::BufferUsageFlags::STORAGE_BUFFER,
            );

            self.state = Some(State {
                window: window,
                instance: instance,
                device: device,
                swapchain: swapchain,
                swapchain_loader: swapchain_loader,
                setup_command_buffer: setup_command_buffer,
                draw_command_buffers: draw_command_buffers,
                present_queue: present_queue,
                present_image_views: present_image_views,
                present_images: present_images,
                depth_image_view: depth_image_view,
                pipeline: pipeline,
                pipeline_layout: pipeline_layout,
                present_complete_semaphore: present_complete_semaphore,
                render_complete_semaphore: render_complete_semaphore,
                frame_index: 0,
                resolution: surface_resolution,
                command_reuse_fences: command_reuse_fences,
                frame_pace_semaphore: frame_pace_semaphore,
                vertex_buffer: vertex_buffer,

                curr_ids_buffer: curr_ids_buffer,
                curr_ids_staging_buffer: curr_ids_staging_buffer,
                curr_ids_staging_buffer_ptr: curr_ids_staging_buffer_ptr,

                algorithm_data: algorithm_data,
                iteration_counter: 0,
                vertex_staging_buffer: vertex_staging_buffer,
                vertex_staging_buffer_ptr: vertex_staging_buffer_ptr,
                setup_commands_reuse_fence: setup_commands_reuse_fence,

                scene_buffer: scene_buffer,
                dispatch_buffer: dispatch_buffer,

                scene_buffer_handles,

                camera: CameraState {
                    pos: Vec3::new(2.0, 2.0, 2.0),
                    pitch: PI / 2.0,
                    yaw: 0.0,
                },
            })
        }
    }

    fn device_event(
        &mut self,
        event_loop: &ActiveEventLoop,
        _id: winit::event::DeviceId,
        event: DeviceEvent,
    ) {
        let state = match &mut self.state {
            Some(x) => x,
            None => {
                println!("No state found");
                return;
            }
        };

        match event {
            DeviceEvent::MouseMotion { delta } => {
                let (dx, dy) = delta;
                const LOOK_SENSITIVITY: f32 = 0.01;
                state.camera.yaw -= LOOK_SENSITIVITY * dx as f32;
                state.camera.pitch += LOOK_SENSITIVITY * dy as f32;

                if state.camera.pitch < 0.0 {
                    state.camera.pitch = 0.0;
                } else if state.camera.pitch > PI {
                    state.camera.pitch = PI;
                }
            }
            _ => (),
        }
    }
    fn window_event(&mut self, event_loop: &ActiveEventLoop, _id: WindowId, event: WindowEvent) {
        let state = match &mut self.state {
            Some(x) => x,
            None => {
                println!("No state found");
                return;
            }
        };
        match event {
            WindowEvent::CloseRequested => {
                event_loop.exit();
            }
            WindowEvent::Resized(_) => {
                state.window.request_redraw(); // first redraw happens here
                // TODO: implement actual resizing
            }
            WindowEvent::RedrawRequested => {
                // println!("redraw requested");
                state.render();
                state.window.request_redraw(); // this is how u do rendering loops in winit i guess
            }
            WindowEvent::KeyboardInput {
                event:
                    KeyEvent {
                        logical_key: key,
                        state: ElementState::Pressed,
                        ..
                    },
                ..
            } => match key.as_ref() {
                Key::Character("S") => {
                    // let split_slots = [1, 7, 9, 14, 19];
                    // let split_slots = [1, 7, 9, 14, 16];
                    // if state.iteration_counter < split_slots.len() as u32 {
                    // let slot = split_slots[state.iteration_counter as usize];
                    let slots = [
                        state.algorithm_data.cbt.one_bit_to_id(
                            state.iteration_counter % state.algorithm_data.cbt.interior[0],
                        ),
                        (state.algorithm_data.cbt.one_bit_to_id(
                            (state.iteration_counter + 1) % state.algorithm_data.cbt.interior[0],
                        )),
                        (state.algorithm_data.cbt.one_bit_to_id(
                            (state.iteration_counter + 2) % state.algorithm_data.cbt.interior[0],
                        )),
                        (state.algorithm_data.cbt.one_bit_to_id(
                            (state.iteration_counter + 3) % state.algorithm_data.cbt.interior[0],
                        )),
                    ];
                    let slots: Vec<u32> = slots
                        .into_iter()
                        .filter(|x| {
                            state.algorithm_data.heapid_buffer[*x as usize] & (1 << 31) == 0
                        })
                        .collect();
                    // println!("Refining tri and at slot {:?}", slot);
                    for slot in slots.iter() {
                        let leaf = state.algorithm_data.cbt.leaves[(slot / 32) as usize]
                            .load(Ordering::Relaxed);
                        // println!("Leaf: {:b}", leaf);
                        debug_assert!(leaf & (1 << (slot % 32)) != 0);
                    }

                    state
                        .algorithm_data
                        .want_split_buffer_count
                        .fetch_add(slots.len() as u32, Ordering::Relaxed);
                    for (i, slot) in slots.iter().enumerate() {
                        state.algorithm_data.want_split_buffer[i] = *slot;
                    }
                    state.algorithm_data.iterate();
                    state.iteration_counter += 1;
                    state.algorithm_data.reset();
                    debug_assert!(
                        state
                            .algorithm_data
                            .want_split_buffer_count
                            .load(Ordering::Relaxed)
                            == 0
                    );
                    // }
                    let vertex_buffer_size =
                        (size_of::<[Vec3; 3]>() * state.algorithm_data.vertex_buffer.len()) as u64;
                    unsafe {
                        let mut staging_buffer_slice = Align::new(
                            state.vertex_staging_buffer_ptr,
                            align_of::<f32>() as u64,
                            vertex_buffer_size,
                        );
                        staging_buffer_slice
                            .copy_from_slice(state.algorithm_data.vertex_buffer.as_slice());

                        let ids_buffer_size = (64 * size_of::<u32>()) as u64;
                        let mut staging_buffer_slice = Align::new(
                            state.curr_ids_staging_buffer_ptr,
                            align_of::<f32>() as u64,
                            ids_buffer_size,
                        );
                        let curr_ids_buffer: Vec<u32> = (0..state.algorithm_data.cbt.interior[0])
                            .map(|tid| min(state.algorithm_data.cbt.one_bit_to_id(tid), 64))
                            .collect();
                        println!("curr_ids_buffer: {:?}", curr_ids_buffer);
                        staging_buffer_slice.copy_from_slice(curr_ids_buffer.as_slice());
                        // WARNING: THERES NO SYNCHRONIZATION HERE
                        // initialization command buffer
                        record_submit_commandbuffer(
                            &state.device,
                            state.setup_command_buffer,
                            state.setup_commands_reuse_fence,
                            state.present_queue,
                            &[],
                            &[],
                            &[],
                            |device, setup_command_buffer| {
                                let vertex_copy =
                                    vk::BufferCopy::default().size(vertex_buffer_size);
                                device.cmd_copy_buffer(
                                    setup_command_buffer,
                                    state.vertex_staging_buffer.buffer,
                                    state.vertex_buffer.buffer,
                                    &[vertex_copy],
                                );

                                let ids_copy = vk::BufferCopy::default().size(ids_buffer_size);
                                device.cmd_copy_buffer(
                                    setup_command_buffer,
                                    state.curr_ids_staging_buffer.buffer,
                                    state.curr_ids_buffer.buffer,
                                    &[ids_copy],
                                );
                            },
                        );
                    }
                }
                Key::Character("M") => {
                    // let split_slots = [1, 7, 9, 14, 19];
                    // let split_slots = [1, 7, 9, 14, 16];
                    // if state.iteration_counter < split_slots.len() as u32 {
                    // let slot = split_slots[state.iteration_counter as usize];
                    let slots: Vec<u32> = (0..(state.algorithm_data.cbt.interior[0]))
                        .map(|tid| state.algorithm_data.cbt.one_bit_to_id(tid))
                        .filter(|slot_idx| {
                            let heapid = state.algorithm_data.heapid_buffer[*slot_idx as usize];
                            if heap_id_depth(heapid) == state.algorithm_data.base_depth {
                                return false;
                            }
                            state.algorithm_data.bisector_state_buffer[*slot_idx as usize] =
                                SIMPLIFY;
                            if heapid % 2 != 1 {
                                return false;
                            }
                            return true;
                        })
                        .collect();

                    println!("num_slots: {:?}", slots.len());

                    state
                        .algorithm_data
                        .want_merge_buffer_count
                        .fetch_add(slots.len() as u32, Ordering::Relaxed);
                    for (i, slot) in slots.iter().enumerate() {
                        state.algorithm_data.want_merge_buffer[i] = *slot;
                    }
                    state.algorithm_data.iterate();
                    state.iteration_counter += 1;
                    state.algorithm_data.reset();
                    let vertex_buffer_size =
                        (size_of::<[Vec3; 3]>() * state.algorithm_data.vertex_buffer.len()) as u64;
                    unsafe {
                        let mut staging_buffer_slice = Align::new(
                            state.vertex_staging_buffer_ptr,
                            align_of::<f32>() as u64,
                            vertex_buffer_size,
                        );
                        staging_buffer_slice
                            .copy_from_slice(state.algorithm_data.vertex_buffer.as_slice());

                        let ids_buffer_size = (64 * size_of::<u32>()) as u64;
                        let mut staging_buffer_slice = Align::new(
                            state.curr_ids_staging_buffer_ptr,
                            align_of::<f32>() as u64,
                            ids_buffer_size,
                        );
                        let curr_ids_buffer: Vec<u32> = (0..state.algorithm_data.cbt.interior[0])
                            .map(|tid| min(state.algorithm_data.cbt.one_bit_to_id(tid), 64))
                            .collect();
                        // println!("curr_ids_buffer: {:?}", curr_ids_buffer);
                        staging_buffer_slice.copy_from_slice(curr_ids_buffer.as_slice());
                        // WARNING: THERES NO SYNCHRONIZATION HERE
                        // initialization command buffer
                        record_submit_commandbuffer(
                            &state.device,
                            state.setup_command_buffer,
                            state.setup_commands_reuse_fence,
                            state.present_queue,
                            &[],
                            &[],
                            &[],
                            |device, setup_command_buffer| {
                                let vertex_copy =
                                    vk::BufferCopy::default().size(vertex_buffer_size);
                                device.cmd_copy_buffer(
                                    setup_command_buffer,
                                    state.vertex_staging_buffer.buffer,
                                    state.vertex_buffer.buffer,
                                    &[vertex_copy],
                                );

                                let ids_copy = vk::BufferCopy::default().size(ids_buffer_size);
                                device.cmd_copy_buffer(
                                    setup_command_buffer,
                                    state.curr_ids_staging_buffer.buffer,
                                    state.curr_ids_buffer.buffer,
                                    &[ids_copy],
                                );
                            },
                        );
                    }
                }
                Key::Character("N") => {
                    for i in 0..state.algorithm_data.cbt.interior[0] {
                        let curr_id = state.algorithm_data.cbt.one_bit_to_id(i);
                        println!("curr_id : {:?}", curr_id);
                        println!(
                            "neighbors: {:?}",
                            state.algorithm_data.neighbors_buffer[curr_id as usize]
                        );
                    }
                }
                Key::Character("w") => {
                    state.camera.pos += state.camera.lookdir() * 0.05;
                }
                Key::Character("a") => {}
                Key::Character("s") => {
                    state.camera.pos += state.camera.lookdir() * -0.05;
                }
                Key::Character("d") => {}
                _ => (),
            },
            _ => (),
        }
    }
}

fn main() -> Result<(), Box<dyn Error>> {
    let event_loop = EventLoop::new()?;
    let mut app = App::new();
    event_loop.run_app(&mut app)?;
    Ok(())
}
