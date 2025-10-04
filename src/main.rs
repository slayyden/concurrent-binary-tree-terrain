// use ash::Instance;
use ash::{
    Entry,
    ext::debug_utils,
    khr::{surface, swapchain},
    util::read_spv,
    vk::{
        self, BufferUsageFlags, CommandBufferSubmitInfo, Extent2D, PhysicalDeviceMemoryProperties,
        PipelineStageFlags2, ShaderStageFlags,
    },
};
use dirt_jam::*;
use glam::{Mat4, Vec3, Vec3A};
use std::{
    alloc, cmp::max, cmp::min, error::Error, f32::consts::PI, ffi::CStr, io::Cursor, iter::Map,
    iter::zip, mem::offset_of, num, os::raw::c_char, sync::atomic::Ordering, thread, time, u64,
};
use winit::{
    application::ApplicationHandler,
    event::{DeviceEvent, ElementState, KeyEvent, WindowEvent},
    event_loop::{ActiveEventLoop, EventLoop},
    keyboard::Key,
    raw_window_handle::{HasDisplayHandle, HasWindowHandle},
    window::{Window, WindowId},
};

const NUM_ROLLBACK_FRAMES: usize = 2;

struct State {
    // dbg
    window: Window,
    // instance: ash::Instance,
    device: ash::Device,
    // setup_commands_reuse_fence: vk::Fence,
    // setup_command_buffer: vk::CommandBuffer,
    draw_command_buffers: [vk::CommandBuffer; 3],
    present_queue: vk::Queue,
    pipeline: vk::Pipeline,
    wire_pipeline: vk::Pipeline,
    pipeline_layout: vk::PipelineLayout,
    // the we can get any number of images above the minimum, length is unknown at comptime
    present_image_views: Vec<vk::ImageView>,
    present_images: Vec<vk::Image>,
    depth_image_views: Vec<vk::ImageView>,
    swapchain: vk::SwapchainKHR,
    swapchain_loader: swapchain::Device,

    frame_index: u64,
    present_complete_semaphore: [vk::Semaphore; 3],
    render_complete_semaphore: [vk::Semaphore; 3],
    frame_pace_semaphore: vk::Semaphore,

    resolution: vk::Extent2D,
    command_reuse_fences: [vk::Fence; 3],

    pipeline_handles: PipelineHandles,
    cbt_scenes: [CBTScene; NUM_ROLLBACK_FRAMES],

    camera: CameraState,
    algorithm_data: PipelineData,
    divide: bool,
    num_iters: i32,
    curr_iter: i32, // must be in [max(0, num_iters - NUM_ROLLBACK_FRAMES + 1), num_iters]
    rendering_mode: RenderingMode,
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
                        float32: [1.0, 0.0, 0.0, 1.0] as [f32; 4],
                    },
                })
                .image_layout(vk::ImageLayout::COLOR_ATTACHMENT_OPTIMAL)
                .image_view(self.present_image_views[present_index as usize])];

            let depth_attachment = vk::RenderingAttachmentInfo::default()
                .image_view(self.depth_image_views[present_index as usize])
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
            dev.reset_command_buffer(*cmdbuf, vk::CommandBufferResetFlags::RELEASE_RESOURCES)
                .expect("Reset command buffer failed.");
            dev.begin_command_buffer(
                *cmdbuf,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )
            .expect("Could not begin command buffer.");

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

            /*
            let memory_barrier = vk::MemoryBarrier::default()
                .src_access_mask(vk::AccessFlags::SHADER_WRITE)
                .dst_access_mask(vk::AccessFlags::SHADER_READ);
            let global_memory_barrier = || {
                dev.cmd_pipeline_barrier(
                    *cmdbuf,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[memory_barrier],
                    &[],
                    &[],
                )
            }; */

            let debug_barrier = vk::MemoryBarrier2::default()
                .src_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .src_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_COMMANDS)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ | vk::AccessFlags2::MEMORY_WRITE);
            let barrier = vk::MemoryBarrier2::default()
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER);
            let barrier = debug_barrier;
            let compute_write_compute_read_memory_barrier = {
                || {
                    dev.cmd_pipeline_barrier2(
                        *cmdbuf,
                        &vk::DependencyInfo::default().memory_barriers(&[barrier]),
                    );
                }
            };
            let barrier = vk::MemoryBarrier2::default()
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::INDIRECT_COMMAND_READ)
                // DRAW_INDIRECT includes consumption of indirect command buffers by an indirect draw, compute, or raytracing command
                .dst_stage_mask(vk::PipelineStageFlags2::DRAW_INDIRECT);
            let barrier = debug_barrier;
            let compute_write_indirect_read_barrier = {
                || {
                    dev.cmd_pipeline_barrier2(
                        *cmdbuf,
                        &vk::DependencyInfo::default().memory_barriers(&[barrier]),
                    );
                }
            };
            let barrier = vk::MemoryBarrier2::default()
                .src_access_mask(vk::AccessFlags2::MEMORY_WRITE)
                .src_stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
                .dst_access_mask(vk::AccessFlags2::MEMORY_READ)
                .dst_stage_mask(vk::PipelineStageFlags2::ALL_GRAPHICS); // includes draw indirect
            let barrier = debug_barrier;
            let compute_write_graphics_read_memory_barrier = {
                || {
                    dev.cmd_pipeline_barrier2(
                        *cmdbuf,
                        &vk::DependencyInfo::default().memory_barriers(&[barrier]),
                    );
                }
            };
            // compute_write_compute_read_memory_barrier();

            let (prev_scene, next_scene, rendering_mode) = if self.divide {
                // get a "mutable reference" to the correct scene
                let (prev_scene, next_scene) = {
                    // get prev and next scenes
                    let prev_idx = self.num_iters % NUM_ROLLBACK_FRAMES as i32;
                    let next_idx = (self.num_iters + 1) % NUM_ROLLBACK_FRAMES as i32;

                    let prev_cbt_scene = &self.cbt_scenes[prev_idx as usize];
                    let next_cbt_scene = &self.cbt_scenes[next_idx as usize];

                    // copy data to start next scene
                    if NUM_ROLLBACK_FRAMES > 1 {
                        zip(
                            prev_cbt_scene.scene_buffer_handles.gpu_slices_iter(),
                            next_cbt_scene.scene_buffer_handles.gpu_slices_iter(),
                        )
                        .for_each(|(prev, next)| {
                            let region = vk::BufferCopy::default().size(prev.size_in_bytes);
                            dev.cmd_copy_buffer(
                                *cmdbuf,
                                prev.gpu_buffer,
                                next.gpu_buffer,
                                &[region],
                            );
                        });
                    }
                    (prev_cbt_scene, next_cbt_scene)
                };
                dev.cmd_push_constants(
                    *cmdbuf,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
                    0,
                    byteslice(&get_push_constants(
                        prev_scene,
                        next_scene,
                        &self.camera,
                        RenderingMode::Default, // compute doesn't care about rendering mode
                    )),
                );
                // COMPUTE
                // classify
                {
                    dev.cmd_bind_pipeline(
                        *cmdbuf,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline_handles.classify_pipeline,
                    );

                    dev.cmd_dispatch_indirect(
                        *cmdbuf,
                        next_scene.dispatch_buffer.buffer(),
                        offset_of!(DispatchDataGPU, dispatch_vertex_compute_command) as u64,
                    );
                }
                // compute to compute barrier
                compute_write_compute_read_memory_barrier();

                // set up indirect buffer for splitting
                {
                    dev.cmd_bind_pipeline(
                        *cmdbuf,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline_handles.dispatch_split_pipeline,
                    );
                    dev.cmd_dispatch(*cmdbuf, 1, 1, 1);
                }

                // compute write -> indirect buffer read
                compute_write_indirect_read_barrier();
                // splitting
                {
                    dev.cmd_bind_pipeline(
                        *cmdbuf,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline_handles.split_element_pipeline,
                    );
                    dev.cmd_dispatch_indirect(
                        *cmdbuf,
                        next_scene.dispatch_buffer.buffer(),
                        offset_of!(DispatchDataGPU, dispatch_split_command) as u64,
                    );
                }
                compute_write_compute_read_memory_barrier();

                // set up indirect dispatch for allocation
                {
                    dev.cmd_bind_pipeline(
                        *cmdbuf,
                        vk::PipelineBindPoint::COMPUTE,
                        self.pipeline_handles.dispatch_allocate_pipeline,
                    );
                    dev.cmd_dispatch(*cmdbuf, 1, 1, 1);
                }
                compute_write_indirect_read_barrier();

                // allocate

                dev.cmd_bind_pipeline(
                    *cmdbuf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_handles.allocate_pipeline,
                );
                dev.cmd_dispatch_indirect(
                    *cmdbuf,
                    next_scene.dispatch_buffer.buffer(),
                    offset_of!(DispatchDataGPU, dispatch_allocate_command) as u64,
                );
                compute_write_compute_read_memory_barrier();

                // compute -> compute global memory barrier
                // update pointers
                dev.cmd_bind_pipeline(
                    *cmdbuf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_handles.update_pointers_pipeline,
                );
                dev.cmd_dispatch_indirect(
                    *cmdbuf,
                    next_scene.dispatch_buffer.buffer(),
                    offset_of!(DispatchDataGPU, dispatch_allocate_command) as u64,
                );
                compute_write_compute_read_memory_barrier();

                // compute -> compute global memory barrier
                // set up indirect dispatch for merging
                dev.cmd_bind_pipeline(
                    *cmdbuf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_handles.dispatch_prepare_merge_pipeline,
                );
                dev.cmd_dispatch(*cmdbuf, 1, 1, 1);
                compute_write_indirect_read_barrier();

                // prepare merge
                dev.cmd_bind_pipeline(
                    *cmdbuf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_handles.prepare_merge_pipeline,
                );
                dev.cmd_dispatch_indirect(
                    *cmdbuf,
                    next_scene.dispatch_buffer.buffer(),
                    offset_of!(DispatchDataGPU, dispatch_prepare_merge_command) as u64,
                );
                compute_write_compute_read_memory_barrier();

                // compute -> compute global memory barrier
                // merge
                dev.cmd_bind_pipeline(
                    *cmdbuf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_handles.merge_pipeline,
                );
                dev.cmd_dispatch_indirect(
                    *cmdbuf,
                    next_scene.dispatch_buffer.buffer(),
                    offset_of!(DispatchDataGPU, dispatch_prepare_merge_command) as u64,
                );
                compute_write_compute_read_memory_barrier();
                // compute -> compute global memory barrier
                // reduce
                dev.cmd_bind_pipeline(
                    *cmdbuf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_handles.reduce_pipeline,
                );
                dev.cmd_dispatch(*cmdbuf, 1, 1, 1);

                // compute -> compute global memory barrier AND indirect read
                compute_write_compute_read_memory_barrier();

                // vertex compute
                dev.cmd_bind_pipeline(
                    *cmdbuf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_handles.vertex_compute_pipeline,
                );
                dev.cmd_dispatch_indirect(
                    *cmdbuf,
                    next_scene.dispatch_buffer.buffer(),
                    offset_of!(DispatchDataGPU, dispatch_vertex_compute_command) as u64,
                );
                // compute -> graphics global memory barrier AND indirect read
                compute_write_graphics_read_memory_barrier();
                compute_write_compute_read_memory_barrier();
                (prev_scene, next_scene, RenderingMode::Default)
            } else {
                if self.curr_iter == self.num_iters {
                    (
                        &self.cbt_scenes[(self.num_iters as usize + NUM_ROLLBACK_FRAMES - 1)
                            % NUM_ROLLBACK_FRAMES],
                        &self.cbt_scenes[self.num_iters as usize % NUM_ROLLBACK_FRAMES],
                        RenderingMode::Default,
                    )
                } else {
                    (
                        &self.cbt_scenes[self.curr_iter as usize % NUM_ROLLBACK_FRAMES],
                        &self.cbt_scenes[(self.curr_iter as usize + 1) % NUM_ROLLBACK_FRAMES],
                        RenderingMode::RollbackDefault,
                    )
                }
            };
            dev.cmd_begin_rendering(*cmdbuf, &rendering_info);

            // RENDERING CORE
            {
                // push constants
                dev.cmd_push_constants(
                    *cmdbuf,
                    self.pipeline_layout,
                    vk::ShaderStageFlags::VERTEX | vk::ShaderStageFlags::COMPUTE,
                    0,
                    byteslice(&get_push_constants(
                        prev_scene,
                        next_scene,
                        &self.camera,
                        rendering_mode,
                    )),
                );
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
                                    float32: [1.0, 0.0, 0.0, 1.0],
                                },
                            }),
                        vk::ClearAttachment::default()
                            .aspect_mask(vk::ImageAspectFlags::DEPTH)
                            .clear_value(vk::ClearValue {
                                depth_stencil: vk::ClearDepthStencilValue {
                                    depth: 0.0,
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

                let vertices_scene = if self.num_iters == self.curr_iter {
                    next_scene
                } else {
                    prev_scene
                };

                // draw the triangles
                dev.cmd_draw_indirect(
                    *cmdbuf,
                    vertices_scene.dispatch_buffer.buffer(),
                    offset_of!(DispatchDataGPU, draw_indirect_command) as u64,
                    1,
                    0,
                );
                // draw the wireframe
                dev.cmd_bind_pipeline(*cmdbuf, vk::PipelineBindPoint::GRAPHICS, self.wire_pipeline);
                dev.cmd_draw_indirect(
                    *cmdbuf,
                    vertices_scene.dispatch_buffer.buffer(),
                    offset_of!(DispatchDataGPU, draw_indirect_command) as u64,
                    1,
                    0,
                );

                dev.cmd_end_rendering(*cmdbuf);
            }

            // POST RENDERING
            if self.divide {
                self.num_iters += 1;
                self.curr_iter += 1;
                println!("num_iters: {:?}", self.num_iters);

                dev.cmd_bind_pipeline(
                    *cmdbuf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_handles.validate_pipeline,
                );
                dev.cmd_dispatch_indirect(
                    *cmdbuf,
                    next_scene.dispatch_buffer.buffer(),
                    offset_of!(DispatchDataGPU, dispatch_vertex_compute_command) as u64,
                );
                compute_write_compute_read_memory_barrier();
                // no barrier needed
                // reset
                dev.cmd_bind_pipeline(
                    *cmdbuf,
                    vk::PipelineBindPoint::COMPUTE,
                    self.pipeline_handles.reset_pipeline,
                );
                dev.cmd_dispatch(*cmdbuf, 1, 1, 1);
            }
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
            dev.end_command_buffer(*cmdbuf)
                .expect("End command buffer.");

            {
                let command_buffer_infos =
                    [CommandBufferSubmitInfo::default().command_buffer(*cmdbuf)];
                let wait_semaphore_infos = [
                    // binary semaphore
                    vk::SemaphoreSubmitInfo::default()
                        .stage_mask(PipelineStageFlags2::ALL_COMMANDS)
                        .semaphore(self.present_complete_semaphore[semaphore_index]),
                    // timeline semaphore
                    vk::SemaphoreSubmitInfo::default()
                        .stage_mask(PipelineStageFlags2::ALL_COMMANDS)
                        .semaphore(self.frame_pace_semaphore)
                        .value(self.frame_index),
                ];
                let signal_semaphore_infos = [
                    // binary semaphore
                    vk::SemaphoreSubmitInfo::default()
                        .stage_mask(PipelineStageFlags2::ALL_COMMANDS)
                        .semaphore(self.render_complete_semaphore[semaphore_index]),
                    // timeline semaphore
                    vk::SemaphoreSubmitInfo::default()
                        .stage_mask(PipelineStageFlags2::ALL_COMMANDS)
                        .semaphore(self.frame_pace_semaphore)
                        .value(self.frame_index + 1), // next frame is ready to process
                ];

                let submit_info = vk::SubmitInfo2::default()
                    .command_buffer_infos(&command_buffer_infos)
                    .wait_semaphore_infos(&wait_semaphore_infos)
                    .signal_semaphore_infos(&signal_semaphore_infos);

                dev.queue_submit2(
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
            if true {
                if true {
                    dev.wait_semaphores(
                        &vk::SemaphoreWaitInfo::default()
                            .semaphores(&[self.frame_pace_semaphore])
                            .values(&[self.frame_index + 1]),
                        u64::MAX,
                    )
                    .expect("Wait semaphore");
                } else {
                    dev.device_wait_idle().expect("wait idle")
                }

                self.frame_index += 1;
            }
        }
        self.divide = false;
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
        let entry;
        unsafe {
            entry = Entry::load().expect("Could not load Vulkan Entry");
        }

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
        // MAY NEED TO REMOVE REFERENCES TO THIS
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
                            let supports_compute = queue_family_info
                                .queue_flags
                                .contains(vk::QueueFlags::COMPUTE);
                            if supports_graphics && supports_surface && supports_compute {
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
            let mut vulkan_10_features =
                vk::PhysicalDeviceFeatures::default().fill_mode_non_solid(true);
            let mut vulkan_11_features =
                vk::PhysicalDeviceVulkan11Features::default().shader_draw_parameters(true);
            let mut vulkan_12_features = vk::PhysicalDeviceVulkan12Features::default()
                .buffer_device_address(true)
                .timeline_semaphore(true)
                .scalar_block_layout(true);
            let mut vulkan_13_features = vk::PhysicalDeviceVulkan13Features::default()
                .dynamic_rendering(true)
                .synchronization2(true);
            let device_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_info))
                .enabled_extension_names(&device_extension_names_raw)
                .enabled_features(&mut vulkan_10_features)
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
            let depth_format = vk::Format::D32_SFLOAT;
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

            let depth_images: Vec<vk::Image> = present_images
                .iter()
                .map(|_| device.create_image(&depth_image_create_info, None).unwrap())
                .collect();

            let depth_image_memory_req = device.get_image_memory_requirements(depth_images[0]);
            let depth_image_memory_index = find_memorytype_index(
                &depth_image_memory_req,
                &device_memory_properties,
                vk::MemoryPropertyFlags::DEVICE_LOCAL,
            )
            .expect("Unable to find suitable memory index for depth image.");

            let depth_image_allocate_info = vk::MemoryAllocateInfo::default()
                .allocation_size(depth_image_memory_req.size)
                .memory_type_index(depth_image_memory_index);

            for depth_image in depth_images.iter() {
                let depth_image_memory = device
                    .allocate_memory(&depth_image_allocate_info, None)
                    .unwrap();
                device
                    .bind_image_memory(*depth_image, depth_image_memory, 0)
                    .expect("Unable to bind depth image memory.");
            }

            let fence_signalled_info =
                vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::SIGNALED);

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

            const SCALE: f32 = 100.0;
            let halfedge_mesh = HalfedgeMesh {
                verts: vec![
                    Vec3::new(SCALE, 0.0, 0.0),
                    Vec3::new(0.0, SCALE, 0.0),
                    Vec3::new(-SCALE, 0.0, 0.0),
                    Vec3::new(0.0, -SCALE, 0.0),
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
            debug_assert!(algorithm_data.cbt.interior.len() == 8191);
            debug_assert!(algorithm_data.want_split_buffer.len() == (1 << 17));
            debug_assert!(algorithm_data.base_depth == 3);

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
                    let layout_transition_barriers: Vec<vk::ImageMemoryBarrier> = depth_images
                        .iter()
                        .map(|depth_image| {
                            vk::ImageMemoryBarrier::default()
                                .image(*depth_image)
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
                                )
                        })
                        .collect();

                    device.cmd_pipeline_barrier(
                        setup_command_buffer,
                        vk::PipelineStageFlags::BOTTOM_OF_PIPE,
                        vk::PipelineStageFlags::LATE_FRAGMENT_TESTS,
                        vk::DependencyFlags::empty(),
                        &[],
                        &[],
                        layout_transition_barriers.as_slice(),
                    );
                },
            );

            device.device_wait_idle().expect("Wait idle.");
            // vertex_staging_buffer.destroy(&device);

            let depth_image_views: Vec<vk::ImageView> = depth_images
                .into_iter()
                .map(|depth_image| {
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
                    device
                        .create_image_view(&depth_image_view_info, None)
                        .unwrap()
                })
                .collect();

            // ----------------------------------------------------------------
            // create rendering pipeline

            let input_assembly_state = vk::PipelineInputAssemblyStateCreateInfo::default()
                .topology(vk::PrimitiveTopology::TRIANGLE_LIST);
            let rasterization_state = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::FILL)
                .cull_mode(vk::CullModeFlags::BACK)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            let rasterization_state_line = vk::PipelineRasterizationStateCreateInfo::default()
                .polygon_mode(vk::PolygonMode::LINE)
                .cull_mode(vk::CullModeFlags::NONE)
                .front_face(vk::FrontFace::COUNTER_CLOCKWISE)
                .line_width(1.0);
            // let blend_attachment_state = vk::PipelineColorBlendAttachmentState::default();
            let color_attachment_state = [vk::PipelineColorBlendAttachmentState::default()
                .color_write_mask(vk::ColorComponentFlags::RGBA)];
            let color_blend_state = vk::PipelineColorBlendStateCreateInfo::default()
                .attachments(&color_attachment_state);

            let depth_stencil_state = vk::PipelineDepthStencilStateCreateInfo::default()
                .depth_compare_op(vk::CompareOp::GREATER);
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

            let wire_pipeline = {
                let mut wireframe_fragment_spv_file =
                    Cursor::new(&include_bytes!("./shader/wirefrag.spv")[..]);
                let wireframe_fragment_code = read_spv(&mut wireframe_fragment_spv_file)
                    .expect("Failed to read wireframe_fragment shader spv file.");
                let wireframe_fragment_shader_info =
                    vk::ShaderModuleCreateInfo::default().code(&wireframe_fragment_code);
                let vertex_shader_module = device
                    .create_shader_module(&vertex_shader_info, None)
                    .expect("Vertex shader module error.");
                let wireframe_fragment_shader_module = device
                    .create_shader_module(&wireframe_fragment_shader_info, None)
                    .expect("wireframe_fragment shader module error.");
                let wireframe_shader_stages = [
                    vk::PipelineShaderStageCreateInfo::default()
                        .name(c"main")
                        .stage(vk::ShaderStageFlags::VERTEX)
                        .module(vertex_shader_module),
                    vk::PipelineShaderStageCreateInfo::default()
                        .name(c"main")
                        .stage(vk::ShaderStageFlags::FRAGMENT)
                        .module(wireframe_fragment_shader_module),
                ];

                let depth_stencil_state_wire = vk::PipelineDepthStencilStateCreateInfo::default()
                    .depth_compare_op(vk::CompareOp::NEVER);
                let wireframe_pipeline_create_info = graphics_pipeline_create_info
                    .stages(&wireframe_shader_stages)
                    .rasterization_state(&rasterization_state_line)
                    .depth_stencil_state(&depth_stencil_state_wire);
                device
                    .create_graphics_pipelines(
                        vk::PipelineCache::null(),
                        &[wireframe_pipeline_create_info],
                        None,
                    )
                    .expect("Could not create graphics pipeline")[0]
            };

            let pipeline = device
                .create_graphics_pipelines(
                    vk::PipelineCache::null(),
                    &[graphics_pipeline_create_info],
                    None,
                )
                .expect("Could not create graphics pipeline")[0];

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
            }

            // for classify.pv
            let mut classify_bytecode = Cursor::new(&include_bytes!("./shader/classify.spv")[..]);
            let classify_pipeline =
                create_compute_pipeline(&device, pipeline_layout, &mut classify_bytecode, c"main");

            // for allocate.spv
            let mut allocate_bytecode = Cursor::new(&include_bytes!("./shader/allocate.spv")[..]);
            let allocate_pipeline =
                create_compute_pipeline(&device, pipeline_layout, &mut allocate_bytecode, c"main");
            // For cbt_reduce.spv (entry reduce)
            let mut reduce_bytecode = Cursor::new(&include_bytes!("./shader/reduce.spv")[..]);
            let reduce_pipeline =
                create_compute_pipeline(&device, pipeline_layout, &mut reduce_bytecode, c"main");

            let mut post_reduce_bytecode =
                Cursor::new(&include_bytes!("./shader/post_reduce.spv")[..]);
            let post_reduce_pipeline = create_compute_pipeline(
                &device,
                pipeline_layout,
                &mut post_reduce_bytecode,
                c"main",
            );

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

            let mut dispatch_allocate_bytecode =
                Cursor::new(&include_bytes!("./shader/dispatch_allocate.spv")[..]);
            let dispatch_allocate_pipeline = create_compute_pipeline(
                &device,
                pipeline_layout,
                &mut dispatch_allocate_bytecode,
                c"main",
            );

            let mut dispatch_split_bytecode =
                Cursor::new(&include_bytes!("./shader/dispatch_split.spv")[..]);
            let dispatch_split_pipeline = create_compute_pipeline(
                &device,
                pipeline_layout,
                &mut dispatch_split_bytecode,
                c"main",
            );

            let mut dispatch_prepare_merge_bytecode =
                Cursor::new(&include_bytes!("./shader/dispatch_prepare_merge.spv")[..]);
            let dispatch_prepare_merge_pipeline = create_compute_pipeline(
                &device,
                pipeline_layout,
                &mut dispatch_prepare_merge_bytecode,
                c"main",
            );

            let mut vertex_compute_bytecode =
                Cursor::new(&include_bytes!("./shader/vertex_compute.spv")[..]);
            let vertex_compute_pipeline = create_compute_pipeline(
                &device,
                pipeline_layout,
                &mut vertex_compute_bytecode,
                c"main",
            );

            let mut validate_bytecode = Cursor::new(&include_bytes!("./shader/validate.spv")[..]);
            let validate_pipeline =
                create_compute_pipeline(&device, pipeline_layout, &mut validate_bytecode, c"main");
            // For reset.spv (entry reset)
            let mut reset_bytecode = Cursor::new(&include_bytes!("./shader/reset.spv")[..]);
            let reset_pipeline =
                create_compute_pipeline(&device, pipeline_layout, &mut reset_bytecode, c"main");

            // For merge.spv (entry merge)
            let mut merge_bytecode = Cursor::new(&include_bytes!("./shader/merge.spv")[..]);
            let merge_pipeline =
                create_compute_pipeline(&device, pipeline_layout, &mut merge_bytecode, c"main");

            let split_cmd_buffer_nonatomic: &[u32] =
                std::mem::transmute(algorithm_data.bisector_split_command_buffer.as_slice());
            let leaves_buffer_nonatomic: &[u32] =
                std::mem::transmute(algorithm_data.cbt.leaves.as_slice());

            let mem_props = device_memory_properties;
            let mut curr_id_data = vec![INVALID_POINTER; algorithm_data.num_memory_blocks as usize];
            for i in 0..6 {
                curr_id_data[i] = i as u32;
            }

            struct BufferCreationBoilerplate<'a> {
                device: &'a ash::Device,
                mem_props: PhysicalDeviceMemoryProperties,
                command_buffer: vk::CommandBuffer,
                command_buffer_reuse_fence: vk::Fence,
                queue: vk::Queue,
            }
            fn allocated_buffer_from_data<'a, T: Copy>(
                data: &[T],
                usage: vk::BufferUsageFlags,
                boilerplate: &'a BufferCreationBoilerplate,
            ) -> AllocatedBuffer<T> {
                GPUBuffer::<T>::new_with_data(
                    boilerplate.device,
                    data,
                    boilerplate.mem_props,
                    usage,
                    vk::SharingMode::EXCLUSIVE,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    boilerplate.command_buffer,
                    boilerplate.command_buffer_reuse_fence,
                    boilerplate.queue,
                )
            }
            fn mapped_buffer_from_data<'a, T: Copy>(
                data: &[T],
                usage: vk::BufferUsageFlags,
                boilerplate: &'a BufferCreationBoilerplate,
            ) -> MappedBuffer<T> {
                GPUBuffer::<T>::new_with_data(
                    boilerplate.device,
                    data,
                    boilerplate.mem_props,
                    usage,
                    vk::SharingMode::EXCLUSIVE,
                    vk::MemoryPropertyFlags::DEVICE_LOCAL,
                    boilerplate.command_buffer,
                    boilerplate.command_buffer_reuse_fence,
                    boilerplate.queue,
                )
            }
            let boilerplate = BufferCreationBoilerplate {
                device: &device,
                mem_props: mem_props,
                command_buffer: setup_command_buffer,
                command_buffer_reuse_fence: setup_commands_reuse_fence,
                queue: present_queue,
            };

            let cbt_scenes: [CBTScene; NUM_ROLLBACK_FRAMES] = std::array::from_fn(|_| {
                let scene_buffer_handles = SceneCPUHandles {
                    root_bisector_vertices: allocated_buffer_from_data(
                        algorithm_data.root_bisector_vertices.as_slice(),
                        BufferUsageFlags::UNIFORM_BUFFER,
                        &boilerplate,
                    ),

                    cbt_interior: mapped_buffer_from_data(
                        algorithm_data.cbt.interior.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),
                    cbt_leaves: mapped_buffer_from_data(
                        leaves_buffer_nonatomic,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),

                    bisector_state_buffer: allocated_buffer_from_data(
                        algorithm_data.bisector_state_buffer.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),

                    bisector_split_command_buffer: allocated_buffer_from_data(
                        split_cmd_buffer_nonatomic,
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),
                    neighbors_buffer: allocated_buffer_from_data(
                        algorithm_data.neighbors_buffer.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),
                    splitting_buffer: allocated_buffer_from_data(
                        algorithm_data.splitting_buffer.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),
                    heapid_buffer: mapped_buffer_from_data(
                        algorithm_data.heapid_buffer.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),

                    allocation_indices_buffer: allocated_buffer_from_data(
                        algorithm_data.allocation_indices_buffer.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),

                    want_split_buffer: allocated_buffer_from_data(
                        algorithm_data.want_split_buffer.as_slice(),
                        vk::BufferUsageFlags::TRANSFER_DST
                            | vk::BufferUsageFlags::INDIRECT_BUFFER
                            | vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),

                    want_merge_buffer: allocated_buffer_from_data(
                        algorithm_data.want_merge_buffer.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),

                    merging_bisector_buffer: allocated_buffer_from_data(
                        algorithm_data.merging_bisector_buffer.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),

                    vertex_buffer: mapped_buffer_from_data(
                        algorithm_data.vertex_buffer.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),
                    curr_id_buffer: allocated_buffer_from_data(
                        curr_id_data.as_slice(),
                        vk::BufferUsageFlags::STORAGE_BUFFER,
                        &boilerplate,
                    ),
                };
                let scene_data_gpu = SceneDataGPU {
                    // written once at initialization
                    root_bisector_vertices: scene_buffer_handles
                        .root_bisector_vertices
                        .device_address(),
                    // our concurrent binary tree
                    // cbt: CBT,
                    cbt_interior: scene_buffer_handles.cbt_interior.device_address(),
                    cbt_leaves: scene_buffer_handles.cbt_leaves.device_address(),

                    // classification state
                    bisector_state_buffer: scene_buffer_handles
                        .bisector_state_buffer
                        .device_address(),

                    // prepare split
                    bisector_split_command_buffer: scene_buffer_handles
                        .bisector_split_command_buffer
                        .device_address(),
                    neighbors_buffer: scene_buffer_handles.neighbors_buffer.device_address(),
                    splitting_buffer: scene_buffer_handles.splitting_buffer.device_address(),
                    heapid_buffer: scene_buffer_handles.heapid_buffer.device_address(),

                    // allocate
                    allocation_indices_buffer: scene_buffer_handles
                        .allocation_indices_buffer
                        .device_address(),

                    // split
                    want_split_buffer: scene_buffer_handles.want_split_buffer.device_address(),

                    // prepare merge
                    want_merge_buffer: scene_buffer_handles.want_merge_buffer.device_address(),

                    // merge
                    merging_bisector_buffer: scene_buffer_handles
                        .merging_bisector_buffer
                        .device_address(),

                    // draw
                    vertex_buffer: scene_buffer_handles.vertex_buffer.device_address(),
                    curr_id_buffer: scene_buffer_handles.curr_id_buffer.device_address(),

                    // integers
                    num_memory_blocks: algorithm_data.cbt.leaves.len() as u32 * BITFIELD_INT_SIZE,
                    base_depth: algorithm_data.base_depth,
                    cbt_depth: algorithm_data.cbt.depth,
                };

                let scene_buffer: AllocatedBuffer<SceneDataGPU> =
                    GPUBuffer::<SceneDataGPU>::new_with_data(
                        &device,
                        std::slice::from_ref(&scene_data_gpu), // &[[SceneDataGPU]]
                        mem_props,
                        vk::BufferUsageFlags::UNIFORM_BUFFER,
                        vk::SharingMode::EXCLUSIVE,
                        vk::MemoryPropertyFlags::DEVICE_LOCAL,
                        setup_command_buffer,
                        setup_commands_reuse_fence,
                        present_queue,
                    );

                let dispatch = DispatchDataGPU {
                    draw_indirect_command: vk::DrawIndirectCommand {
                        vertex_count: (algorithm_data.root_bisector_vertices.len() * 3) as u32,
                        instance_count: 1,
                        first_vertex: 0,
                        first_instance: 0,
                    },

                    dispatch_split_command: linear_dispatch(0),
                    dispatch_allocate_command: linear_dispatch(0),
                    dispatch_prepare_merge_command: linear_dispatch(0),
                    dispatch_vertex_compute_command: linear_dispatch(
                        (algorithm_data.cbt.interior[0] + 64 - 1) / 64,
                    ),

                    remaining_memory_count: algorithm_data
                        .remaining_memory_count
                        .load(Ordering::Relaxed),
                    allocation_counter: 0,
                    want_split_buffer_count: 0,
                    splitting_buffer_count: 0,
                    want_merge_buffer_count: 0,
                    merging_bisector_count: 0,
                    num_allocated_blocks: algorithm_data.cbt.interior[0],
                    debug_data: DebugData::new(),
                };
                let dispatch_buffer: MappedBuffer<DispatchDataGPU> = mapped_buffer_from_data(
                    std::slice::from_ref(&dispatch), // &[DispatchSizeGPU]
                    vk::BufferUsageFlags::STORAGE_BUFFER | vk::BufferUsageFlags::INDIRECT_BUFFER,
                    &boilerplate,
                );
                CBTScene {
                    scene_buffer_handles: scene_buffer_handles,
                    scene_buffer: scene_buffer,
                    dispatch_buffer: dispatch_buffer,
                }
            });
            device.device_wait_idle().expect("Wait idle");

            let pipeline_handles = PipelineHandles {
                reset_pipeline: reset_pipeline,
                dispatch_allocate_pipeline: dispatch_allocate_pipeline,
                allocate_pipeline: allocate_pipeline,
                reduce_pipeline: reduce_pipeline,
                dispatch_split_pipeline: dispatch_split_pipeline,
                split_element_pipeline: split_element_pipeline,
                update_pointers_pipeline: update_pointers_pipeline,
                dispatch_prepare_merge_pipeline: dispatch_prepare_merge_pipeline,
                prepare_merge_pipeline: prepare_merge_pipeline,
                merge_pipeline: merge_pipeline,
                classify_pipeline: classify_pipeline,
                vertex_compute_pipeline: vertex_compute_pipeline,
                validate_pipeline: validate_pipeline,
            };

            self.state = Some(State {
                window: window,
                // instance: instance,
                device: device,
                swapchain: swapchain,
                swapchain_loader: swapchain_loader,
                // setup_command_buffer: setup_command_buffer,
                draw_command_buffers: draw_command_buffers,
                present_queue: present_queue,
                present_image_views: present_image_views,
                present_images: present_images,
                depth_image_views: depth_image_views,
                pipeline: pipeline,
                wire_pipeline: wire_pipeline,
                pipeline_layout: pipeline_layout,
                present_complete_semaphore: present_complete_semaphore,
                render_complete_semaphore: render_complete_semaphore,
                frame_index: 0,
                resolution: surface_resolution,
                command_reuse_fences: command_reuse_fences,
                frame_pace_semaphore: frame_pace_semaphore,
                rendering_mode: RenderingMode::Default,

                // algorithm_data: algorithm_data,
                // setup_commands_reuse_fence: setup_commands_reuse_fence,
                camera: CameraState::new(
                    Vec3::new(0.0, 0.0, 50.0),
                    0.1,
                    0.0,
                    PI / 2.0,
                    Extent2D {
                        width: 1920,
                        height: 1080,
                    },
                    0.1,
                ),

                algorithm_data: algorithm_data,
                divide: false,
                num_iters: 0,
                // swapback: false,
                pipeline_handles: pipeline_handles,
                cbt_scenes: cbt_scenes,
                curr_iter: 0,
            })
        }
    }

    fn device_event(
        &mut self,
        _event_loop: &ActiveEventLoop,
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
                state.camera.pitch -= LOOK_SENSITIVITY * dy as f32;

                let eps = 10e-3;
                if state.camera.pitch < 0.0 + eps {
                    state.camera.pitch = 0.0 + eps;
                } else if state.camera.pitch > PI - eps {
                    state.camera.pitch = PI - eps;
                }
                // println!("Pitch: {:?}", state.camera.pitch);
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
                Key::Character("w") => {
                    state.camera.pos += state.camera.lookdir() * 0.3;
                    // println!("Camera Position: {:?}", state.camera.pos);
                }
                Key::Character("a") => {
                    let left = Vec3::Z.cross(state.camera.lookdir()).normalize();
                    state.camera.pos += left * 0.3;
                    // println!("Camera Position: {:?}", state.camera.pos);
                }
                Key::Character("s") => {
                    state.camera.pos += state.camera.lookdir() * -0.3;
                    // println!("Camera Position: {:?}", state.camera.pos);
                }
                Key::Character("d") => {
                    let right = -Vec3::Z.cross(state.camera.lookdir()).normalize();
                    state.camera.pos += right * 0.3;
                    // println!("Camera Position: {:?}", state.camera.pos);
                }
                Key::Character("r") => {
                    if state.curr_iter == state.num_iters {
                        state.divide = true;
                    } else {
                        println!("viewing wrong iteration");
                    }
                }
                Key::Character("j") => {
                    state.curr_iter = max(
                        0,
                        max(
                            state.num_iters - NUM_ROLLBACK_FRAMES as i32 + 1,
                            state.curr_iter - 1,
                        ),
                    );
                    println!("curr_iter: {:?}", state.curr_iter);
                }
                Key::Character("k") => {
                    state.curr_iter = min(state.num_iters, state.curr_iter + 1);
                    println!("curr_iter: {:?}", state.curr_iter);
                }
                Key::Character("f") => unsafe {
                    /*
                    state.dispatch_buffer.mapped_slice()[0]
                        .draw_indirect_command
                        .vertex_count = ((1 << 17) * 3) as u32;
                    println!(
                        "Draw Indirect Count: {:?}",
                        state.dispatch_buffer.mapped_slice()[0]
                            .draw_indirect_command
                            .vertex_count
                    )*/
                },

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
