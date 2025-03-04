use core::ffi;
use std::{borrow::Cow, cell::Cell, ffi::CString, sync::Mutex};

use crate::{
    Backend, BackendInstance, BindGroup, Buffer, CommandRecorder, CompiledKernel, Fence,
    GpuResource, MappedBuffer, PipelineCache, RecorderSubmitInfo, Semaphore,
};
use ash::{khr, vk, Entry};
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    AllocationError, AllocationSizes, AllocatorDebugSettings,
};
use log::Level;
use shaders::ShaderResourceType;
use thiserror::Error;
use types::{to_static_lifetime, BufferDescriptor, InstanceProperties};

use scopeguard::defer;

pub struct Vulkan;
impl Backend for Vulkan {
    type Buffer = VulkanBuffer;
    type BindGroup = VulkanBindGroup;
    type CommandRecorder = VulkanCommandRecorder;
    type Fence = VulkanFence;
    type Instance = VulkanInstance;
    type Kernel = VulkanKernel;
    type MappedBuffer = VulkanMappedBuffer;
    type PipelineCache = VulkanPipelineCache;
    type Semaphore = VulkanSemaphore;

    type Error = VulkanError;
}
impl Vulkan {
    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = *p_callback_data;
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy()
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy()
        };
        let level = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::Error,
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Level::Info,
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Level::Trace,
            _ => Level::Error,
        };

        log::log!(level, "{message_severity:?}: {message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
    );

        vk::FALSE
    }
    pub fn create_instance(debug: bool) -> Result<VulkanInstance, VulkanError> {
        // TODO: currently, if this errors, memory leaks happen.
        unsafe {
            let entry = Entry::load()?;
            let app_info =
                vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
            let validation_layers = if debug {
                vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
            } else {
                Vec::new()
            };
            let extension_names = if debug {
                vec![ash::ext::debug_utils::NAME.as_ptr()]
            } else {
                Vec::new()
            };
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_layer_names(&validation_layers)
                    .enabled_extension_names(&extension_names),
                None,
            )?;
            let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                .message_severity(
                    vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                        | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                        | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                )
                .message_type(
                    vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                        | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                        | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                )
                .pfn_user_callback(Some(Self::vulkan_debug_callback));
            let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
            let debug_callback =
                debug_utils_loader.create_debug_utils_messenger(&debug_info, None)?;
            // TODO: add phyd filtering
            let (phyd, queue_family_idx) = instance
                .enumerate_physical_devices()?
                .iter()
                .find_map(|phyd| {
                    let queue_families =
                        instance.get_physical_device_queue_family_properties(*phyd);

                    queue_families
                        .iter()
                        .enumerate()
                        .find_map(|(i, q)| {
                            if q.queue_flags
                                .contains(vk::QueueFlags::COMPUTE | vk::QueueFlags::TRANSFER)
                            {
                                Some(i)
                            } else {
                                None
                            }
                        })
                        .map(|i| (*phyd, i as u32))
                })
                .ok_or(VulkanError::NoSupportedDevice)?;
            let mut timeline_semaphore =
                vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
            let mut sync2 =
                vk::PhysicalDeviceSynchronization2FeaturesKHR::default().synchronization2(true);
            // TODO: multiple queues. currently we only use a general queue, but this could potentially be optimized by using special compute queues and special transfer queues
            let queue_priority = 1.0;
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_priorities(std::slice::from_ref(&queue_priority))
                .queue_family_index(queue_family_idx);
            let ext = [khr::synchronization2::NAME.as_ptr()];
            let dev_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&ext)
                .push_next(&mut timeline_semaphore)
                .push_next(&mut sync2);
            let device = instance.create_device(phyd, &dev_create_info, None)?;
            let queue = device.get_device_queue(queue_family_idx, 0);
            Self::from_existing(
                debug,
                entry,
                instance,
                device,
                phyd,
                queue,
                queue_family_idx,
                None,
                Some(debug_callback),
            )
        }
    }
    #[allow(clippy::too_many_arguments)]
    pub fn from_existing(
        debug: bool,
        entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        phyd: vk::PhysicalDevice,
        queue: vk::Queue,
        queue_family_idx: u32,
        renderer_queue_family_idx: Option<u32>,
        debug_callback: Option<vk::DebugUtilsMessengerEXT>,
    ) -> Result<VulkanInstance, VulkanError> {
        unsafe {
            let alloc = Allocator::new(&AllocatorCreateDesc {
                instance: instance.clone(),
                device: device.clone(),
                physical_device: phyd,
                debug_settings: if debug {
                    AllocatorDebugSettings {
                        log_leaks_on_shutdown: true,
                        log_stack_traces: true,
                        log_memory_information: true,
                        ..Default::default()
                    }
                } else {
                    AllocatorDebugSettings::default()
                },
                buffer_device_address: false,
                allocation_sizes: AllocationSizes::default(),
            })?;
            let pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_idx)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            Ok(VulkanInstance {
                sync2_dev: khr::synchronization2::Device::new(&instance, &device),
                entry,
                instance,
                device,
                phyd,
                alloc: Mutex::new(alloc),
                queue,
                queue_family_idx,
                renderer_queue_family_idx,
                pool,
                debug: debug_callback,
            })
        }
    }
}
#[must_use]
#[derive(Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    VulkanRaw(#[from] vk::Result),
    #[error("{0}")]
    VulkanLoadError(#[from] ash::LoadingError),
    #[error("{0}")]
    AllocationError(#[from] gpu_allocator::AllocationError),
    #[error("An unsupported dispatch mode(indirect) was called")]
    DispatchModeUnsupported,
    #[error("{0}")]
    LockError(String),
    #[error("Using compute buffers from an external renderer is currently unsupported")]
    ExternalRendererUnsupported,
    #[error("No supported vulkan device")]
    NoSupportedDevice,
}
impl crate::Error<Vulkan> for VulkanError {
    fn is_out_of_device_memory(&self) -> bool {
        match self {
            Self::VulkanRaw(e) => *e == vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            Self::AllocationError(e) => matches!(e, AllocationError::OutOfMemory),
            _ => false,
        }
    }
    fn is_out_of_host_memory(&self) -> bool {
        match self {
            Self::VulkanRaw(e) => *e == vk::Result::ERROR_OUT_OF_HOST_MEMORY,
            _ => false,
        }
    }
    fn is_timeout(&self) -> bool {
        match self {
            Self::VulkanRaw(e) => *e == vk::Result::TIMEOUT,
            _ => false,
        }
    }
}
#[allow(dead_code)]
pub struct VulkanInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    phyd: vk::PhysicalDevice,
    alloc: Mutex<Allocator>,
    queue: vk::Queue,
    queue_family_idx: u32,
    renderer_queue_family_idx: Option<u32>,
    pool: vk::CommandPool,
    debug: Option<vk::DebugUtilsMessengerEXT>,
    sync2_dev: khr::synchronization2::Device,
}
impl BackendInstance<Vulkan> for VulkanInstance {
    fn get_properties(&mut self) -> InstanceProperties {
        InstanceProperties {
            indirect: false,
            indirect_count: false,
            pipeline_cache: true,
            shader_type: types::ShaderTarget::Spirv {
                version: types::SpirvVersion::V1_0,
            },
        }
    }
    fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &shaders::ShaderReflectionInfo,
        cache: Option<&mut VulkanPipelineCache>,
    ) -> Result<<Vulkan as Backend>::Kernel, <Vulkan as Backend>::Error> {
        unsafe {
            let err = Cell::new(true);
            let shader_create_info = &vk::ShaderModuleCreateInfo::default();
            let ptr = binary.as_ptr() as *const u32;
            let shader = if ptr.is_aligned() {
                self.device.create_shader_module(
                    &shader_create_info.code(std::slice::from_raw_parts(ptr, binary.len() / 4)),
                    None,
                )?
            } else {
                let mut v = Vec::<u32>::with_capacity(binary.len() / 4);
                binary
                    .as_ptr()
                    .copy_to(v.as_mut_ptr() as *mut u8, binary.len());
                v.set_len(binary.len() / 4);
                self.device
                    .create_shader_module(&shader_create_info.code(&v), None)?
            };
            defer! {
                if err.get() {
                    self.device.destroy_shader_module(shader, None);
                }
            }
            let bindings: Vec<_> = reflection
                .resources
                .iter()
                .enumerate()
                .map(|(i, res)| {
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_count(1)
                        .descriptor_type(match res {
                            ShaderResourceType::Buffer => vk::DescriptorType::STORAGE_BUFFER,
                            ShaderResourceType::UniformBuffer => vk::DescriptorType::UNIFORM_BUFFER,
                            ShaderResourceType::Unknown => panic!("Unknown shader binding"),
                        })
                        .stage_flags(vk::ShaderStageFlags::COMPUTE)
                })
                .collect();
            let desc_set_layout_create =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let descriptor_set_layout = self
                .device
                .create_descriptor_set_layout(&desc_set_layout_create, None)?;
            defer! {
                if err.get() {
                    self.device.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
            }
            let pipeline_layout = self.device.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[descriptor_set_layout]),
                None,
            )?;
            let _cache_lock;
            let cache = if let Some(c) = cache {
                let lock = c
                    .inner
                    .lock()
                    .map_err(|e| VulkanError::LockError(e.to_string()))?;
                let cache = *lock;
                _cache_lock = Some(lock);
                cache
            } else {
                _cache_lock = None;
                vk::PipelineCache::null()
            };
            let entry = CString::new(reflection.entry_name.clone()).unwrap();
            let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(shader)
                        .name(&entry),
                )
                .layout(pipeline_layout);
            let pipeline = self
                .device
                .create_compute_pipelines(cache, &[pipeline_create_info], None)
                .map_err(|e| e.1)?[0];
            drop(_cache_lock);
            err.set(false);
            Ok(VulkanKernel {
                shader,
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pools: Vec::new(),
            })
        }
    }
    fn destroy_kernel(
        &mut self,
        kernel: <Vulkan as Backend>::Kernel,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            for pool in kernel.descriptor_pools {
                self.device.destroy_descriptor_pool(pool.pool, None);
            }
            self.device
                .destroy_descriptor_set_layout(kernel.descriptor_set_layout, None);
            self.device.destroy_pipeline(kernel.pipeline, None);
            self.device
                .destroy_pipeline_layout(kernel.pipeline_layout, None);
            self.device.destroy_shader_module(kernel.shader, None);
            Ok(())
        }
    }
    fn create_buffer(
        &mut self,
        alloc_info: &types::BufferDescriptor,
    ) -> Result<<Vulkan as Backend>::Buffer, <Vulkan as Backend>::Error> {
        unsafe {
            let err = Cell::new(true);
            if alloc_info.visible_to_renderer {
                return Err(VulkanError::ExternalRendererUnsupported);
            }
            let queue_family_indices = [self.queue_family_idx]; // This would need to change to support external renderers
            let create_info = vk::BufferCreateInfo::default()
                .size(alloc_info.size)
                .sharing_mode(if alloc_info.visible_to_renderer {
                    vk::SharingMode::CONCURRENT
                } else {
                    vk::SharingMode::EXCLUSIVE
                })
                .queue_family_indices(&queue_family_indices)
                .usage({
                    use vk::BufferUsageFlags as F;
                    let mut flags = F::STORAGE_BUFFER;
                    if alloc_info.transfer_src {
                        flags |= F::TRANSFER_SRC
                    };
                    if alloc_info.transfer_dst {
                        flags |= F::TRANSFER_DST
                    }
                    if alloc_info.indirect_capable {
                        flags |= F::INDIRECT_BUFFER
                    }
                    flags
                });
            let buffer = self.device.create_buffer(&create_info, None)?;
            defer! {
                if err.get() {
                    self.device.destroy_buffer(buffer, None);
                }
            }
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            use types::MemoryType::*;
            let mut allocation = self
                .alloc
                .lock()
                .map_err(|e| VulkanError::LockError(e.to_string()))?
                .allocate(&AllocationCreateDesc {
                    name: "",
                    requirements,
                    location: match alloc_info.memory_type {
                        Any => gpu_allocator::MemoryLocation::Unknown,
                        GpuOnly => gpu_allocator::MemoryLocation::GpuOnly,
                        Upload => gpu_allocator::MemoryLocation::CpuToGpu,
                        Download => gpu_allocator::MemoryLocation::GpuToCpu,
                        UploadDownload => gpu_allocator::MemoryLocation::CpuToGpu,
                    },
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })?;
            let alloc_ptr = &mut allocation as *mut Allocation;
            defer! {
                if err.get() {
                    // TODO: is this undefined behavior? Lets find out!
                    self.alloc.lock().unwrap().free(std::mem::take(&mut *alloc_ptr)).unwrap();
                }
            }
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            err.set(false);
            Ok(VulkanBuffer {
                buffer,
                allocation,
                create_info: *alloc_info,
            })
        }
    }
    fn destroy_buffer(
        &mut self,
        buffer: <Vulkan as Backend>::Buffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.device.destroy_buffer(buffer.buffer, None);
            self.alloc
                .lock()
                .map_err(|e| VulkanError::LockError(e.to_string()))?
                .free(buffer.allocation)?;
            Ok(())
        }
    }
    fn create_pipeline_cache(
        &mut self,
        initial_data: &[u8],
    ) -> Result<<Vulkan as Backend>::PipelineCache, <Vulkan as Backend>::Error> {
        unsafe {
            let create_info = vk::PipelineCacheCreateInfo::default().initial_data(initial_data);
            let pc = self.device.create_pipeline_cache(&create_info, None)?;
            Ok(VulkanPipelineCache {
                inner: Mutex::new(pc),
            })
        }
    }
    fn destroy_pipeline_cache(
        &mut self,
        cache: <Vulkan as Backend>::PipelineCache,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            let lock = cache
                .inner
                .lock()
                .map_err(|e| VulkanError::LockError(e.to_string()))?;
            self.device.destroy_pipeline_cache(*lock, None);
            drop(lock);
            Ok(())
        }
    }
    fn get_pipeline_cache_data(
        &mut self,
        cache: <Vulkan as Backend>::PipelineCache,
    ) -> Result<Vec<u8>, <Vulkan as Backend>::Error> {
        unsafe {
            let lock = cache
                .inner
                .lock()
                .map_err(|e| VulkanError::LockError(e.to_string()))?;
            let data = self.device.get_pipeline_cache_data(*lock)?;
            drop(lock);
            Ok(data)
        }
    }
    fn create_bind_group(
        &mut self,
        kernel: &mut VulkanKernel,
        resources: &mut [crate::GpuResource<Vulkan>],
    ) -> Result<VulkanBindGroup, VulkanError> {
        let mut pool_idx = None;
        for (i, pool) in kernel.descriptor_pools.iter_mut().enumerate() {
            if pool.max_size > pool.current_size {
                pool_idx = Some(i);
                break;
            }
        }
        if pool_idx.is_none() {
            pool_idx = Some(kernel.descriptor_pools.len());
            let next_size = kernel
                .descriptor_pools
                .last()
                .map(|s| s.max_size * 2)
                .unwrap_or(8)
                .next_power_of_two();
            let mut num_buffers = 0;
            let mut num_uniform_buffers = 0;
            for res in resources.iter() {
                match res {
                    GpuResource::Buffer {
                        buffer:
                            VulkanBuffer {
                                create_info: BufferDescriptor { uniform, .. },
                                ..
                            },
                        ..
                    } => {
                        if *uniform {
                            num_uniform_buffers += 1;
                        } else {
                            num_buffers += 1;
                        }
                    }
                }
            }
            let mut sizes = vec![];
            if num_buffers > 0 {
                sizes.push(
                    vk::DescriptorPoolSize::default()
                        .descriptor_count(num_buffers * next_size)
                        .ty(vk::DescriptorType::STORAGE_BUFFER),
                );
            }
            if num_uniform_buffers > 0 {
                sizes.push(
                    vk::DescriptorPoolSize::default()
                        .descriptor_count(num_uniform_buffers * next_size)
                        .ty(vk::DescriptorType::UNIFORM_BUFFER),
                );
            }
            unsafe {
                let create_info = vk::DescriptorPoolCreateInfo::default()
                    .max_sets(next_size)
                    .pool_sizes(&sizes)
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
                let pool = self.device.create_descriptor_pool(&create_info, None)?;
                kernel.descriptor_pools.push(DescriptorPoolData {
                    pool,
                    max_size: next_size,
                    current_size: 0,
                });
            }
        }
        let pool_idx = pool_idx.unwrap();
        unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(kernel.descriptor_pools[pool_idx].pool)
                .set_layouts(std::slice::from_ref(&kernel.descriptor_set_layout));
            let descriptor_set = self.device.allocate_descriptor_sets(&alloc_info)?[0];
            let mut writes = Vec::with_capacity(resources.len());
            let mut buffer_infos = Vec::new();
            for (i, resource) in resources.iter().enumerate() {
                let mut write = vk::WriteDescriptorSet::default()
                    .dst_set(descriptor_set)
                    .descriptor_count(1)
                    .dst_binding(i as u32)
                    .descriptor_type(match resource {
                        GpuResource::Buffer {
                            buffer:
                                VulkanBuffer {
                                    create_info: BufferDescriptor { uniform, .. },
                                    ..
                                },
                            ..
                        } => {
                            if *uniform {
                                vk::DescriptorType::UNIFORM_BUFFER
                            } else {
                                vk::DescriptorType::STORAGE_BUFFER
                            }
                        }
                    });
                match resource {
                    GpuResource::Buffer {
                        buffer,
                        offset,
                        size,
                    } => {
                        buffer_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(buffer.buffer)
                                .offset(*offset)
                                .range(*size),
                        );
                        // TODO: fix (potentially) Undefined behavior
                        write = write.buffer_info(std::slice::from_ref(to_static_lifetime(
                            &buffer_infos[buffer_infos.len() - 1],
                        )));
                    }
                }
                writes.push(write);
            }
            self.device.update_descriptor_sets(&writes, &[]);
            Ok(VulkanBindGroup {
                inner: descriptor_set,
                pool_idx: pool_idx as u32,
            })
        }
    }
    fn destroy_bind_group(
        &mut self,
        kernel: &mut <Vulkan as Backend>::Kernel,
        bind_group: VulkanBindGroup,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.device.free_descriptor_sets(
                kernel.descriptor_pools[bind_group.pool_idx as usize].pool,
                &[bind_group.inner],
            )?;
        }
        Ok(())
    }
    fn create_recorders(
        &mut self,
        num: u32,
    ) -> Result<Vec<<Vulkan as Backend>::CommandRecorder>, <Vulkan as Backend>::Error> {
        unsafe {
            let create_info = vk::CommandBufferAllocateInfo::default()
                .command_pool(self.pool)
                .command_buffer_count(num)
                .level(vk::CommandBufferLevel::PRIMARY);
            let cbs = self.device.allocate_command_buffers(&create_info)?;
            Ok(cbs
                .into_iter()
                .map(|cb| VulkanCommandRecorder { inner: cb })
                .collect())
        }
    }
    fn destroy_recorders(
        &mut self,
        recorders: Vec<<Vulkan as Backend>::CommandRecorder>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let cb: Vec<vk::CommandBuffer> = recorders.into_iter().map(|cb| cb.inner).collect();
        unsafe {
            self.device.free_command_buffers(self.pool, &cb);
        }
        Ok(())
    }
    fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<Vulkan>],
        fence: Option<&mut VulkanFence>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let mut semaphore_infos = Vec::new();
        let mut cbs = Vec::new();
        for submit in infos.iter() {
            semaphore_infos.extend(submit.wait_semaphores.iter().map(|(s, v)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(s.inner)
                    .value(*v)
                    .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            }));
            cbs.extend(
                submit
                    .command_recorders
                    .iter()
                    .map(|a| vk::CommandBufferSubmitInfo::default().command_buffer(a.inner)),
            );
        }
        let signal_start = semaphore_infos.len();
        for submit in infos.iter() {
            semaphore_infos.extend(submit.out_semaphores.iter().map(|(s, v)| {
                vk::SemaphoreSubmitInfo::default()
                    .semaphore(s.inner)
                    .value(*v)
                    .stage_mask(vk::PipelineStageFlags2::COMPUTE_SHADER)
            }));
        }
        let mut submits = Vec::new();
        let mut cb_idx = 0;
        let mut wait_idx = 0;
        let mut signal_idx = signal_start;
        for submit in infos.iter() {
            submits.push(
                vk::SubmitInfo2::default()
                    .command_buffer_infos(&cbs[cb_idx..cb_idx + submit.command_recorders.len()])
                    .wait_semaphore_infos(
                        &semaphore_infos[wait_idx..wait_idx + submit.wait_semaphores.len()],
                    )
                    .signal_semaphore_infos(
                        &semaphore_infos[signal_idx..signal_idx + submit.out_semaphores.len()],
                    ),
            );
            cb_idx += submit.command_recorders.len();
            wait_idx += submit.wait_semaphores.len();
            signal_idx += submit.out_semaphores.len();
        }
        unsafe {
            self.sync2_dev.queue_submit2(
                self.queue,
                &submits,
                if let Some(f) = fence {
                    f.inner
                } else {
                    vk::Fence::null()
                },
            )?;
        }
        Ok(())
    }
    fn clear_recorders(
        &mut self,
        recorders: &mut [&mut VulkanCommandRecorder],
    ) -> Result<(), VulkanError> {
        unsafe {
            for cb in recorders {
                self.device
                    .reset_command_buffer(cb.inner, vk::CommandBufferResetFlags::empty())?;
            }
        }
        Ok(())
    }
    fn wait_for_idle(&mut self) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.device.queue_wait_idle(self.queue)?;
            Ok(())
        }
    }
    fn map_buffer(
        &mut self,
        buffer: &mut <Vulkan as Backend>::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<<Vulkan as Backend>::MappedBuffer, <Vulkan as Backend>::Error> {
        unsafe {
            /*let ptr = self.device.map_memory(
                buffer.allocation.memory(),
                buffer.allocation.offset() + offset,
                size,
                vk::MemoryMapFlags::empty(),
            )?;*/

            // Memory is automatically mapped by gpu-allocator

            Ok(VulkanMappedBuffer {
                slice: std::slice::from_raw_parts_mut(
                    buffer
                        .allocation
                        .mapped_ptr()
                        .unwrap()
                        .byte_add(offset as usize)
                        .as_ptr() as *mut u8,
                    size as usize,
                ),
                _buffer_offset: offset,
            })
        }
    }
    fn flush_mapped_buffer(
        &self,
        _buffer: &mut <Vulkan as Backend>::Buffer,
        _map: &mut <Vulkan as Backend>::MappedBuffer,
    ) -> Result<(), VulkanError> {
        /*if buffer.create_info.needs_flush {
            unsafe {
                let range = vk::MappedMemoryRange::default()
                    .memory(buffer.allocation.memory())
                    .offset(map.buffer_offset + buffer.allocation.offset())
                    .size(map.slice.len() as u64);
                self.device.flush_mapped_memory_ranges(&[range])?;
            }
        }*/

        // Memory is always coherent with gpu-allocator

        Ok(())
    }
    fn update_mapped_buffer(
        &self,
        _buffer: &mut <Vulkan as Backend>::Buffer,
        _map: &mut <Vulkan as Backend>::MappedBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        /*if buffer.create_info.needs_flush {
            unsafe {
                let range = vk::MappedMemoryRange::default()
                    .memory(buffer.allocation.memory())
                    .offset(map.buffer_offset + buffer.allocation.offset())
                    .size(map.slice.len() as u64);
                self.device.invalidate_mapped_memory_ranges(&[range])?;
            }
        }*/

        // Memory is always coherent with gpu-allocator

        Ok(())
    }
    fn unmap_buffer(
        &mut self,
        _buffer: &mut <Vulkan as Backend>::Buffer,
        _map: <Vulkan as Backend>::MappedBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        // unsafe {self.device.unmap_memory(buffer.allocation.memory());}

        // Memory is always mapped with gpu-allocator

        Ok(())
    }
    fn write_buffer(
        &mut self,
        buffer: &mut <Vulkan as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let mut b = self.map_buffer(buffer, offset, data.len() as u64)?;
        b.slice.copy_from_slice(data);
        self.flush_mapped_buffer(buffer, &mut b)?;
        self.unmap_buffer(buffer, b)?;
        Ok(())
    }
    fn create_fence(&mut self) -> std::result::Result<VulkanFence, VulkanError> {
        unsafe {
            Ok(VulkanFence {
                inner: self.device.create_fence(
                    &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::empty()),
                    None,
                )?,
            })
        }
    }
    fn destroy_fence(&mut self, fence: VulkanFence) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.device.destroy_fence(fence.inner, None);
            Ok(())
        }
    }
    fn wait_for_fences(
        &mut self,
        fences: &mut [&mut <Vulkan as Backend>::Fence],
        all: bool,
        timeout_seconds: f32,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            let fences: Vec<_> = fences.iter().map(|f| f.inner).collect();
            self.device.wait_for_fences(
                &fences,
                all,
                (timeout_seconds * 1_000_000_000.0) as u64,
            )?;
            Ok(())
        }
    }
    fn create_semaphore(
        &mut self,
        timeline: bool,
    ) -> std::result::Result<VulkanSemaphore, VulkanError> {
        unsafe {
            let mut ext = vk::SemaphoreTypeCreateInfo::default().semaphore_type(if timeline {
                vk::SemaphoreType::TIMELINE
            } else {
                vk::SemaphoreType::BINARY
            });
            let create_info = vk::SemaphoreCreateInfo::default()
                .flags(vk::SemaphoreCreateFlags::empty())
                .push_next(&mut ext);
            Ok(VulkanSemaphore {
                inner: self.device.create_semaphore(&create_info, None)?,
                _timeline: timeline,
            })
        }
    }
    fn destroy_semaphore(
        &mut self,
        semaphore: VulkanSemaphore,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.device.destroy_semaphore(semaphore.inner, None);
        }
        Ok(())
    }
    fn wait_for_semaphores(
        &mut self,
        semaphores: &mut [(&mut <Vulkan as Backend>::Semaphore, u64)],
        all: bool,
        timeout: f32,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let sems: Vec<_> = semaphores.iter().map(|a| a.0.inner).collect();
        let values: Vec<_> = semaphores.iter().map(|a| a.1).collect();
        let wait_info = vk::SemaphoreWaitInfo::default()
            .flags(if all {
                vk::SemaphoreWaitFlags::empty()
            } else {
                vk::SemaphoreWaitFlags::ANY
            })
            .semaphores(&sems)
            .values(&values);
        unsafe {
            self.device
                .wait_semaphores(&wait_info, (timeout * 1_000_000_000.0) as u64)?;
        }
        Ok(())
    }
}
impl VulkanInstance {
    pub fn destroy(mut self) {
        unsafe {
            self.alloc
                .get_mut()
                .unwrap()
                .report_memory_leaks(log::Level::Error);
            drop(self.alloc);
            self.device.destroy_command_pool(self.pool, None);
            self.device.destroy_device(None);
            if let Some(debug) = self.debug {
                ash::ext::debug_utils::Instance::new(&self.entry, &self.instance)
                    .destroy_debug_utils_messenger(debug, None);
            }
            self.instance.destroy_instance(None);
        }
    }
}
pub struct VulkanKernel {
    pub shader: vk::ShaderModule,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_pools: Vec<DescriptorPoolData>,
}
impl CompiledKernel<Vulkan> for VulkanKernel {}
pub struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub create_info: types::BufferDescriptor,
}
impl Buffer<Vulkan> for VulkanBuffer {}
pub struct VulkanMappedBuffer {
    slice: &'static mut [u8],
    _buffer_offset: u64,
}
impl MappedBuffer<Vulkan> for VulkanMappedBuffer {
    fn readable(&mut self) -> &[u8] {
        self.slice
    }
    fn writable(&mut self) -> &mut [u8] {
        self.slice
    }
}
pub struct VulkanCommandRecorder {
    inner: vk::CommandBuffer,
}
impl VulkanCommandRecorder {}
impl CommandRecorder<Vulkan> for VulkanCommandRecorder {
    fn begin(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
        allow_resubmits: bool,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.begin_command_buffer(
                self.inner,
                &vk::CommandBufferBeginInfo::default().flags(if allow_resubmits {
                    vk::CommandBufferUsageFlags::empty()
                } else {
                    vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT
                }),
            )?;
            Ok(())
        }
    }
    fn end(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.end_command_buffer(self.inner)?;
            Ok(())
        }
    }
    fn copy_buffer(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
        src_buffer: &mut <Vulkan as Backend>::Buffer,
        dst_buffer: &mut <Vulkan as Backend>::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
        _sync: crate::CommandSynchronization<Vulkan>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.cmd_copy_buffer(
                self.inner,
                src_buffer.buffer,
                dst_buffer.buffer,
                &[vk::BufferCopy::default()
                    .src_offset(src_offset)
                    .dst_offset(dst_offset)
                    .size(size)],
            );
        }
        // TODO: Do sync stuff
        todo!()
    }
    fn dispatch_kernel(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
        shader: &mut <Vulkan as Backend>::Kernel,
        descriptor_set: &mut <Vulkan as Backend>::BindGroup,
        push_constants: &[u8],
        workgroup_dims: [u32; 3],
        _sync: crate::CommandSynchronization<Vulkan>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.cmd_bind_pipeline(
                self.inner,
                vk::PipelineBindPoint::COMPUTE,
                shader.pipeline,
            );
            instance.device.cmd_bind_descriptor_sets(
                self.inner,
                vk::PipelineBindPoint::COMPUTE,
                shader.pipeline_layout,
                0,
                &[descriptor_set.inner],
                &[],
            );
            if !push_constants.is_empty() {
                instance.device.cmd_push_constants(
                    self.inner,
                    shader.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            instance.device.cmd_dispatch(
                self.inner,
                workgroup_dims[0],
                workgroup_dims[1],
                workgroup_dims[2],
            );
        }
        // TODO: Do sync stuff
        Ok(())
    }
    fn dispatch_kernel_indirect(
        &mut self,
        _instance: &mut <Vulkan as Backend>::Instance,
        _shader: &mut <Vulkan as Backend>::Kernel,
        _descriptor_set: &mut <Vulkan as Backend>::BindGroup,
        _push_constants: &[u8],
        _indirect_buffer: &mut <Vulkan as Backend>::Buffer,
        _buffer_offset: u64,
        _num_dispatches: u64,
        _validate_dispatches: bool,
        _sync: crate::CommandSynchronization<Vulkan>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        Err(VulkanError::DispatchModeUnsupported)
    }
    fn dispatch_kernel_indirect_count(
        &mut self,
        _instance: &mut <Vulkan as Backend>::Instance,
        _shader: &mut <Vulkan as Backend>::Kernel,
        _descriptor_set: &mut <Vulkan as Backend>::BindGroup,
        _push_constants: &[u8],
        _indirect_buffer: &mut <Vulkan as Backend>::Buffer,
        _buffer_offset: u64,
        _count_buffer: &mut <Vulkan as Backend>::Buffer,
        _count_offset: u64,
        _max_dispatches: u64,
        _validate_dispatches: bool,
        _sync: crate::CommandSynchronization<Vulkan>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        Err(VulkanError::DispatchModeUnsupported)
    }
}
pub struct DescriptorPoolData {
    pub pool: vk::DescriptorPool,
    pub max_size: u32,
    pub current_size: u32,
}
pub struct VulkanBindGroup {
    inner: vk::DescriptorSet,
    pool_idx: u32,
}
impl BindGroup<Vulkan> for VulkanBindGroup {}
pub struct VulkanPipelineCache {
    inner: Mutex<vk::PipelineCache>,
}
impl PipelineCache<Vulkan> for VulkanPipelineCache {}
pub struct VulkanSemaphore {
    inner: vk::Semaphore,
    _timeline: bool,
}
impl Semaphore<Vulkan> for VulkanSemaphore {
    fn get_timeline_counter(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
    ) -> Result<u64, <Vulkan as Backend>::Error> {
        unsafe { Ok(instance.device.get_semaphore_counter_value(self.inner)?) }
    }
    fn signal(
        &mut self,
        instance: &mut VulkanInstance,
        signal: u64,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(self.inner)
                    .value(signal),
            )?;
            Ok(())
        }
    }
}
pub struct VulkanFence {
    inner: vk::Fence,
}
impl Fence<Vulkan> for VulkanFence {
    fn get_signalled(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
    ) -> Result<bool, <Vulkan as Backend>::Error> {
        unsafe { Ok(instance.device.get_fence_status(self.inner)?) }
    }
    fn reset(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.reset_fences(&[self.inner])?;
            Ok(())
        }
    }
}

#[cfg(test)]
mod tests {
    use shaders::ShaderReflectionInfo;

    use super::*;
    fn create_storage_buf(instance: &mut VulkanInstance, data: &[u8]) -> VulkanBuffer {
        let mut buf = instance
            .create_buffer(&BufferDescriptor {
                size: data.len() as u64,
                memory_type: types::MemoryType::UploadDownload,
                mapped_at_creation: false,
                visible_to_renderer: false,
                indirect_capable: false,
                transfer_src: false,
                transfer_dst: false,
                uniform: false,
                needs_flush: true,
            })
            .unwrap();
        let mut mapped = instance.map_buffer(&mut buf, 0, data.len() as u64).unwrap();
        mapped.writable().copy_from_slice(data);
        instance.flush_mapped_buffer(&mut buf, &mut mapped).unwrap();
        instance.unmap_buffer(&mut buf, mapped).unwrap();
        buf
    }
    fn read_buf(instance: &mut VulkanInstance, buf: &mut VulkanBuffer, out_data: &mut [u8]) {
        let mapped = instance.map_buffer(buf, 0, out_data.len() as u64).unwrap();
        out_data.copy_from_slice(mapped.slice);
        instance.unmap_buffer(buf, mapped).unwrap();
    }
    #[test]
    fn vulkan_main_test() {
        env_logger::init();
        let mut instance = Vulkan::create_instance(true).unwrap();
        let mut cache = instance.create_pipeline_cache(&[]).unwrap();
        let mut fence = instance.create_fence().unwrap();
        let mut fun_semaphore = instance.create_semaphore(true).unwrap();
        let mut kernel = instance
            .compile_kernel(
                include_bytes!("test.spirv"),
                &ShaderReflectionInfo {
                    workgroup_size: [1, 1, 1],
                    entry_name: "main".to_owned(),
                    resources: vec![
                        ShaderResourceType::Buffer,
                        ShaderResourceType::Buffer,
                        ShaderResourceType::Buffer,
                    ],
                    push_constant_len: 0,
                },
                Some(&mut cache),
            )
            .unwrap();
        let uniform_buf = instance
            .create_buffer(&BufferDescriptor {
                size: 16,
                memory_type: types::MemoryType::Upload,
                mapped_at_creation: false,
                visible_to_renderer: false,
                indirect_capable: false,
                transfer_src: false,
                transfer_dst: false,
                uniform: true,
                needs_flush: true,
            })
            .unwrap();
        let mut sb1 = create_storage_buf(&mut instance, bytemuck::bytes_of(&[5u32, 0, 0, 0]));
        let mut sb2 = create_storage_buf(&mut instance, bytemuck::bytes_of(&[8u32, 0, 0, 0]));
        let mut sbout = create_storage_buf(&mut instance, bytemuck::bytes_of(&[2u32, 0, 0, 0]));
        let mut bind_group = instance
            .create_bind_group(
                &mut kernel,
                &mut [
                    GpuResource::buffer(&mut sb1, 0, 16),
                    GpuResource::buffer(&mut sb2, 0, 16),
                    GpuResource::buffer(&mut sbout, 0, 16),
                ],
            )
            .unwrap();

        let mut recorder = instance
            .create_recorders(1)
            .unwrap()
            .into_iter()
            .next()
            .unwrap();

        recorder.begin(&mut instance, true).unwrap();
        recorder
            .dispatch_kernel(
                &mut instance,
                &mut kernel,
                &mut bind_group,
                &[],
                [1, 1, 1],
                crate::CommandSynchronization {
                    waits: &mut [],
                    resources: &mut [],
                    out_fence: None,
                    out_semaphore: None,
                },
            )
            .unwrap();
        recorder.end(&mut instance).unwrap();
        instance
            .submit_recorders(
                std::slice::from_mut(&mut RecorderSubmitInfo {
                    command_recorders: &mut [&mut recorder],
                    wait_semaphores: &mut [],
                    out_semaphores: &mut [(&mut fun_semaphore, 3)],
                }),
                Some(&mut fence),
            )
            .unwrap();
        instance
            .wait_for_fences(&mut [&mut fence], true, 1.0)
            .unwrap();
        instance
            .wait_for_semaphores(&mut [(&mut fun_semaphore, 2)], true, 1.0)
            .unwrap();

        let mut res = [3u32, 0, 0, 0];
        read_buf(&mut instance, &mut sbout, bytemuck::bytes_of_mut(&mut res));
        if res[0] != 13 {
            panic!("Expected 13, got {}", res[0]);
        }

        instance.destroy_recorders(vec![recorder]).unwrap();

        instance.destroy_semaphore(fun_semaphore).unwrap();
        instance.destroy_fence(fence).unwrap();
        instance
            .destroy_bind_group(&mut kernel, bind_group)
            .unwrap();
        instance.destroy_kernel(kernel).unwrap();
        instance.destroy_buffer(uniform_buf).unwrap();
        instance.destroy_buffer(sb1).unwrap();
        instance.destroy_buffer(sb2).unwrap();
        instance.destroy_buffer(sbout).unwrap();
        instance.destroy_pipeline_cache(cache).unwrap();
        instance.destroy();
    }
}
