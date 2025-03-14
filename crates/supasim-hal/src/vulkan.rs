use core::ffi;
use std::{borrow::Cow, cell::Cell, ffi::CString, sync::Mutex};

use crate::{
    Backend, BackendInstance, BindGroup, Buffer, CommandRecorder, CompiledKernel, Fence,
    GpuCommand, GpuResource, MappedBuffer, PipelineCache, RecorderSubmitInfo, Semaphore,
};
use ash::{Entry, khr, vk};
use gpu_allocator::{
    AllocationError, AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
};
use log::Level;
use thiserror::Error;
use types::{
    BufferDescriptor, InstanceProperties, ShaderReflectionInfo, ShaderResourceType,
    to_static_lifetime,
};

use scopeguard::defer;

#[derive(Debug, Clone)]
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
        let callback_data = unsafe { *p_callback_data };
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
        };
        let level = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::Error,
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Level::Info,
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Level::Trace,
            _ => Level::Error,
        };

        log::log!(
            level,
            "{message_severity:?}: {message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
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
                command_pool: pool,
                unused_command_buffers: Vec::new(),
                unused_events: Vec::new(),
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
    #[error(
        "A command recorder was submitted that would've required signalled semaphores for a complex DAG structure"
    )]
    SemaphoreSignalInDag,
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
    command_pool: vk::CommandPool,
    unused_command_buffers: Vec<vk::CommandBuffer>,
    unused_events: Vec<vk::Event>,
    debug: Option<vk::DebugUtilsMessengerEXT>,
    sync2_dev: khr::synchronization2::Device,
}
impl VulkanInstance {
    pub fn get_command_buffer(&mut self) -> Result<vk::CommandBuffer, VulkanError> {
        match self.unused_command_buffers.pop() {
            Some(c) => Ok(c),
            None => unsafe {
                Ok(self
                    .device
                    .allocate_command_buffers(
                        &vk::CommandBufferAllocateInfo::default()
                            .command_pool(self.command_pool)
                            .level(vk::CommandBufferLevel::PRIMARY)
                            .command_buffer_count(1),
                    )?
                    .into_iter()
                    .next()
                    .unwrap())
            },
        }
    }
}
impl BackendInstance<Vulkan> for VulkanInstance {
    fn get_properties(&mut self) -> InstanceProperties {
        InstanceProperties {
            needs_explicit_sync: true,
            indirect: false,
            pipeline_cache: true,
            shader_type: types::ShaderTarget::Spirv {
                version: types::SpirvVersion::V1_0,
            },
        }
    }
    fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &ShaderReflectionInfo,
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
        cache: &mut <Vulkan as Backend>::PipelineCache,
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
        resources: &[crate::GpuResource<Vulkan>],
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
            defer! {}
            let mut bg = VulkanBindGroup {
                inner: descriptor_set,
                pool_idx: pool_idx as u32,
            };
            if let Err(e) = self.update_bind_group(&mut bg, kernel, resources) {
                let _ = self.device.free_descriptor_sets(
                    kernel.descriptor_pools[pool_idx].pool,
                    &[descriptor_set],
                );
                Err(e)
            } else {
                Ok(bg)
            }
        }
    }
    fn update_bind_group(
        &mut self,
        bg: &mut <Vulkan as Backend>::BindGroup,
        _kernel: &mut <Vulkan as Backend>::Kernel,
        resources: &[GpuResource<Vulkan>],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let mut writes = Vec::with_capacity(resources.len());
        let mut buffer_infos = Vec::new();
        for (i, resource) in resources.iter().enumerate() {
            let mut write = vk::WriteDescriptorSet::default()
                .dst_set(bg.inner)
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
                    write = write.buffer_info(std::slice::from_ref(unsafe {
                        to_static_lifetime(&buffer_infos[buffer_infos.len() - 1])
                    }));
                }
            }
            writes.push(write);
        }
        unsafe {
            self.device.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
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
    fn create_recorder(
        &mut self,
        allow_resubmits: bool,
    ) -> Result<<Vulkan as Backend>::CommandRecorder, <Vulkan as Backend>::Error> {
        Ok(VulkanCommandRecorder {
            cbs: Vec::new(),
            allow_resubmits,
            used_events: Vec::new(),
        })
    }
    fn destroy_recorder(
        &mut self,
        recorder: <Vulkan as Backend>::CommandRecorder,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        self.unused_events.extend(&recorder.used_events);
        self.unused_command_buffers
            .extend(recorder.cbs.into_iter().map(|a| a.cb));
        Ok(())
    }
    fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<Vulkan>],
        fence: Option<&mut VulkanFence>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let mut cbs = Vec::new();
        let mut semaphores: Vec<vk::Semaphore> = Vec::new();
        let mut pipeline_stage_flags = Vec::new();
        let mut ensure_enough_flags = |num| {
            while pipeline_stage_flags.len() < num {
                pipeline_stage_flags.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
        };
        let mut values = Vec::new();
        let mut ensure_enough_values = |num| {
            while values.len() < num {
                values.push(1);
            }
        };
        let mut timeline_ext = Vec::new();
        for info in &*infos {
            for &event in &info.command_recorder.used_events {
                unsafe {
                    self.device.reset_event(event)?;
                }
            }
            let num_tails = info
                .command_recorder
                .cbs
                .iter()
                .filter_map(|a| if a.is_tail { Some(()) } else { None })
                .count();
            if num_tails == 0 {
                return Ok(());
            }
            if num_tails > 1 && !info.signal_semaphores.is_empty() {
                return Err(VulkanError::SemaphoreSignalInDag);
            }
            for a in &info.command_recorder.cbs {
                cbs.push(a.cb);
                let num_wait_semaphores = a.wait_semaphores.len()
                    + if a.is_head {
                        info.wait_semaphores.len()
                    } else {
                        0
                    };
                let num_signal_semaphores = a.signal_semaphores.len()
                    + if a.is_tail {
                        info.signal_semaphores.len()
                    } else {
                        0
                    };
                ensure_enough_values(num_wait_semaphores);
                ensure_enough_values(num_signal_semaphores);
                ensure_enough_flags(num_signal_semaphores);
                for &s in &a.wait_semaphores {
                    semaphores.push(s);
                }
                if a.is_head {
                    semaphores.extend(info.wait_semaphores.iter().map(|a| a.inner));
                }
                for &s in &a.signal_semaphores {
                    semaphores.push(s);
                }
                if a.is_tail {
                    semaphores.extend(info.signal_semaphores.iter().map(|a| a.inner));
                }
            }
        }
        for info in &*infos {
            for a in &info.command_recorder.cbs {
                let num_wait_semaphores = a.wait_semaphores.len()
                    + if a.is_head {
                        info.wait_semaphores.len()
                    } else {
                        0
                    };
                let num_signal_semaphores = a.signal_semaphores.len()
                    + if a.is_tail {
                        info.signal_semaphores.len()
                    } else {
                        0
                    };
                timeline_ext.push(
                    vk::TimelineSemaphoreSubmitInfo::default()
                        .wait_semaphore_values(&values[..num_wait_semaphores])
                        .signal_semaphore_values(&values[..num_signal_semaphores]),
                );
            }
        }
        let mut submits = Vec::new();
        {
            let mut cb_idx = 0;
            let mut semaphore_idx = 0;
            for info in &*infos {
                for cb in &info.command_recorder.cbs {
                    let num_wait_semaphores = cb.wait_semaphores.len()
                        + if cb.is_head {
                            info.wait_semaphores.len()
                        } else {
                            0
                        };
                    let num_signal_semaphores = cb.signal_semaphores.len()
                        + if cb.is_tail {
                            info.signal_semaphores.len()
                        } else {
                            0
                        };
                    let submit = vk::SubmitInfo::default()
                        .command_buffers(&cbs[cb_idx..cb_idx + 1])
                        .wait_dst_stage_mask(&pipeline_stage_flags[..num_signal_semaphores])
                        .wait_semaphores(
                            &semaphores[semaphore_idx..semaphore_idx + num_wait_semaphores],
                        )
                        .signal_semaphores(
                            &semaphores[semaphore_idx + num_wait_semaphores
                                ..semaphore_idx + num_wait_semaphores + num_signal_semaphores],
                        )
                        .push_next(unsafe { &mut *timeline_ext.as_mut_ptr().add(cb_idx) });
                    semaphore_idx += num_wait_semaphores + num_signal_semaphores;
                    cb_idx += 1;
                    submits.push(submit);
                }
            }
        }
        unsafe {
            self.device.queue_submit(
                self.queue,
                &submits,
                match fence {
                    Some(f) => f.inner,
                    None => vk::Fence::null(),
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
            for recorder in recorders {
                for cb in &recorder.cbs {
                    self.device
                        .reset_command_buffer(cb.cb, vk::CommandBufferResetFlags::empty())?;
                }
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
        buffer: &<Vulkan as Backend>::Buffer,
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
        _buffer: &<Vulkan as Backend>::Buffer,
        _map: &<Vulkan as Backend>::MappedBuffer,
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
        _buffer: &<Vulkan as Backend>::Buffer,
        _map: &<Vulkan as Backend>::MappedBuffer,
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
        _buffer: &<Vulkan as Backend>::Buffer,
        _map: <Vulkan as Backend>::MappedBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        // unsafe {self.device.unmap_memory(buffer.allocation.memory());}

        // Memory is always mapped with gpu-allocator

        Ok(())
    }
    fn write_buffer(
        &mut self,
        buffer: &<Vulkan as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let b = self.map_buffer(buffer, offset, data.len() as u64)?;
        b.slice.copy_from_slice(data);
        self.flush_mapped_buffer(buffer, &b)?;
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
        fences: &[&<Vulkan as Backend>::Fence],
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
    fn create_semaphore(&mut self) -> std::result::Result<VulkanSemaphore, VulkanError> {
        unsafe {
            let mut next = vk::SemaphoreTypeCreateInfo::default()
                .initial_value(0)
                .semaphore_type(vk::SemaphoreType::TIMELINE);
            let create_info = vk::SemaphoreCreateInfo::default()
                .flags(vk::SemaphoreCreateFlags::empty())
                .push_next(&mut next);
            Ok(VulkanSemaphore {
                inner: self.device.create_semaphore(&create_info, None)?,
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
        semaphores: &[&VulkanSemaphore],
        all: bool,
        timeout: f32,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let sems: Vec<_> = semaphores.iter().map(|a| a.inner).collect();
        let values: Vec<_> = Vec::from_iter(std::iter::repeat_n(1, semaphores.len()));
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
    fn cleanup_cached_resources(&mut self) -> Result<(), <Vulkan as Backend>::Error> {
        if !self.unused_command_buffers.is_empty() {
            unsafe {
                self.device
                    .free_command_buffers(self.command_pool, &self.unused_command_buffers);
            }
        }
        if !self.unused_events.is_empty() {
            for &event in &self.unused_events {
                unsafe {
                    self.device.destroy_event(event, None);
                }
            }
        }
        self.unused_command_buffers.clear();
        self.unused_command_buffers.clear();
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
            self.device.destroy_command_pool(self.command_pool, None);
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
    fn readable(&self) -> &[u8] {
        self.slice
    }
    fn writable(&mut self) -> &mut [u8] {
        self.slice
    }
}
struct CommandBufferSubmit {
    cb: vk::CommandBuffer,
    wait_semaphores: Vec<vk::Semaphore>,
    signal_semaphores: Vec<vk::Semaphore>,
    is_tail: bool,
    is_head: bool,
}
pub struct VulkanCommandRecorder {
    cbs: Vec<CommandBufferSubmit>,
    used_events: Vec<vk::Event>,
    allow_resubmits: bool,
}
impl VulkanCommandRecorder {
    fn begin(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default().flags(if self.allow_resubmits {
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
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.end_command_buffer(cb)?;
            Ok(())
        }
    }
    #[allow(clippy::too_many_arguments)]
    fn copy_buffer(
        &mut self,
        instance: &<Vulkan as Backend>::Instance,
        src_buffer: &<Vulkan as Backend>::Buffer,
        dst_buffer: &<Vulkan as Backend>::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.cmd_copy_buffer(
                cb,
                src_buffer.buffer,
                dst_buffer.buffer,
                &[vk::BufferCopy::default()
                    .src_offset(src_offset)
                    .dst_offset(dst_offset)
                    .size(size)],
            );
        }
        // TODO: Do sync stuff
        Ok(())
    }
    fn dispatch_kernel(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
        shader: &<Vulkan as Backend>::Kernel,
        descriptor_set: &<Vulkan as Backend>::BindGroup,
        push_constants: &[u8],
        workgroup_dims: [u32; 3],
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance
                .device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, shader.pipeline);
            instance.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                shader.pipeline_layout,
                0,
                &[descriptor_set.inner],
                &[],
            );
            if !push_constants.is_empty() {
                instance.device.cmd_push_constants(
                    cb,
                    shader.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            instance.device.cmd_dispatch(
                cb,
                workgroup_dims[0],
                workgroup_dims[1],
                workgroup_dims[2],
            );
        }
        // TODO: Do sync stuff
        Ok(())
    }
    #[allow(clippy::too_many_arguments)]
    fn dispatch_kernel_indirect(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
        shader: &<Vulkan as Backend>::Kernel,
        descriptor_set: &<Vulkan as Backend>::BindGroup,
        push_constants: &[u8],
        indirect_buffer: &<Vulkan as Backend>::Buffer,
        buffer_offset: u64,
        validate_dispatches: bool,
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            if validate_dispatches {
                return Err(VulkanError::DispatchModeUnsupported);
            }
            instance
                .device
                .cmd_bind_pipeline(cb, vk::PipelineBindPoint::COMPUTE, shader.pipeline);
            instance.device.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                shader.pipeline_layout,
                0,
                &[descriptor_set.inner],
                &[],
            );
            if !push_constants.is_empty() {
                instance.device.cmd_push_constants(
                    cb,
                    shader.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            instance
                .device
                .cmd_dispatch_indirect(cb, indirect_buffer.buffer, buffer_offset);
        }
        // TODO: Do sync stuff
        Ok(())
    }
    fn record_command(
        &mut self,
        instance: &mut VulkanInstance,
        cb: vk::CommandBuffer,
        command: &mut GpuCommand<Vulkan>,
        validate_dispatches: bool,
    ) -> Result<(), VulkanError> {
        match command {
            GpuCommand::CopyBuffer {
                src_buffer,
                dst_buffer,
                src_offset,
                dst_offset,
                size,
            } => self.copy_buffer(
                instance,
                src_buffer,
                dst_buffer,
                *src_offset,
                *dst_offset,
                *size,
                cb,
            )?,
            GpuCommand::DispatchKernel {
                shader,
                bind_group,
                push_constants,
                workgroup_dims,
            } => self.dispatch_kernel(
                instance,
                shader,
                bind_group,
                push_constants,
                *workgroup_dims,
                cb,
            )?,
            GpuCommand::DispatchKernelIndirect {
                shader,
                bind_group,
                push_constants,
                indirect_buffer,
                buffer_offset,
            } => self.dispatch_kernel_indirect(
                instance,
                shader,
                bind_group,
                push_constants,
                indirect_buffer,
                *buffer_offset,
                validate_dispatches,
                cb,
            )?,
        }
        Ok(())
    }
}
impl CommandRecorder<Vulkan> for VulkanCommandRecorder {
    fn record_dag(
        &mut self,
        _instance: &mut <Vulkan as Backend>::Instance,
        _resources: &[&GpuResource<Vulkan>],
        _dag: &mut daggy::Dag<crate::GpuOperation<Vulkan>, (usize, usize)>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        todo!()
    }
    fn record_commands(
        &mut self,
        instance: &mut VulkanInstance,
        commands: &mut [crate::GpuOperation<Vulkan>],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let cb = instance.get_command_buffer()?;
        self.begin(instance, cb)?;
        for command in commands {
            self.record_command(
                instance,
                cb,
                &mut command.command,
                command.validate_indirect,
            )?;
            unsafe {
                instance.device.cmd_pipeline_barrier(
                    cb,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::PipelineStageFlags::ALL_COMMANDS,
                    vk::DependencyFlags::empty(),
                    &[],
                    &[],
                    &[],
                );
            }
        }
        self.end(instance, cb)?;
        self.cbs.push(CommandBufferSubmit {
            cb,
            wait_semaphores: Vec::new(),
            signal_semaphores: Vec::new(),
            is_tail: true,
            is_head: true,
        });
        Ok(())
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
}
impl Semaphore<Vulkan> for VulkanSemaphore {
    fn signal(&mut self, instance: &mut VulkanInstance) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(self.inner)
                    .value(1),
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
    use crate::{CommandSynchronization, GpuCommand, GpuOperation};
    use types::ShaderReflectionInfo;

    use super::*;
    fn create_storage_buf(instance: &mut VulkanInstance, data: &[u8]) -> VulkanBuffer {
        let buf = instance
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
        let mut mapped = instance.map_buffer(&buf, 0, data.len() as u64).unwrap();
        mapped.writable().copy_from_slice(data);
        instance.flush_mapped_buffer(&buf, &mapped).unwrap();
        instance.unmap_buffer(&buf, mapped).unwrap();
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
        let fun_semaphore = instance.create_semaphore().unwrap();
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
        let sb1 = create_storage_buf(&mut instance, bytemuck::bytes_of(&[5u32, 0, 0, 0]));
        let sb2 = create_storage_buf(&mut instance, bytemuck::bytes_of(&[8u32, 0, 0, 0]));
        let mut sbout = create_storage_buf(&mut instance, bytemuck::bytes_of(&[2u32, 0, 0, 0]));
        let bind_group = instance
            .create_bind_group(
                &mut kernel,
                &[
                    GpuResource::buffer(&sb1, 0, 16),
                    GpuResource::buffer(&sb2, 0, 16),
                    GpuResource::buffer(&sbout, 0, 16),
                ],
            )
            .unwrap();

        let mut recorder = instance.create_recorder(false).unwrap();

        recorder
            .record_commands(
                &mut instance,
                &mut [GpuOperation {
                    command: GpuCommand::DispatchKernel {
                        shader: &kernel,
                        bind_group: &bind_group,
                        push_constants: &[],
                        workgroup_dims: [1, 1, 1],
                    },
                    sync: CommandSynchronization {
                        resources_needing_sync: &mut [],
                        out_semaphore: None,
                    },
                    validate_indirect: false,
                }],
            )
            .unwrap();
        instance
            .submit_recorders(
                std::slice::from_mut(&mut RecorderSubmitInfo {
                    command_recorder: &mut recorder,
                    wait_semaphores: &mut [],
                    signal_semaphores: &[&fun_semaphore],
                }),
                Some(&mut fence),
            )
            .unwrap();
        instance.wait_for_fences(&[&fence], true, 1.0).unwrap();
        instance
            .wait_for_semaphores(&[&fun_semaphore], true, 1.0)
            .unwrap();

        let mut res = [3u32, 0, 0, 0];
        read_buf(&mut instance, &mut sbout, bytemuck::bytes_of_mut(&mut res));
        if res[0] != 13 {
            panic!("Expected 13, got {}", res[0]);
        }

        instance.destroy_recorder(recorder).unwrap();

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
