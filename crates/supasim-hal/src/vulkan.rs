/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 SupaMaggie70 (Magnus Larsson)


  SupaSim is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 3
  of the License, or (at your option) any later version.

  SupaSim is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
END LICENSE */
use crate::{
    Backend, BackendInstance, BindGroup, Buffer, BufferCommand, CommandRecorder, GpuResource,
    Kernel, KernelCache, RecorderSubmitInfo, Semaphore,
};
use ash::{
    Entry, khr,
    vk::{self, Handle},
};
use core::ffi;
use gpu_allocator::{
    AllocationError, AllocationSizes, AllocatorDebugSettings,
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
};
use log::{Level, warn};
use std::fmt::{Debug, Display};
use std::{
    borrow::Cow,
    cell::Cell,
    ffi::{CStr, CString},
    sync::Mutex,
};
use thiserror::Error;
use types::{
    BufferDescriptor, BufferType, Dag, InstanceProperties, ShaderReflectionInfo,
    ShaderResourceType, SyncOperations, to_static_lifetime,
};

use scopeguard::defer;

#[derive(Debug, Clone)]
pub struct Vulkan;
impl Backend for Vulkan {
    type Buffer = VulkanBuffer;
    type BindGroup = VulkanBindGroup;
    type CommandRecorder = VulkanCommandRecorder;
    type Instance = VulkanInstance;
    type Kernel = VulkanKernel;
    type KernelCache = VulkanPipelineCache;
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
        unsafe {
            let err = Cell::new(true);
            let entry = Entry::load()?;
            let app_info =
                vk::ApplicationInfo::default().api_version(vk::make_api_version(0, 1, 2, 0));
            let debug = if debug
                && entry
                    .enumerate_instance_extension_properties(None)
                    .unwrap()
                    .iter()
                    .any(|e| e.extension_name_as_c_str().unwrap() == c"VK_LAYER_KHRONOS_validation")
            {
                true
            } else if debug {
                warn!("Debug support was requested but is not available on the current system!");
                false
            } else {
                false
            };
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
            // Check instance extensions
            {
                for &ext in &extension_names {
                    let ext_name = CStr::from_ptr(ext);
                    if !entry
                        .enumerate_instance_extension_properties(None)
                        .unwrap()
                        .iter()
                        .map(|a| a.extension_name_as_c_str().unwrap())
                        .any(|a| a == ext_name)
                    {
                        return Err(VulkanError::VulkanInstanceExtensionNotSupported(ext_name));
                    }
                }
            }
            // Check for validation layers
            {
                for &layer in &validation_layers {
                    let layer_name = CStr::from_ptr(layer);
                    if !entry
                        .enumerate_instance_layer_properties()
                        .unwrap()
                        .iter()
                        .map(|a| a.layer_name_as_c_str().unwrap())
                        .any(|a| a == layer_name)
                    {
                        return Err(VulkanError::VulkanLayerNotSupported(layer_name));
                    }
                }
            }
            // Check vulkan version
            {
                match entry.try_enumerate_instance_version().unwrap() {
                    Some(v) => {
                        if v < vk::make_api_version(0, 1, 2, 0) {
                            return Err(VulkanError::VulkanVersionTooLow(format!(
                                "{}.{}.{}",
                                vk::api_version_major(v),
                                vk::api_version_minor(v),
                                vk::api_version_patch(v)
                            )));
                        }
                    }
                    None => {
                        return Err(VulkanError::VulkanVersionTooLow("1.0".to_owned()));
                    }
                }
            }
            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_layer_names(&validation_layers)
                    .enabled_extension_names(&extension_names),
                None,
            )?;
            defer! {
                if err.get() {
                    instance.destroy_instance(None);
                }
            }
            let debug_callback = if debug {
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
                Some(debug_callback)
            } else {
                None
            };
            defer! {
                if err.get() {
                    if let Some(debug_callback) = debug_callback {
                        let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
                        debug_utils_loader.destroy_debug_utils_messenger(debug_callback, None);
                    }
                }
            }
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
                            if q.queue_flags.contains(vk::QueueFlags::COMPUTE) {
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
            // TODO: investigate multiple queues. currently we only use a general queue, but this could potentially be optimized by using special compute queues and special transfer queues
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
            defer! {
                if err.get() {
                    device.destroy_device(None);
                }
            }
            let queue = device.get_device_queue(queue_family_idx, 0);
            let s = Self::from_existing(
                debug,
                entry.clone(),
                instance.clone(),
                device.clone(),
                phyd,
                queue,
                queue_family_idx,
                debug_callback,
            )?;
            err.set(false);
            Ok(s)
        }
    }
    /// # Safety
    /// * Queue family must support `COMPUTE`
    /// * Queue must be of the given queue family, and belong to the given device
    /// * Vulkan instance must have version 1.2 or higher, and must be created from the given entry
    /// * Phyd must be from the given vulkan instance
    /// * Device must be from the given physical device, and must support timeline semaphores and synchronization 2
    /// * The queue must not be used outside of this hal instance
    /// * All resources belonging to the vulkan instance must be destroyed before the hal instance
    /// * The instance and all resources will be destroyed when the hal instance is destroyed
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_existing(
        debug: bool,
        entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        phyd: vk::PhysicalDevice,
        queue: vk::Queue,
        queue_family_idx: u32,
        debug_callback: Option<vk::DebugUtilsMessengerEXT>,
    ) -> Result<VulkanInstance, VulkanError> {
        unsafe {
            let err = Cell::new(true);
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
            defer! {
                if err.get() {
                    device.destroy_command_pool(pool, None);
                }
            }
            let s = VulkanInstance {
                sync2_dev: khr::synchronization2::Device::new(&instance, &device),
                entry,
                instance,
                device: device.clone(),
                phyd,
                alloc: Mutex::new(alloc),
                queue,
                queue_family_idx,
                command_pool: pool,
                unused_command_buffers: Vec::new(),
                unused_events: Vec::new(),
                debug: debug_callback,
            };
            err.set(false);
            Ok(s)
        }
    }
}
#[must_use]
#[derive(Error, Debug)]
pub enum VulkanError {
    #[error(
        "Provided Vulkan version is too low. Vulkan version 1.2 or higher is required, but only {0} is supported."
    )]
    VulkanVersionTooLow(String),
    #[error("Instance extension {0:?}, which is required, is not supported")]
    VulkanInstanceExtensionNotSupported(&'static CStr),
    #[error("Layer {0:?}, which is required, is not supported")]
    VulkanLayerNotSupported(&'static CStr),

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
pub struct VulkanInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    device: ash::Device,
    phyd: vk::PhysicalDevice,
    alloc: Mutex<Allocator>,
    queue: vk::Queue,
    queue_family_idx: u32,
    command_pool: vk::CommandPool,
    unused_command_buffers: Vec<vk::CommandBuffer>,
    unused_events: Vec<vk::Event>,
    debug: Option<vk::DebugUtilsMessengerEXT>,
    sync2_dev: khr::synchronization2::Device,
}
impl Debug for VulkanInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VulkanInstance")
    }
}
impl Display for VulkanInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "VulkanInstance({})", self.device.handle().as_raw())
    }
}
impl VulkanInstance {
    #[tracing::instrument]
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
    #[tracing::instrument]
    unsafe fn destroy(mut self) -> Result<(), VulkanError> {
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
        Ok(())
    }
    #[tracing::instrument]
    fn get_properties(&mut self) -> InstanceProperties {
        let is_unified_memory = unsafe {
            let memory_props = self
                .instance
                .get_physical_device_memory_properties(self.phyd);
            // The idea here is that implementations are required to provide a host visible and coherent memory type.
            // Heaps and types are different, but implementations generally provide (pinned) system memory as a heap.
            // If this heap isn't separate from the device heap, we know any system memory type is also device local.
            // The spec recommends this for UMA systems: https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#memory-device
            memory_props.memory_heaps.len() == 1
                && memory_props.memory_heaps_as_slice()[0]
                    .flags
                    .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
        };
        InstanceProperties {
            sync_mode: types::SyncMode::VulkanStyle,
            indirect: false,
            pipeline_cache: true,
            // TODO: detect spirv version
            shader_type: types::ShaderTarget::Spirv {
                version: types::SpirvVersion::V1_0,
            },
            easily_update_bind_groups: false,
            semaphore_signal: true,
            // TODO: add unified memory detection
            is_unified_memory,
        }
    }
    #[tracing::instrument(skip(binary))]
    unsafe fn compile_kernel(
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
                            ShaderResourceType::Unknown => unreachable!("Unknown shader binding"),
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
    #[tracing::instrument]
    unsafe fn destroy_kernel(
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
    #[tracing::instrument]
    unsafe fn create_buffer(
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
                    let mut flags = match alloc_info.memory_type {
                        BufferType::Storage => {
                            F::TRANSFER_SRC | F::TRANSFER_DST | F::STORAGE_BUFFER
                        }
                        BufferType::Upload => F::TRANSFER_SRC,
                        BufferType::Download => F::TRANSFER_DST,
                        BufferType::Other => F::TRANSFER_DST,
                        BufferType::Any => unreachable!(),
                    };
                    if alloc_info.indirect_capable {
                        flags |= F::INDIRECT_BUFFER | F::STORAGE_BUFFER
                    }
                    if alloc_info.uniform {
                        flags |= F::UNIFORM_BUFFER
                    };
                    flags
                });
            let buffer = self.device.create_buffer(&create_info, None)?;
            defer! {
                if err.get() {
                    self.device.destroy_buffer(buffer, None);
                }
            }
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            use types::BufferType::*;
            let mut allocation = self
                .alloc
                .lock()
                .map_err(|e| VulkanError::LockError(e.to_string()))?
                .allocate(&AllocationCreateDesc {
                    name: "",
                    requirements,
                    location: match alloc_info.memory_type {
                        Any => gpu_allocator::MemoryLocation::Unknown,
                        Storage => gpu_allocator::MemoryLocation::GpuOnly,
                        Upload => gpu_allocator::MemoryLocation::CpuToGpu,
                        Download => gpu_allocator::MemoryLocation::GpuToCpu,
                        Other => gpu_allocator::MemoryLocation::CpuToGpu,
                    },
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged,
                })?;
            let alloc_ptr = &mut allocation as *mut Allocation;
            defer! {
                if err.get() {
                    self.alloc.lock().unwrap().free(std::mem::take(&mut *alloc_ptr)).unwrap();
                }
            }
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
            err.set(false);
            if alloc_info.memory_type == BufferType::Upload
                || alloc_info.memory_type == BufferType::Download
            {
                assert!(allocation.mapped_ptr().is_some());
            }
            Ok(VulkanBuffer {
                buffer,
                allocation,
                create_info: *alloc_info,
            })
        }
    }
    #[tracing::instrument]
    unsafe fn destroy_buffer(
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
    #[tracing::instrument(skip(initial_data))]
    unsafe fn create_kernel_cache(
        &mut self,
        initial_data: &[u8],
    ) -> Result<<Vulkan as Backend>::KernelCache, <Vulkan as Backend>::Error> {
        unsafe {
            let create_info = vk::PipelineCacheCreateInfo::default().initial_data(initial_data);
            let pc = self.device.create_pipeline_cache(&create_info, None)?;
            Ok(VulkanPipelineCache {
                inner: Mutex::new(pc),
            })
        }
    }
    #[tracing::instrument]
    unsafe fn destroy_kernel_cache(
        &mut self,
        cache: <Vulkan as Backend>::KernelCache,
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
    #[tracing::instrument]
    unsafe fn get_kernel_cache_data(
        &mut self,
        cache: &mut <Vulkan as Backend>::KernelCache,
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
    #[tracing::instrument]
    unsafe fn create_bind_group(
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
    #[tracing::instrument]
    unsafe fn update_bind_group(
        &mut self,
        bg: &mut <Vulkan as Backend>::BindGroup,
        _kernel: &mut <Vulkan as Backend>::Kernel,
        resources: &[GpuResource<Vulkan>],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let mut writes = Vec::with_capacity(resources.len());
        let mut buffer_infos = Vec::with_capacity(resources.len());
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
                    len: size,
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
    #[tracing::instrument]
    unsafe fn destroy_bind_group(
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
    #[tracing::instrument]
    unsafe fn create_recorder(
        &mut self,
    ) -> Result<<Vulkan as Backend>::CommandRecorder, <Vulkan as Backend>::Error> {
        Ok(VulkanCommandRecorder {
            cbs: Vec::new(),
            used_events: Vec::new(),
        })
    }
    #[tracing::instrument]
    unsafe fn destroy_recorder(
        &mut self,
        recorder: <Vulkan as Backend>::CommandRecorder,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        self.unused_events.extend(&recorder.used_events);
        self.unused_command_buffers
            .extend(recorder.cbs.into_iter().map(|a| a.cb));
        Ok(())
    }
    #[tracing::instrument]
    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<Vulkan>],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        // Holy shit this function can't be right
        let mut cbs = Vec::new();
        let mut semaphores: Vec<vk::Semaphore> = Vec::new();
        let mut pipeline_stage_flags = Vec::new();
        let mut ensure_enough_flags = |num| {
            while pipeline_stage_flags.len() < num {
                pipeline_stage_flags.push(vk::PipelineStageFlags::ALL_COMMANDS);
            }
        };
        // TODO: make this work with new semaphores
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
            if num_tails > 1 && info.signal_semaphore.is_some() {
                return Err(VulkanError::SemaphoreSignalInDag);
            }
            for a in &info.command_recorder.cbs {
                cbs.push(a.cb);
                let num_wait_semaphores = a.wait_semaphores.len()
                    + if a.is_head {
                        info.wait_semaphore.is_some() as usize
                    } else {
                        0
                    };
                let num_signal_semaphores = a.signal_semaphores.len()
                    + if a.is_tail {
                        info.signal_semaphore.is_some() as usize
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
                    semaphores.extend(info.wait_semaphore.map(|a| a.inner));
                }
                for &s in &a.signal_semaphores {
                    semaphores.push(s);
                }
                if a.is_tail {
                    semaphores.extend(info.signal_semaphore.iter().map(|a| a.inner));
                }
            }
        }
        for info in &*infos {
            for a in &info.command_recorder.cbs {
                let num_wait_semaphores = a.wait_semaphores.len()
                    + if a.is_head {
                        info.wait_semaphore.is_some() as usize
                    } else {
                        0
                    };
                let num_signal_semaphores = a.signal_semaphores.len()
                    + if a.is_tail {
                        info.signal_semaphore.is_some() as usize
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
                            info.wait_semaphore.is_some() as usize
                        } else {
                            0
                        };
                    let num_signal_semaphores = cb.signal_semaphores.len()
                        + if cb.is_tail {
                            info.signal_semaphore.is_some() as usize
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
            self.device
                .queue_submit(self.queue, &submits, vk::Fence::null())?;
        }
        Ok(())
    }
    #[tracing::instrument]
    unsafe fn wait_for_idle(&mut self) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.device.queue_wait_idle(self.queue)?;
            Ok(())
        }
    }
    #[tracing::instrument(skip(data), fields(len=data.len()))]
    unsafe fn write_buffer(
        &mut self,
        buffer: &<Vulkan as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            let b =
                (buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8).add(offset as usize);
            let slice = std::slice::from_raw_parts_mut(b, data.len());
            slice.copy_from_slice(data);
            Ok(())
        }
    }
    #[tracing::instrument(skip(data), fields(len=data.len()))]
    unsafe fn read_buffer(
        &mut self,
        buffer: &<Vulkan as Backend>::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            let b =
                (buffer.allocation.mapped_ptr().unwrap().as_ptr() as *mut u8).add(offset as usize);
            let slice = std::slice::from_raw_parts(b as *const u8, data.len());
            data.copy_from_slice(slice);
        }
        Ok(())
    }
    #[tracing::instrument]
    unsafe fn create_semaphore(&mut self) -> std::result::Result<VulkanSemaphore, VulkanError> {
        unsafe {
            let mut next = vk::SemaphoreTypeCreateInfo::default()
                .initial_value(0)
                .semaphore_type(vk::SemaphoreType::TIMELINE);
            let create_info = vk::SemaphoreCreateInfo::default()
                .flags(vk::SemaphoreCreateFlags::empty())
                .push_next(&mut next);
            Ok(VulkanSemaphore {
                inner: self.device.create_semaphore(&create_info, None)?,
                current_value: 0,
            })
        }
    }
    #[tracing::instrument]
    unsafe fn destroy_semaphore(
        &mut self,
        semaphore: VulkanSemaphore,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.device.destroy_semaphore(semaphore.inner, None);
        }
        Ok(())
    }
    #[tracing::instrument]
    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Vulkan as Backend>::Error> {
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
#[derive(Debug)]
pub struct VulkanKernel {
    pub shader: vk::ShaderModule,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_pools: Vec<DescriptorPoolData>,
}
impl Kernel<Vulkan> for VulkanKernel {}

#[derive(Debug)]
pub struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub allocation: Allocation,
    pub create_info: types::BufferDescriptor,
}
impl Buffer<Vulkan> for VulkanBuffer {}
#[derive(Debug)]
struct CommandBufferSubmit {
    cb: vk::CommandBuffer,
    wait_semaphores: Vec<vk::Semaphore>,
    signal_semaphores: Vec<vk::Semaphore>,
    is_tail: bool,
    is_head: bool,
}
#[derive(Debug)]
pub struct VulkanCommandRecorder {
    cbs: Vec<CommandBufferSubmit>,
    used_events: Vec<vk::Event>,
}
impl VulkanCommandRecorder {
    #[tracing::instrument]
    fn begin(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            Ok(())
        }
    }
    #[tracing::instrument]
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
    #[tracing::instrument]
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
        Ok(())
    }
    #[tracing::instrument(skip(push_constants), fields(push_constants_len=push_constants.len()))]
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
        Ok(())
    }
    #[tracing::instrument(skip(push_constants), fields(push_constants_len=push_constants.len()))]
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
        Ok(())
    }
    fn stage_mask_khr(sync_ops: SyncOperations) -> vk::PipelineStageFlags2KHR {
        match sync_ops {
            SyncOperations::Transfer => vk::PipelineStageFlags2KHR::TRANSFER,
            SyncOperations::ComputeDispatch => vk::PipelineStageFlags2KHR::COMPUTE_SHADER,
            SyncOperations::Both => vk::PipelineStageFlags2KHR::ALL_COMMANDS,
        }
    }
    /// First command must be a pipeline barrier or wait event command. Following commands must be memory barriers
    fn sync_command<'a>(
        &mut self,
        instance: &<Vulkan as Backend>::Instance,
        cb: vk::CommandBuffer,
        commands: impl IntoIterator<Item = &'a BufferCommand<'a, Vulkan>>,
    ) -> Result<(), VulkanError> {
        let mut barriers = Vec::new();
        let mut pre_flags = vk::PipelineStageFlags2KHR::empty();
        let mut post_flags = vk::PipelineStageFlags2KHR::empty();
        for command in commands {
            match command {
                BufferCommand::MemoryBarrier {
                    resource:
                        GpuResource::Buffer {
                            buffer,
                            offset,
                            len: size,
                        },
                } => barriers.push(
                    vk::BufferMemoryBarrier2KHR::default()
                        .buffer(buffer.buffer)
                        .offset(*offset)
                        .size(*size)
                        .src_queue_family_index(instance.queue_family_idx)
                        .dst_queue_family_index(instance.queue_family_idx)
                        .src_access_mask(
                            vk::AccessFlags2KHR::MEMORY_READ_KHR
                                | vk::AccessFlags2KHR::MEMORY_WRITE_KHR,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2KHR::MEMORY_READ_KHR
                                | vk::AccessFlags2KHR::MEMORY_WRITE_KHR,
                        ),
                ),
                BufferCommand::PipelineBarrier { before, after } => {
                    pre_flags |= Self::stage_mask_khr(*before);
                    post_flags |= Self::stage_mask_khr(*after);
                }
                _ => unreachable!(),
            }
        }
        if pre_flags.is_empty() || post_flags.is_empty() {
            return Ok(());
        }
        for barrier in &mut barriers {
            *barrier = barrier.src_stage_mask(pre_flags).dst_stage_mask(post_flags);
        }
        let dependency_info = vk::DependencyInfoKHR::default().buffer_memory_barriers(&barriers);
        unsafe {
            instance
                .sync2_dev
                .cmd_pipeline_barrier2(cb, &dependency_info)
        };
        Ok(())
    }
    #[tracing::instrument]
    fn record_command(
        &mut self,
        instance: &mut VulkanInstance,
        cb: vk::CommandBuffer,
        command: &mut BufferCommand<Vulkan>,
    ) -> Result<(), VulkanError> {
        match command {
            BufferCommand::CopyBuffer {
                src_buffer,
                dst_buffer,
                src_offset,
                dst_offset,
                len,
            } => self.copy_buffer(
                instance,
                src_buffer,
                dst_buffer,
                *src_offset,
                *dst_offset,
                *len,
                cb,
            )?,
            BufferCommand::DispatchKernel {
                kernel: shader,
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
            BufferCommand::DispatchKernelIndirect {
                kernel: shader,
                bind_group,
                push_constants,
                indirect_buffer,
                buffer_offset,
                validate,
            } => self.dispatch_kernel_indirect(
                instance,
                shader,
                bind_group,
                push_constants,
                indirect_buffer,
                *buffer_offset,
                *validate,
                cb,
            )?,
            BufferCommand::PipelineBarrier { .. } => {
                unreachable!()
            }
            BufferCommand::MemoryBarrier { .. } => {
                unreachable!()
            }
            BufferCommand::UpdateBindGroup { .. } => unreachable!(),
        }
        Ok(())
    }
}
impl CommandRecorder<Vulkan> for VulkanCommandRecorder {
    unsafe fn record_dag(
        &mut self,
        _instance: &mut <Vulkan as Backend>::Instance,
        _resources: &[&GpuResource<Vulkan>],
        _dag: &mut Dag<crate::BufferCommand<Vulkan>>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unreachable!()
    }
    #[tracing::instrument]
    unsafe fn record_commands(
        &mut self,
        instance: &mut VulkanInstance,
        commands: &mut [crate::BufferCommand<Vulkan>],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let cb = instance.get_command_buffer()?;
        self.begin(instance, cb)?;
        let mut pipeline_chain_start = None;
        for i in 0..commands.len() {
            match &commands[i] {
                BufferCommand::MemoryBarrier { .. } | BufferCommand::PipelineBarrier { .. } => {
                    if pipeline_chain_start.is_none() {
                        pipeline_chain_start = Some(i);
                    }
                }
                _ => {
                    if let Some(start) = pipeline_chain_start {
                        self.sync_command(instance, cb, &commands[start..i])?;
                        pipeline_chain_start = None;
                    }
                    self.record_command(instance, cb, &mut commands[i])?;
                }
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
    unsafe fn clear(
        &mut self,
        _instance: &mut <Vulkan as Backend>::Instance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        todo!()
    }
}
#[derive(Debug)]
pub struct DescriptorPoolData {
    pub pool: vk::DescriptorPool,
    pub max_size: u32,
    pub current_size: u32,
}

#[derive(Debug)]
pub struct VulkanBindGroup {
    inner: vk::DescriptorSet,
    pool_idx: u32,
}
impl BindGroup<Vulkan> for VulkanBindGroup {}
#[derive(Debug)]
pub struct VulkanPipelineCache {
    inner: Mutex<vk::PipelineCache>,
}
impl KernelCache<Vulkan> for VulkanPipelineCache {}
#[derive(Debug)]
pub struct VulkanSemaphore {
    inner: vk::Semaphore,
    current_value: u64,
}
impl Semaphore<Vulkan> for VulkanSemaphore {
    #[tracing::instrument]
    unsafe fn wait(
        &mut self,
        instance: &mut VulkanInstance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(std::slice::from_ref(&self.inner))
                    .values(&[self.current_value + 1]),
                u64::MAX,
            )?;
        }
        Ok(())
    }
    #[tracing::instrument]
    unsafe fn is_signalled(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
    ) -> Result<bool, <Vulkan as Backend>::Error> {
        Ok(
            unsafe { instance.device.get_semaphore_counter_value(self.inner)? }
                == self.current_value + 1,
        )
    }
    #[tracing::instrument]
    unsafe fn signal(
        &mut self,
        instance: &mut <Vulkan as Backend>::Instance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            instance.device.signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(self.inner)
                    .value(self.current_value + 1),
            )?;
        }
        Ok(())
    }
    unsafe fn reset(
        &mut self,
        _instance: &mut <Vulkan as Backend>::Instance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        self.current_value += 1;
        Ok(())
    }
}
