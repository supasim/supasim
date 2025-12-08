/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

mod command_recorder;
mod init;

pub use command_recorder::VulkanCommandRecorder;

use crate::{
    Backend, BackendInstance, BindGroup, Buffer, Device, HalBufferSlice, Kernel, KernelDescriptor,
    RecorderSubmitInfo, Semaphore, Stream,
};
use ash::{
    khr,
    prelude::VkResult,
    vk::{self, Handle},
};
use std::{cell::Cell, ffi::CStr, ops::Deref, sync::Mutex};
use std::{
    fmt::{Debug, Display},
    sync::Arc,
};
use thiserror::Error;
use types::{HalBufferType, HalInstanceProperties};
use vk_mem::{Alloc, AllocationCreateFlags, Allocator};

use scopeguard::defer;

/// # Overview
/// The default backend for platforms on which it is supported.
///
/// ## Issues/workarounds
/// * Invalid usage won't be caught without debug enabled
/// * Even with debug enabled, validation is likely to miss many issues
/// * Debug requires more system libraries than just default GPU drivers - Vulkan SDK is recommended
/// * Improper usage can cause serious issues, including
///   * Hanging indefinitely on waits
///   * Memory corruption
///   * Memory leaks
///   * Segfaults and the like
/// * Very dependent on extensions
///   * Some very esoteric systems may not support the required extensions
///   * Systems with untested combinations of extensions might experience other issues
/// * Detection of unified memory systems is flawed
/// * Synchronization is good but may miss opportunities for parallelization on some systems
///   * Cuda is much stronger in this regard
///   * Lack of multi queue support is a big reason for this
#[derive(Debug, Clone)]
pub struct Vulkan;
impl Backend for Vulkan {
    type Instance = VulkanInstance;
    type Device = VulkanDevice;
    type Stream = VulkanStream;

    type Buffer = VulkanBuffer;
    type BindGroup = VulkanBindGroup;
    type CommandRecorder = VulkanCommandRecorder;
    type Kernel = VulkanKernel;
    type Semaphore = VulkanSemaphore;

    type Error = VulkanError;

    #[cfg_attr(feature = "trace", tracing::instrument)]
    fn setup_default_descriptor() -> Result<crate::InstanceDescriptor<Self>, Self::Error> {
        Self::create_instance(true)
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
    AllocationError(String),
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
    #[error("A buffer export was attempted under invalid conditions")]
    ExternalMemoryExport,
}

impl crate::Error<Vulkan> for VulkanError {
    fn is_out_of_device_memory(&self) -> bool {
        match self {
            Self::VulkanRaw(e) => *e == vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            Self::AllocationError(_) => true,
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

pub struct VulkanDevice {
    shared: Arc<SharedDeviceInfo>,
    alloc: Allocator,
    is_unified_memory: bool,
}

impl std::fmt::Debug for VulkanDevice {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        std::fmt::Debug::fmt(&self.shared, f)?;
        std::fmt::Debug::fmt(&self.is_unified_memory, f)?;
        Ok(())
    }
}

impl Device<Vulkan> for VulkanDevice {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    fn get_properties(
        &self,
        _instance: &<Vulkan as Backend>::Instance,
    ) -> types::HalDeviceProperties {
        let props = unsafe {
            _instance
                .instance
                .get_physical_device_properties(_instance._phyd)
        };
        types::HalDeviceProperties {
            is_unified_memory: self.is_unified_memory,
            host_mappable_buffers: self.is_unified_memory,
            // TODO: this is vulkan specific, fix that
            driver_id: props.vendor_id as u64 + ((props.device_id as u64) << 32),
            supports_buffer_import: false,
            supports_semaphore_import: false,
        }
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn cleanup_cached_resources(
        &self,
        _instance: &<Vulkan as Backend>::Instance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_buffer(
        &self,
        desc: &types::HalBufferDescriptor,
    ) -> Result<<Vulkan as Backend>::Buffer, <Vulkan as Backend>::Error> {
        unsafe {
            let buffer_info = vk::BufferCreateInfo::default()
                .size(desc.size)
                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                .queue_family_indices(&self.shared.queue_family_indices)
                .usage({
                    use vk::BufferUsageFlags as F;
                    match desc.memory_type {
                        HalBufferType::Storage => {
                            F::TRANSFER_SRC | F::TRANSFER_DST | F::STORAGE_BUFFER
                        }
                        HalBufferType::Upload => F::TRANSFER_SRC,
                        HalBufferType::Download => F::TRANSFER_DST,
                        HalBufferType::UploadDownload => F::TRANSFER_SRC | F::TRANSFER_DST,
                    }
                });
            let mapped = matches!(
                desc.memory_type,
                HalBufferType::Download | HalBufferType::Upload | HalBufferType::UploadDownload
            );
            let alloc_info = vk_mem::AllocationCreateInfo {
                usage: vk_mem::MemoryUsage::Auto,
                flags: if mapped {
                    AllocationCreateFlags::MAPPED
                        | AllocationCreateFlags::HOST_ACCESS_SEQUENTIAL_WRITE
                } else {
                    AllocationCreateFlags::empty()
                },
                ..Default::default()
            };
            let (buffer, mut allocation) = self.alloc.create_buffer(&buffer_info, &alloc_info)?;
            let mapped_ptr = if mapped {
                Some(self.alloc.map_memory(&mut allocation)?)
            } else {
                None
            };
            Ok(VulkanBuffer {
                buffer,
                allocation,
                create_info: *desc,
                mapped_ptr,
            })
        }
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn import_buffer(
        &self,
        _info: &types::ExternalBufferDescriptor,
    ) -> Result<<Vulkan as Backend>::Buffer, <Vulkan as Backend>::Error> {
        unreachable!()
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_semaphore(&self) -> std::result::Result<VulkanSemaphore, VulkanError> {
        unsafe {
            let mut next = vk::SemaphoreTypeCreateInfo::default()
                .initial_value(0)
                .semaphore_type(vk::SemaphoreType::TIMELINE);
            let create_info = vk::SemaphoreCreateInfo::default()
                .flags(vk::SemaphoreCreateFlags::empty())
                .push_next(&mut next);
            Ok(VulkanSemaphore {
                inner: self.shared.functions.create_semaphore(&create_info, None)?,
                current_value: Mutex::new(0),
            })
        }
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn import_semaphore(
        &self,
        _info: &types::ExternalSemaphoreDescriptor,
    ) -> Result<<Vulkan as Backend>::Semaphore, <Vulkan as Backend>::Error> {
        unreachable!()
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _instance: &mut <Vulkan as Backend>::Instance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        // TODO: do something with these stats?
        let _stats = self.alloc.calculate_statistics()?;
        drop(self.alloc);
        Ok(())
    }
}

#[derive(Debug)]
pub struct VulkanStream {
    shared: Arc<SharedDeviceInfo>,
    queue: vk::Queue,
    queue_family_idx: u32,
    command_pool: vk::CommandPool,
    unused_command_buffers: Mutex<Vec<vk::CommandBuffer>>,
}

impl Stream<Vulkan> for VulkanStream {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn wait_for_idle(&mut self) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.shared.functions.queue_wait_idle(self.queue)?;
            Ok(())
        }
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_recorder(
        &self,
    ) -> Result<<Vulkan as Backend>::CommandRecorder, <Vulkan as Backend>::Error> {
        Ok(VulkanCommandRecorder {
            inner: self.get_command_buffer()?,
        })
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<Vulkan>],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let mut submits = Vec::new();
        let mut wait_semaphores = Vec::new();
        let mut signal_semaphores = Vec::new();
        let mut cb_infos = Vec::new();
        for info in &*infos {
            for sem in info.wait_semaphores {
                wait_semaphores.push(
                    vk::SemaphoreSubmitInfoKHR::default()
                        .semaphore(sem.inner)
                        .value(*sem.current_value.lock().unwrap() + 1)
                        .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS_KHR),
                );
            }
            if let Some(s) = info.signal_semaphore {
                signal_semaphores.push(
                    vk::SemaphoreSubmitInfoKHR::default()
                        .semaphore(s.inner)
                        .value(*s.current_value.lock().unwrap() + 1)
                        .stage_mask(vk::PipelineStageFlags2KHR::ALL_COMMANDS_KHR),
                );
            }
            cb_infos.push(
                vk::CommandBufferSubmitInfoKHR::default()
                    .command_buffer(info.command_recorder.inner),
            );
        }
        {
            let mut wait_idx = 0;
            let mut signal_idx = 0;
            for (cb_idx, info) in (*infos).iter().enumerate() {
                let submit = vk::SubmitInfo2KHR::default()
                    .command_buffer_infos(std::slice::from_ref(&cb_infos[cb_idx]))
                    .wait_semaphore_infos(
                        &wait_semaphores[wait_idx..wait_idx + info.wait_semaphores.len()],
                    )
                    .signal_semaphore_infos(if info.signal_semaphore.is_some() {
                        std::slice::from_ref(&signal_semaphores[signal_idx])
                    } else {
                        &[]
                    });
                submits.push(submit);
                wait_idx += info.wait_semaphores.len();
                signal_idx += info.signal_semaphore.is_some() as usize;
            }
        }

        unsafe {
            self.shared
                .functions
                .supa_queue_submit2(self.queue, &submits, vk::Fence::null())?;
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_bind_group(
        &self,
        device: &VulkanDevice,
        kernel: &VulkanKernel,
        resources: &[crate::HalBufferSlice<Vulkan>],
    ) -> Result<VulkanBindGroup, VulkanError> {
        let mut lock = kernel.descriptor_pools.lock().unwrap();
        let mut pool_idx = None;
        for (i, pool) in kernel
            .descriptor_pools
            .lock()
            .unwrap()
            .iter_mut()
            .enumerate()
        {
            if pool.max_size > pool.current_size {
                pool_idx = Some(i);
                break;
            }
        }
        if pool_idx.is_none() {
            pool_idx = Some(lock.len());
            let next_size = lock
                .last()
                .map(|s| s.max_size * 2)
                .unwrap_or(8)
                .next_power_of_two();
            let num_buffers = resources.len() as u32;
            let mut sizes = vec![];
            if num_buffers > 0 {
                sizes.push(
                    vk::DescriptorPoolSize::default()
                        .descriptor_count(num_buffers * next_size)
                        .ty(vk::DescriptorType::STORAGE_BUFFER),
                );
            }
            unsafe {
                let create_info = vk::DescriptorPoolCreateInfo::default()
                    .max_sets(next_size)
                    .pool_sizes(&sizes)
                    .flags(vk::DescriptorPoolCreateFlags::FREE_DESCRIPTOR_SET);
                let pool = self
                    .shared
                    .functions
                    .create_descriptor_pool(&create_info, None)?;
                lock.push(DescriptorPoolData {
                    pool,
                    max_size: next_size,
                    current_size: 0,
                });
            }
        }
        let pool_idx = pool_idx.unwrap();
        let pool = lock[pool_idx].pool;
        lock[pool_idx].current_size += 1;
        drop(lock);

        unsafe {
            let alloc_info = vk::DescriptorSetAllocateInfo::default()
                .descriptor_pool(pool)
                .set_layouts(std::slice::from_ref(&kernel.descriptor_set_layout));
            let descriptor_set = self
                .shared
                .functions
                .allocate_descriptor_sets(&alloc_info)?[0];
            defer! {}
            let mut bg = VulkanBindGroup {
                inner: descriptor_set,
                pool_idx: pool_idx as u32,
            };

            if let Err(e) = bg.update(device, self, kernel, resources) {
                let _ = self
                    .shared
                    .functions
                    .free_descriptor_sets(pool, &[descriptor_set]);
                kernel.descriptor_pools.lock().unwrap()[pool_idx].current_size -= 1;
                Err(e)
            } else {
                Ok(bg)
            }
        }
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn cleanup_cached_resources(
        &self,
        instance: &<Vulkan as Backend>::Instance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let mut vk_cbs = self.unused_command_buffers.lock().unwrap();
        unsafe {
            instance
                .shared
                .functions
                .free_command_buffers(self.command_pool, &vk_cbs);
        }
        vk_cbs.clear();
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _device: &mut <Vulkan as Backend>::Device,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.shared
                .functions
                .destroy_command_pool(self.command_pool, None);
        }
        Ok(())
    }
}

pub struct VulkanInstance {
    entry: ash::Entry,
    instance: ash::Instance,
    _phyd: vk::PhysicalDevice,
    debug: Option<vk::DebugUtilsMessengerEXT>,
    shared: Arc<SharedDeviceInfo>,
    spirv_version: types::SpirvVersion,
    _api_version: u32,
    atomic_int64: bool,
}

impl Debug for VulkanInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("VulkanInstance")
    }
}

impl Display for VulkanInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "VulkanInstance({})",
            self.shared.functions.handle().as_raw()
        )
    }
}

impl VulkanStream {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    pub fn get_command_buffer(&self) -> Result<vk::CommandBuffer, VulkanError> {
        match self.unused_command_buffers.lock().unwrap().pop() {
            Some(c) => Ok(c),
            None => unsafe {
                Ok(self
                    .shared
                    .functions
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
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(self) -> Result<(), VulkanError> {
        unsafe {
            self.shared.functions.destroy_device(None);
            if let Some(debug) = self.debug {
                ash::ext::debug_utils::Instance::new(&self.entry, &self.instance)
                    .destroy_debug_utils_messenger(debug, None);
            }
            self.instance.destroy_instance(None);
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    fn get_properties(&self) -> HalInstanceProperties {
        HalInstanceProperties {
            sync_mode: types::SyncMode::VulkanStyle,
            kernel_lang: types::KernelTarget::Spirv {
                version: self.spirv_version,
            },
            easily_update_bind_groups: false,
            semaphore_signal: true,
            map_buffers: true,
            map_buffer_while_gpu_use: true,
            upload_download_buffers: true,
            atomic_int64: self.atomic_int64,
        }
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn compile_kernel(
        &self,
        desc: KernelDescriptor,
    ) -> Result<<Vulkan as Backend>::Kernel, <Vulkan as Backend>::Error> {
        unsafe {
            let err = Cell::new(true);
            let kernel_create_info = &vk::ShaderModuleCreateInfo::default();
            let ptr = desc.binary.as_ptr() as *const u32;
            assert!(desc.binary.len().is_multiple_of(4));
            let kernel = if ptr.is_aligned() {
                self.shared.functions.create_shader_module(
                    &kernel_create_info
                        .code(std::slice::from_raw_parts(ptr, desc.binary.len() / 4)),
                    None,
                )?
            } else {
                let mut v = Vec::<u32>::with_capacity(desc.binary.len() / 4);
                #[allow(clippy::uninit_vec)]
                v.set_len(desc.binary.len() / 4);
                desc.binary
                    .as_ptr()
                    .copy_to(v.as_mut_ptr() as *mut u8, desc.binary.len());
                self.shared
                    .functions
                    .create_shader_module(&kernel_create_info.code(&v), None)?
            };
            defer! {
                if err.get() {
                    self.shared.functions.destroy_shader_module(kernel, None);
                }
            }
            let mut bindings = Vec::with_capacity(desc.reflection.buffers.len());
            for i in 0..desc.reflection.buffers.len() {
                bindings.push(
                    vk::DescriptorSetLayoutBinding::default()
                        .binding(i as u32)
                        .descriptor_count(1)
                        .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                        .stage_flags(vk::ShaderStageFlags::COMPUTE),
                );
            }
            let desc_set_layout_create =
                vk::DescriptorSetLayoutCreateInfo::default().bindings(&bindings);
            let descriptor_set_layout = self
                .shared
                .functions
                .create_descriptor_set_layout(&desc_set_layout_create, None)?;
            defer! {
                if err.get() {
                    self.shared.functions.destroy_descriptor_set_layout(descriptor_set_layout, None);
                }
            }
            let pipeline_layout = self.shared.functions.create_pipeline_layout(
                &vk::PipelineLayoutCreateInfo::default().set_layouts(&[descriptor_set_layout]),
                None,
            )?;
            let entry = c"main";
            let pipeline_create_info = vk::ComputePipelineCreateInfo::default()
                .stage(
                    vk::PipelineShaderStageCreateInfo::default()
                        .stage(vk::ShaderStageFlags::COMPUTE)
                        .module(kernel)
                        .name(entry),
                )
                .layout(pipeline_layout);
            let pipeline = self
                .shared
                .functions
                .create_compute_pipelines(vk::PipelineCache::null(), &[pipeline_create_info], None)
                .map_err(|e| e.1)?[0];
            err.set(false);
            Ok(VulkanKernel {
                kernel,
                pipeline,
                pipeline_layout,
                descriptor_set_layout,
                descriptor_pools: Mutex::new(Vec::new()),
            })
        }
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_semaphore(&self) -> std::result::Result<VulkanSemaphore, VulkanError> {
        unsafe {
            let mut next = vk::SemaphoreTypeCreateInfo::default()
                .initial_value(0)
                .semaphore_type(vk::SemaphoreType::TIMELINE);
            let create_info = vk::SemaphoreCreateInfo::default()
                .flags(vk::SemaphoreCreateFlags::empty())
                .push_next(&mut next);
            Ok(VulkanSemaphore {
                inner: self.shared.functions.create_semaphore(&create_info, None)?,
                current_value: Mutex::new(0),
            })
        }
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Vulkan as Backend>::Error> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct VulkanKernel {
    pub kernel: vk::ShaderModule,
    pub pipeline: vk::Pipeline,
    pub descriptor_set_layout: vk::DescriptorSetLayout,
    pub pipeline_layout: vk::PipelineLayout,
    pub descriptor_pools: Mutex<Vec<DescriptorPoolData>>,
}

impl Kernel<Vulkan> for VulkanKernel {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        mut self,
        instance: &VulkanInstance,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            for pool in self.descriptor_pools.get_mut().unwrap() {
                instance
                    .shared
                    .functions
                    .destroy_descriptor_pool(pool.pool, None);
            }
            instance
                .shared
                .functions
                .destroy_pipeline(self.pipeline, None);
            instance
                .shared
                .functions
                .destroy_pipeline_layout(self.pipeline_layout, None);
            instance
                .shared
                .functions
                .destroy_shader_module(self.kernel, None);
            instance
                .shared
                .functions
                .destroy_descriptor_set_layout(self.descriptor_set_layout, None);
            Ok(())
        }
    }
}

#[derive(Debug)]
pub struct VulkanBuffer {
    pub buffer: vk::Buffer,
    pub allocation: vk_mem::Allocation,
    pub create_info: types::HalBufferDescriptor,
    pub mapped_ptr: Option<*mut u8>,
}

// Mapped ptr is not safely sendable. However, it is just a ptr to some memory,
// only referred to by this buffer, so this is safe.
unsafe impl Send for VulkanBuffer {}

impl Buffer<Vulkan> for VulkanBuffer {
    #[cfg_attr(feature = "trace", tracing::instrument(skip(data), fields(len=data.len())))]
    unsafe fn write(
        &mut self,
        _device: &VulkanDevice,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            let b = self.mapped_ptr.unwrap().add(offset as usize);
            let slice = std::slice::from_raw_parts_mut(b, data.len());
            slice.copy_from_slice(data);
            Ok(())
        }
    }

    #[cfg_attr(feature = "trace", tracing::instrument(skip(data), fields(len=data.len())))]
    unsafe fn read(
        &mut self,
        _device: &VulkanDevice,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            let b = self.mapped_ptr.unwrap().add(offset as usize);
            let slice = std::slice::from_raw_parts(b as *const u8, data.len());
            data.copy_from_slice(slice);
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn map(
        &mut self,
        _device: &VulkanDevice,
    ) -> Result<*mut u8, <Vulkan as Backend>::Error> {
        Ok(self.mapped_ptr.unwrap())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn unmap(&mut self, _device: &VulkanDevice) -> Result<(), <Vulkan as Backend>::Error> {
        // Unmapping isn't necessary on vulkan as long as the mapped pointer isn't used
        // while it could be modified elsewhere
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(mut self, device: &VulkanDevice) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            if self.mapped_ptr.is_some() {
                device.alloc.unmap_memory(&mut self.allocation);
            }
            device
                .alloc
                .destroy_buffer(self.buffer, &mut self.allocation);
            Ok(())
        }
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

impl BindGroup<Vulkan> for VulkanBindGroup {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn update(
        &mut self,
        device: &VulkanDevice,
        _stream: &VulkanStream,
        _kernel: &<Vulkan as Backend>::Kernel,
        resources: &[HalBufferSlice<Vulkan>],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let mut writes = Vec::with_capacity(resources.len());
        let mut buffer_infos = Vec::with_capacity(resources.len());
        for resource in resources {
            buffer_infos.push(
                vk::DescriptorBufferInfo::default()
                    .buffer(resource.buffer.buffer)
                    .offset(resource.offset)
                    .range(resource.length),
            );
        }
        for (i, info) in buffer_infos.iter().enumerate() {
            writes.push(
                vk::WriteDescriptorSet::default()
                    .dst_set(self.inner)
                    .descriptor_count(1)
                    .dst_binding(i as u32)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(std::slice::from_ref(info)),
            );
        }
        unsafe {
            device.shared.functions.update_descriptor_sets(&writes, &[]);
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        stream: &VulkanStream,
        kernel: &<Vulkan as Backend>::Kernel,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            stream.shared.functions.free_descriptor_sets(
                kernel.descriptor_pools.lock().unwrap()[self.pool_idx as usize].pool,
                &[self.inner],
            )?;
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct VulkanSemaphore {
    inner: vk::Semaphore,
    current_value: Mutex<u64>,
}

impl Semaphore<Vulkan> for VulkanSemaphore {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn wait(&self, device: &VulkanInstance) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            device.shared.functions.supa_wait_semaphores(
                &vk::SemaphoreWaitInfo::default()
                    .semaphores(std::slice::from_ref(&self.inner))
                    .values(&[*self.current_value.lock().unwrap() + 1]),
                u64::MAX,
            )?;
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn is_signalled(
        &self,
        device: &VulkanInstance,
    ) -> Result<bool, <Vulkan as Backend>::Error> {
        Ok(unsafe {
            device
                .shared
                .functions
                .supa_get_semaphore_counter_value(self.inner)?
        } == *self.current_value.lock().unwrap() + 1)
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn signal(&mut self, device: &VulkanInstance) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            device.shared.functions.supa_signal_semaphore(
                &vk::SemaphoreSignalInfo::default()
                    .semaphore(self.inner)
                    .value(*self.current_value.lock().unwrap() + 1),
            )?;
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn reset(&mut self, _device: &VulkanInstance) -> Result<(), <Vulkan as Backend>::Error> {
        *self.current_value.lock().unwrap() += 1;
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(self, device: &VulkanInstance) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            device.shared.functions.destroy_semaphore(self.inner, None);
        }
        Ok(())
    }
}

#[derive(Debug)]
pub struct SharedDeviceInfo {
    functions: DeviceFunctions,
    queue_family_indices: Vec<u32>,
}

pub struct DeviceFunctions {
    device: ash::Device,
    sync2_device: Option<khr::synchronization2::Device>,
    timeline_device: Option<khr::timeline_semaphore::Device>,
}

impl Debug for DeviceFunctions {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{:?}", self.device.handle())
    }
}

impl Deref for DeviceFunctions {
    type Target = ash::Device;
    fn deref(&self) -> &Self::Target {
        &self.device
    }
}

impl DeviceFunctions {
    unsafe fn supa_signal_semaphore(
        &self,
        signal_info: &vk::SemaphoreSignalInfo<'_>,
    ) -> VkResult<()> {
        if let Some(dev) = &self.timeline_device {
            unsafe { dev.signal_semaphore(signal_info) }
        } else {
            unsafe { self.device.signal_semaphore(signal_info) }
        }
    }

    unsafe fn supa_get_semaphore_counter_value(&self, semaphore: vk::Semaphore) -> VkResult<u64> {
        if let Some(dev) = &self.timeline_device {
            unsafe { dev.get_semaphore_counter_value(semaphore) }
        } else {
            unsafe { self.device.get_semaphore_counter_value(semaphore) }
        }
    }

    unsafe fn supa_wait_semaphores(
        &self,
        wait_info: &vk::SemaphoreWaitInfo<'_>,
        timeout: u64,
    ) -> VkResult<()> {
        if let Some(dev) = &self.timeline_device {
            unsafe { dev.wait_semaphores(wait_info, timeout) }
        } else {
            unsafe { self.device.wait_semaphores(wait_info, timeout) }
        }
    }

    unsafe fn supa_queue_submit2(
        &self,
        queue: vk::Queue,
        submits: &[vk::SubmitInfo2<'_>],
        fence: vk::Fence,
    ) -> VkResult<()> {
        if let Some(dev) = &self.sync2_device {
            unsafe { dev.queue_submit2(queue, submits, fence) }
        } else {
            unsafe { self.queue_submit2(queue, submits, fence) }
        }
    }

    unsafe fn supa_cmd_pipeline_barrier2(
        &self,
        command_buffer: vk::CommandBuffer,
        dependency_info: &vk::DependencyInfoKHR<'_>,
    ) {
        if let Some(dev) = &self.sync2_device {
            unsafe {
                dev.cmd_pipeline_barrier2(command_buffer, dependency_info);
            }
        } else {
            unsafe {
                self.cmd_pipeline_barrier2(command_buffer, dependency_info);
            }
        }
    }
}
