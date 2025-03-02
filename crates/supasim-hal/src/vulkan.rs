use std::sync::Mutex;

use crate::{
    Backend, BackendInstance, BindGroup, Buffer, CommandRecorder, CompiledKernel, Fence,
    GpuResource, MappedBuffer, PipelineCache, RecorderSubmitInfo, Semaphore,
};
use ash::vk;
use gpu_allocator::{
    vulkan::{Allocation, AllocationCreateDesc, Allocator, AllocatorCreateDesc},
    AllocationError, AllocationSizes, AllocatorDebugSettings,
};
use shaders::ShaderResourceType;
use thiserror::Error;
use types::{to_static_lifetime, BufferDescriptor, InstanceProperties};

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
    pub fn create_instance(_debug: bool) -> VulkanInstance {
        //let instance = ash::Instance::load(&vk::InstanceCreateInfo::default().application_info(), instance)
        todo!()
    }
    pub fn from_existing(
        debug: bool,
        instance: ash::Instance,
        device: ash::Device,
        phyd: vk::PhysicalDevice,
        queue: vk::Queue,
        queue_family_idx: u32,
        renderer_queue_family_idx: Option<u32>,
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
                instance,
                device,
                phyd,
                alloc: Mutex::new(alloc),
                queue,
                queue_family_idx,
                renderer_queue_family_idx,
                pool,
            })
        }
    }
}
#[derive(Error, Debug)]
pub enum VulkanError {
    #[error("{0}")]
    VulkanRaw(#[from] vk::Result),
    #[error("{0}")]
    AllocationError(#[from] gpu_allocator::AllocationError),
    #[error("An unsupported dispatch mode(indirect) was called")]
    DispatchModeUnsupported,
    #[error("{0}")]
    LockError(String),
    #[error("Using compute buffers from an external renderer is currently unsupported")]
    ExternalRendererUnsupported,
}
impl crate::Error<Vulkan> for VulkanError {
    fn is_out_of_device_memory(&self) -> bool {
        match self {
            Self::VulkanRaw(e) => *e == vk::Result::ERROR_OUT_OF_DEVICE_MEMORY,
            Self::AllocationError(e) => matches!(e, AllocationError::OutOfMemory),
            _ => false,
        }
    }
}
#[allow(dead_code)]
pub struct VulkanInstance {
    instance: ash::Instance,
    device: ash::Device,
    phyd: vk::PhysicalDevice,
    alloc: Mutex<Allocator>,
    queue: vk::Queue,
    queue_family_idx: u32,
    renderer_queue_family_idx: Option<u32>,
    pool: vk::CommandPool,
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
        reflection: shaders::ShaderReflectionInfo,
        cache: Option<&mut VulkanPipelineCache>,
    ) -> Result<<Vulkan as Backend>::Kernel, <Vulkan as Backend>::Error> {
        unsafe {
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
            let pipeline_layout = self
                .device
                .create_pipeline_layout(&vk::PipelineLayoutCreateInfo::default(), None)?;
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
            let pipeline_create_info = vk::ComputePipelineCreateInfo::default();
            let pipeline = self
                .device
                .create_compute_pipelines(cache, &[pipeline_create_info], None)
                .map_err(|e| e.1)?[0];
            drop(_cache_lock);
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
            let requirements = self.device.get_buffer_memory_requirements(buffer);
            use types::MemoryType::*;
            let allocation = self
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
                        UploadDownload => gpu_allocator::MemoryLocation::GpuToCpu,
                    },
                    linear: true,
                    allocation_scheme: gpu_allocator::vulkan::AllocationScheme::GpuAllocatorManaged, // TODO: verify this is correct
                })?;
            self.device
                .bind_buffer_memory(buffer, allocation.memory(), allocation.offset())?;
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
            let create_info = vk::PipelineCacheCreateInfo::default()
                .initial_data(initial_data)
                .flags(vk::PipelineCacheCreateFlags::EXTERNALLY_SYNCHRONIZED);
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
        resources: &mut [&mut crate::GpuResource<Vulkan>],
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
            unsafe {
                let create_info = vk::DescriptorPoolCreateInfo::default();
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
                        GpuResource::Buffer(VulkanBuffer {
                            create_info: BufferDescriptor { uniform, .. },
                            ..
                        }) => {
                            if *uniform {
                                vk::DescriptorType::UNIFORM_BUFFER
                            } else {
                                vk::DescriptorType::STORAGE_BUFFER
                            }
                        }
                    });
                match resource {
                    GpuResource::Buffer(b) => {
                        buffer_infos.push(
                            vk::DescriptorBufferInfo::default()
                                .buffer(b.buffer)
                                .offset(b.allocation.offset())
                                .range(b.allocation.size()),
                        );
                        // TODO: fix Undefined behavior
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
    fn destroy_bind_groups(
        &mut self,
        kernel: &mut <Vulkan as Backend>::Kernel,
        sets: &mut [&mut <Vulkan as Backend>::BindGroup],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let mut sets_per_pool = Vec::new();
        sets_per_pool.resize_with(kernel.descriptor_pools.len(), Vec::new);
        for set in sets {
            sets_per_pool[set.pool_idx as usize].push(set.inner);
        }
        for (i, pool) in kernel.descriptor_pools.iter_mut().enumerate() {
            if !sets_per_pool[i].is_empty() {
                unsafe {
                    self.device
                        .free_descriptor_sets(pool.pool, &sets_per_pool[i])?;
                }
                pool.current_size -= sets_per_pool[i].len() as u32;
            }
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
                .map(|cb| VulkanCommandRecorder { cb })
                .collect())
        }
    }
    fn destroy_recorders(
        &mut self,
        recorders: Vec<<Vulkan as Backend>::CommandRecorder>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let cb: Vec<vk::CommandBuffer> = recorders.into_iter().map(|cb| cb.cb).collect();
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
            semaphore_infos.extend(submit.wait_semaphores.iter().map(|a| a.inner));
            cbs.extend(submit.command_recorders.iter().map(|a| a.cb));
        }
        let signal_start = semaphore_infos.len();
        for submit in infos.iter() {
            semaphore_infos.extend(submit.out_semaphores.iter().map(|a| a.inner));
        }
        let mut submits = Vec::new();
        let mut cb_idx = 0;
        let mut wait_idx = 0;
        let mut signal_idx = signal_start;
        for submit in infos.iter() {
            submits.push(
                vk::SubmitInfo::default()
                    .command_buffers(&cbs[cb_idx..cb_idx + submit.command_recorders.len()])
                    .wait_semaphores(
                        &semaphore_infos[wait_idx..wait_idx + submit.wait_semaphores.len()],
                    )
                    .signal_semaphores(
                        &semaphore_infos[signal_idx..signal_idx + submit.out_semaphores.len()],
                    ),
            );
            cb_idx += submit.command_recorders.len();
            wait_idx += submit.wait_semaphores.len();
            signal_idx += submit.out_semaphores.len();
        }
        unsafe {
            self.device.queue_submit(
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
                    .reset_command_buffer(cb.cb, vk::CommandBufferResetFlags::empty())?;
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
            let ptr = self.device.map_memory(
                buffer.allocation.memory(),
                buffer.allocation.offset() + offset,
                size,
                vk::MemoryMapFlags::empty(),
            )?;
            Ok(VulkanMappedBuffer {
                slice: std::slice::from_raw_parts_mut(ptr as *mut u8, size as usize),
                buffer_offset: offset,
            })
        }
    }
    fn flush_mapped_buffer(
        &self,
        buffer: &mut <Vulkan as Backend>::Buffer,
        map: &mut <Vulkan as Backend>::MappedBuffer,
    ) -> Result<(), VulkanError> {
        if buffer.create_info.needs_flush {
            unsafe {
                let range = vk::MappedMemoryRange::default()
                    .memory(buffer.allocation.memory())
                    .offset(map.buffer_offset + buffer.allocation.offset())
                    .size(map.slice.len() as u64);
                self.device.flush_mapped_memory_ranges(&[range])?;
            }
        }
        Ok(())
    }
    fn update_mapped_buffer(
        &self,
        buffer: &mut <Vulkan as Backend>::Buffer,
        map: &mut <Vulkan as Backend>::MappedBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        if buffer.create_info.needs_flush {
            unsafe {
                let range = vk::MappedMemoryRange::default()
                    .memory(buffer.allocation.memory())
                    .offset(map.buffer_offset + buffer.allocation.offset())
                    .size(map.slice.len() as u64);
                self.device.invalidate_mapped_memory_ranges(&[range])?;
            }
        }
        Ok(())
    }
    fn unmap_buffer(
        &mut self,
        buffer: &mut <Vulkan as Backend>::Buffer,
        _map: <Vulkan as Backend>::MappedBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            self.device.unmap_memory(buffer.allocation.memory());
            Ok(())
        }
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
    fn create_fences(
        &mut self,
        num: u32,
    ) -> Result<Vec<<Vulkan as Backend>::Fence>, <Vulkan as Backend>::Error> {
        unsafe {
            let mut fences = Vec::new();
            for _ in 0..num {
                fences.push(VulkanFence {
                    inner: self.device.create_fence(
                        &vk::FenceCreateInfo::default().flags(vk::FenceCreateFlags::empty()),
                        None,
                    )?,
                });
            }
            Ok(fences)
        }
    }
    fn destroy_fences(
        &mut self,
        fences: Vec<<Vulkan as Backend>::Fence>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            for fence in fences {
                self.device.destroy_fence(fence.inner, None);
            }
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
    fn create_semaphores(
        &mut self,
        timeline: bool,
        num: u32,
    ) -> Result<Vec<<Vulkan as Backend>::Semaphore>, <Vulkan as Backend>::Error> {
        unsafe {
            let mut semaphores = Vec::new();
            let mut ext = vk::SemaphoreTypeCreateInfo::default().semaphore_type(if timeline {
                vk::SemaphoreType::TIMELINE
            } else {
                vk::SemaphoreType::BINARY
            });
            let create_info = vk::SemaphoreCreateInfo::default()
                .flags(vk::SemaphoreCreateFlags::empty())
                .push_next(&mut ext);
            for _ in 0..num {
                semaphores.push(VulkanSemaphore {
                    inner: self.device.create_semaphore(&create_info, None)?,
                });
            }
            Ok(semaphores)
        }
    }
    fn destroy_semaphores(
        &mut self,
        semaphores: Vec<<Vulkan as Backend>::Semaphore>,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        for semaphore in semaphores {
            unsafe {
                self.device.destroy_semaphore(semaphore.inner, None);
            }
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
    buffer_offset: u64,
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
    cb: vk::CommandBuffer,
}
impl VulkanCommandRecorder {}
impl CommandRecorder<Vulkan> for VulkanCommandRecorder {
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
                self.cb,
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
                self.cb,
                vk::PipelineBindPoint::COMPUTE,
                shader.pipeline,
            );
            instance.device.cmd_bind_descriptor_sets(
                self.cb,
                vk::PipelineBindPoint::COMPUTE,
                shader.pipeline_layout,
                0,
                &[descriptor_set.inner],
                &[],
            );
            instance.device.cmd_push_constants(
                self.cb,
                shader.pipeline_layout,
                vk::ShaderStageFlags::COMPUTE,
                0,
                push_constants,
            );
            instance.device.cmd_dispatch(
                self.cb,
                workgroup_dims[0],
                workgroup_dims[1],
                workgroup_dims[2],
            );
        }
        // TODO: Do sync stuff
        todo!()
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
