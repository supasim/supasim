#![allow(unused_variables, dead_code)]
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

use crate::*;
pub use ::wgpu::Backends;
use std::fmt::Debug;
use std::{borrow::Cow, cell::Cell, num::NonZero};
use wgpu::RequestAdapterError;

pub use ::wgpu;

#[derive(Clone, Copy, Debug)]
pub struct Wgpu;
impl Backend for Wgpu {
    type Instance = WgpuInstance;

    type Kernel = WgpuKernel;

    type Buffer = WgpuBuffer;

    type CommandRecorder = WgpuCommandRecorder;

    type BindGroup = WgpuBindGroup;

    type KernelCache = WgpuKernelCache;

    type Semaphore = WgpuSemaphore;

    type Error = WgpuError;
}
impl Wgpu {
    #[tracing::instrument]
    pub fn create_instance(
        advanced_dbg: bool,
        backends: wgpu::Backends,
        preset_unified_memory: Option<bool>,
    ) -> Result<WgpuInstance, WgpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags: if advanced_dbg {
                wgpu::InstanceFlags::advanced_debugging()
            } else {
                wgpu::InstanceFlags::debugging()
            },
            backends,
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .map_err(WgpuError::NoSuitableAdapters)?;
        let unified_memory = if let Some(a) = preset_unified_memory {
            a
        } else {
            let device_type = adapter.get_info().device_type;
            device_type == wgpu::DeviceType::Cpu || device_type == wgpu::DeviceType::IntegratedGpu
        };

        let mut features = wgpu::Features::empty();
        if adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
            features |= wgpu::Features::PIPELINE_CACHE;
        }
        if unified_memory {
            features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        let (device, queue) =
            pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
                trace: wgpu::Trace::Off,
            }))?;
        Ok(WgpuInstance {
            instance,
            adapter,
            device,
            queue,
            unified_memory,
        })
    }
}

#[derive(Debug)]
pub struct WgpuInstance {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    unified_memory: bool,
}
impl BackendInstance<Wgpu> for WgpuInstance {
    #[tracing::instrument]
    fn get_properties(&mut self) -> HalInstanceProperties {
        HalInstanceProperties {
            sync_mode: SyncMode::Automatic,
            pipeline_cache: self
                .adapter
                .features()
                .contains(wgpu::Features::PIPELINE_CACHE),
            kernel_lang: KernelTarget::Spirv {
                version: SpirvVersion::V1_0,
            },
            easily_update_bind_groups: false,
            semaphore_signal: false,
            // TODO: detect unified memory
            is_unified_memory: self.unified_memory,
        }
    }

    #[tracing::instrument]
    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::KernelReflectionInfo,
        cache: Option<&mut <Wgpu as Backend>::KernelCache>,
    ) -> Result<<Wgpu as Backend>::Kernel, <Wgpu as Backend>::Error> {
        let kernel = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::SpirV(
                    if (binary.as_ptr() as *const u32).is_aligned() {
                        Cow::Borrowed(bytemuck::cast_slice(binary))
                    } else {
                        let mut v = Vec::with_capacity(binary.len() / 4);
                        unsafe {
                            std::ptr::copy(
                                binary.as_ptr(),
                                v.as_mut_ptr() as *mut u8,
                                binary.len(),
                            );
                            v.set_len(binary.len() / 4);
                        }
                        Cow::Owned(v)
                    },
                ),
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: None,
                module: &kernel,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: cache.map(|a| &a.inner),
            });
        let bgl = pipeline.get_bind_group_layout(0);
        Ok(WgpuKernel {
            kernel,
            pipeline,
            bgl,
        })
    }

    #[tracing::instrument]
    unsafe fn create_kernel_cache(
        &mut self,
        initial_data: &[u8],
    ) -> Result<<Wgpu as Backend>::KernelCache, <Wgpu as Backend>::Error> {
        unsafe {
            let cache = self
                .device
                .create_pipeline_cache(&wgpu::PipelineCacheDescriptor {
                    label: None,
                    data: if initial_data.is_empty() {
                        None
                    } else {
                        Some(initial_data)
                    },
                    fallback: false,
                });
            Ok(WgpuKernelCache { inner: cache })
        }
    }

    #[tracing::instrument]
    unsafe fn destroy_kernel_cache(
        &mut self,
        cache: <Wgpu as Backend>::KernelCache,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn get_kernel_cache_data(
        &mut self,
        cache: &mut <Wgpu as Backend>::KernelCache,
    ) -> Result<Vec<u8>, <Wgpu as Backend>::Error> {
        Ok(cache.inner.get_data().unwrap())
    }

    #[tracing::instrument]
    unsafe fn destroy_kernel(
        &mut self,
        kernel: <Wgpu as Backend>::Kernel,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn wait_for_idle(&mut self) -> Result<(), <Wgpu as Backend>::Error> {
        self.device.poll(wgpu::MaintainBase::Wait)?;
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn create_recorder(
        &mut self,
    ) -> Result<<Wgpu as Backend>::CommandRecorder, <Wgpu as Backend>::Error> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Ok(WgpuCommandRecorder::Unrecorded(encoder))
    }

    #[tracing::instrument]
    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let mut new_recorders = Vec::new();
        for _ in 0..infos.len() {
            new_recorders.push(unsafe { self.create_recorder()? });
        }
        let idx = self.queue.submit(infos.iter_mut().map(|a| {
            let thing = std::mem::replace(a.command_recorder, new_recorders.pop().unwrap());
            match thing {
                WgpuCommandRecorder::Unrecorded(a) => a.finish(),
                WgpuCommandRecorder::Recorded(a) => a,
            }
        }));
        for info in infos {
            if let Some(signal) = info.signal_semaphore {
                signal.inner.set(Some(idx.clone()));
            }
        }
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn destroy_recorder(
        &mut self,
        recorder: <Wgpu as Backend>::CommandRecorder,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn create_buffer(
        &mut self,
        alloc_info: &HalBufferDescriptor,
    ) -> Result<<Wgpu as Backend>::Buffer, <Wgpu as Backend>::Error> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: alloc_info.size,
            usage: {
                let mut usage = match alloc_info.memory_type {
                    HalBufferType::Storage => {
                        wgpu::BufferUsages::STORAGE
                            | wgpu::BufferUsages::COPY_SRC
                            | wgpu::BufferUsages::COPY_DST
                    }
                    HalBufferType::Download => {
                        wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::MAP_READ
                    }
                    HalBufferType::Upload => {
                        wgpu::BufferUsages::COPY_SRC | wgpu::BufferUsages::MAP_WRITE
                    }
                    HalBufferType::Other => wgpu::BufferUsages::COPY_DST,
                    HalBufferType::Any => {
                        unreachable!()
                    }
                };

                if alloc_info.indirect_capable {
                    usage |= wgpu::BufferUsages::INDIRECT | wgpu::BufferUsages::STORAGE;
                }
                if alloc_info.uniform {
                    usage |= wgpu::BufferUsages::UNIFORM;
                }
                usage
            },
            mapped_at_creation: false,
        });
        Ok(WgpuBuffer {
            inner: buffer.clone(),
            mapped_ptr: Cell::new(None),
            map_mut: match alloc_info.memory_type {
                HalBufferType::Download => Some(false),
                HalBufferType::Upload => Some(true),
                _ => None,
            },
        })
    }

    #[tracing::instrument]
    unsafe fn destroy_buffer(
        &mut self,
        buffer: <Wgpu as Backend>::Buffer,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        if buffer.mapped_ptr.get().is_some() {
            buffer.inner.unmap();
        }
        // Destroyed on drop
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn write_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let was_mapped = buffer.mapped_ptr.get().is_some();
        let ptr = unsafe { self.map_buffer(buffer)?.add(offset as usize) };
        unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) }.clone_from_slice(data);
        if !was_mapped {
            unsafe { self.unmap_buffer(buffer)? };
        }
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn read_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let was_mapped = buffer.mapped_ptr.get().is_some();
        let ptr = unsafe { self.map_buffer(buffer)?.add(offset as usize) };
        unsafe { data.clone_from_slice(std::slice::from_raw_parts(ptr, data.len())) };
        if !was_mapped {
            unsafe { self.unmap_buffer(buffer)? };
        }
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn create_bind_group(
        &mut self,
        kernel: &mut <Wgpu as Backend>::Kernel,
        resources: &[GpuResource<Wgpu>],
    ) -> Result<<Wgpu as Backend>::BindGroup, <Wgpu as Backend>::Error> {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
            .iter()
            .enumerate()
            .map(|(i, a)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: match a {
                    GpuResource::Buffer {
                        buffer,
                        offset,
                        len: size,
                    } => wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &buffer.inner,
                        offset: *offset,
                        size: NonZero::<u64>::new(*size),
                    }),
                },
            })
            .collect();
        let bg = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &kernel.bgl,
            entries: &entries,
        });
        Ok(WgpuBindGroup { inner: bg })
    }

    #[tracing::instrument]
    unsafe fn update_bind_group(
        &mut self,
        bg: &mut <Wgpu as Backend>::BindGroup,
        kernel: &mut <Wgpu as Backend>::Kernel,
        resources: &[GpuResource<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        *bg = unsafe { self.create_bind_group(kernel, resources)? };
        todo!()
    }

    #[tracing::instrument]
    unsafe fn destroy_bind_group(
        &mut self,
        kernel: &mut <Wgpu as Backend>::Kernel,
        bind_group: <Wgpu as Backend>::BindGroup,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn create_semaphore(
        &mut self,
    ) -> Result<<Wgpu as Backend>::Semaphore, <Wgpu as Backend>::Error> {
        Ok(WgpuSemaphore {
            inner: Cell::new(None),
        })
    }

    #[tracing::instrument]
    unsafe fn destroy_semaphore(
        &mut self,
        semaphore: <Wgpu as Backend>::Semaphore,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn destroy(self) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }
}
impl WgpuInstance {
    #[tracing::instrument]
    unsafe fn map_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
    ) -> Result<*mut u8, <Wgpu as Backend>::Error> {
        // Todo: this is awful code
        let ptr = match buffer.mapped_ptr.get() {
            Some(ptr) => ptr,
            None => {
                let slice = buffer.inner.slice(..);
                slice.map_async(
                    if buffer.map_mut.unwrap() {
                        wgpu::MapMode::Write
                    } else {
                        wgpu::MapMode::Read
                    },
                    |_| (),
                );
                self.device.poll(wgpu::PollType::Wait)?;
                slice.get_mapped_range_mut().as_mut_ptr()
            }
        };
        buffer.mapped_ptr.set(Some(ptr));
        Ok(ptr)
    }
    #[tracing::instrument]
    unsafe fn unmap_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        buffer.inner.unmap();
        buffer.mapped_ptr.set(None);
        Ok(())
    }
}
#[derive(Debug)]
pub struct WgpuKernel {
    kernel: wgpu::ShaderModule,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}
impl Kernel<Wgpu> for WgpuKernel {}
#[derive(Debug)]
pub struct WgpuBuffer {
    inner: wgpu::Buffer,
    mapped_ptr: Cell<Option<*mut u8>>,
    map_mut: Option<bool>,
}
impl Buffer<Wgpu> for WgpuBuffer {}
#[derive(Debug)]
pub enum WgpuCommandRecorder {
    Unrecorded(wgpu::CommandEncoder),
    Recorded(wgpu::CommandBuffer),
}
impl CommandRecorder<Wgpu> for WgpuCommandRecorder {
    #[tracing::instrument]
    unsafe fn record_commands(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
        commands: &mut [BufferCommand<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let r = match self {
            Self::Unrecorded(r) => r,
            _ => unreachable!(),
        };
        for command in commands {
            match command {
                BufferCommand::CopyBuffer {
                    src_buffer,
                    dst_buffer,
                    src_offset,
                    dst_offset,
                    len,
                } => {
                    r.copy_buffer_to_buffer(
                        &src_buffer.inner,
                        *src_offset,
                        &dst_buffer.inner,
                        *dst_offset,
                        *len,
                    );
                }
                BufferCommand::DispatchKernel {
                    kernel,
                    bind_group,
                    push_constants,
                    workgroup_dims,
                } => {
                    let mut pass = r.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&kernel.pipeline);
                    pass.set_bind_group(0, &bind_group.inner, &[]);
                    pass.dispatch_workgroups(
                        workgroup_dims[0],
                        workgroup_dims[1],
                        workgroup_dims[2],
                    );
                }
                BufferCommand::UpdateBindGroup {
                    bg,
                    kernel,
                    resources,
                } => {
                    unreachable!()
                }
                BufferCommand::MemoryBarrier { .. } | BufferCommand::PipelineBarrier { .. } => (),
            }
        }

        Ok(())
    }
    #[tracing::instrument]
    unsafe fn record_dag(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
        resources: &[&GpuResource<Wgpu>],
        dag: &mut Dag<BufferCommand<Wgpu>>,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        unreachable!()
    }
    #[tracing::instrument]
    unsafe fn clear(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        *self = unsafe { instance.create_recorder()? };
        Ok(())
    }
}
#[derive(Debug)]
pub struct WgpuBindGroup {
    inner: wgpu::BindGroup,
}
impl BindGroup<Wgpu> for WgpuBindGroup {}
#[derive(Debug)]
pub struct WgpuKernelCache {
    inner: wgpu::PipelineCache,
}
impl KernelCache<Wgpu> for WgpuKernelCache {}
pub struct WgpuSemaphore {
    inner: Cell<Option<wgpu::SubmissionIndex>>,
}
impl Debug for WgpuSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WgpuSemaphore")
    }
}
impl Semaphore<Wgpu> for WgpuSemaphore {
    #[tracing::instrument]
    unsafe fn wait(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        if let Some(a) = self.inner.get_mut() {
            instance
                .device
                .poll(wgpu::PollType::WaitForSubmissionIndex(a.clone()))?;
        }
        self.inner.set(None);
        Ok(())
    }
    #[tracing::instrument]
    unsafe fn is_signalled(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<bool, <Wgpu as Backend>::Error> {
        if self.inner.get_mut().is_some() {
            Ok(instance
                .device
                .poll(wgpu::PollType::Poll)
                .is_ok_and(|a| a.wait_finished()))
        } else {
            Ok(true)
        }
    }
    #[tracing::instrument]
    unsafe fn signal(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        unreachable!()
    }
    unsafe fn reset(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        self.inner.set(None);
        Ok(())
    }
}
#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("Wgpu internal error: {0}")]
    WgpuError(#[from] wgpu::Error),
    #[error("No suitable adapters found: {0}")]
    NoSuitableAdapters(#[from] RequestAdapterError),
    #[error("Error requesting wgpu device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
    #[error("Error polling: {0}")]
    PollError(#[from] wgpu::PollError),
}
impl Error<Wgpu> for WgpuError {
    fn is_out_of_device_memory(&self) -> bool {
        matches!(self, Self::WgpuError(wgpu::Error::OutOfMemory { .. }))
    }
    fn is_out_of_host_memory(&self) -> bool {
        false
    }
    fn is_timeout(&self) -> bool {
        matches!(self, Self::PollError(wgpu::PollError::Timeout))
    }
}
