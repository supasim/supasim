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
use std::fmt::Debug;
use std::sync::Mutex;
use std::{borrow::Cow, num::NonZero};
pub use wgpu::Backends;
use wgpu::RequestAdapterError;

pub use ::wgpu;

/// # Overview
/// The wgpu backend. It supports many backends on many systems, including WebGPU and metal. It also includes extra validation and synchronization.
/// Any issues not reported by wgpu on this backend are considered a bug.
///
/// ## Issues/workarounds
/// * Synchronization is nonexistent. Commands can be assumed to execute in order and not in parallel. For this reason performance is awful.
/// * Buffer mapping is weird. Expect to see simple reads/writes of dormant buffers blocking a device for the most recent submission.
/// * CPU->GPU synchronization is impossible, as semaphores are not actually exposed
/// * Any error should immediately result in a panic
/// * Expanding on the previous point, OOM issues are at best an immediate crash
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
            backend_options: wgpu::BackendOptions {
                dx12: wgpu::Dx12BackendOptions {
                    shader_compiler: wgpu::Dx12Compiler::default_dynamic_dxc(),
                },
                ..Default::default()
            },
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

        let mut features = wgpu::Features::SHADER_INT64;
        if adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
            features |= wgpu::Features::PIPELINE_CACHE;
        }
        if unified_memory {
            features |= wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        }
        if adapter
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_FD)
        {
            features |= wgpu::Features::VULKAN_EXTERNAL_MEMORY_FD;
        }
        if adapter
            .features()
            .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32)
        {
            features |= wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32;
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
            _instance: instance,
            adapter,
            device,
            queue,
            unified_memory,
        })
    }
}

#[derive(Debug)]
pub struct WgpuInstance {
    _instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
    unified_memory: bool,
}
impl WgpuInstance {
    /// # Safety
    /// * Currently unspecified safety requirements
    pub unsafe fn get_device_queue(&self) -> (wgpu::Device, wgpu::Queue) {
        (self.device.clone(), self.queue.clone())
    }
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
            map_buffers: true,
            map_buffer_while_gpu_use: false,
            upload_download_buffers: false,
            export_memory: false,
        }
    }

    #[tracing::instrument]
    unsafe fn can_share_memory_to_device(
        &mut self,
        device: &dyn Any,
    ) -> Result<bool, <Wgpu as Backend>::Error> {
        #[cfg(feature = "external_wgpu")]
        if let Some(info) = device.downcast_ref::<crate::WgpuDeviceExportInfo>() {
            return Ok(info.supports_external_memory() && self.get_properties().export_memory);
        }
        Ok(false)
    }

    #[tracing::instrument]
    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::KernelReflectionInfo,
        cache: Option<&mut <Wgpu as Backend>::KernelCache>,
    ) -> Result<<Wgpu as Backend>::Kernel, <Wgpu as Backend>::Error> {
        #[allow(clippy::uninit_vec)]
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
                            v.set_len(binary.len() / 4);
                            std::ptr::copy(
                                binary.as_ptr(),
                                v.as_mut_ptr() as *mut u8,
                                binary.len(),
                            );
                        }
                        Cow::Owned(v)
                    },
                ),
            });
        let mut layout_entries = Vec::new();
        for (i, &b) in reflection.buffers.iter().enumerate() {
            layout_entries.push(wgpu::BindGroupLayoutEntry {
                binding: i as u32,
                visibility: wgpu::ShaderStages::COMPUTE,
                ty: wgpu::BindingType::Buffer {
                    ty: wgpu::BufferBindingType::Storage { read_only: !b },
                    has_dynamic_offset: false,
                    min_binding_size: None,
                },
                count: None,
            });
        }
        let bgl = self
            .device
            .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &layout_entries,
            });
        let layout = self
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: None,
                bind_group_layouts: &[&bgl],
                push_constant_ranges: &[],
            });
        let pipeline = self
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: None,
                layout: Some(&layout),
                module: &kernel,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: cache.map(|a| &a.inner),
            });
        Ok(WgpuKernel {
            _kernel: kernel,
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
        self.device.poll(wgpu::PollType::Wait).unwrap();
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn create_recorder(
        &mut self,
    ) -> Result<<Wgpu as Backend>::CommandRecorder, <Wgpu as Backend>::Error> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Ok(WgpuCommandRecorder {
            inner: Some(encoder),
        })
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
            let thing = std::mem::take(&mut a.command_recorder.inner).unwrap();
            thing.finish()
        }));
        for info in infos {
            if let Some(signal) = info.signal_semaphore {
                *signal.inner.lock().unwrap() = Some(idx.clone());
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
            usage: match alloc_info.memory_type {
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
                HalBufferType::UploadDownload => unreachable!(),
            },
            mapped_at_creation: false,
        });
        Ok(WgpuBuffer {
            inner: Box::new(buffer.clone()),
            slice: None,
            mapped_slice: None,
            map_mut: match alloc_info.memory_type {
                HalBufferType::Download => Some(false),
                HalBufferType::Upload => Some(true),
                _ => None,
            },
            create_info: *alloc_info,
        })
    }

    #[tracing::instrument]
    unsafe fn destroy_buffer(
        &mut self,
        mut buffer: <Wgpu as Backend>::Buffer,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        if buffer.mapped_slice.is_some() {
            // Drop the mapped slice first
            buffer.mapped_slice = None;
            buffer.slice = None;
            buffer.inner.unmap();
        }
        // Destroyed on drop
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn write_buffer(
        &mut self,
        buffer: &mut <Wgpu as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let was_mapped = buffer.mapped_slice.is_some();
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
        buffer: &mut <Wgpu as Backend>::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let was_mapped = buffer.mapped_slice.is_some();
        let ptr = unsafe { self.map_buffer(buffer)?.add(offset as usize) };
        unsafe { data.clone_from_slice(std::slice::from_raw_parts(ptr, data.len())) };
        if !was_mapped {
            unsafe { self.unmap_buffer(buffer)? };
        }
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn map_buffer(
        &mut self,
        buffer: &mut <Wgpu as Backend>::Buffer,
    ) -> Result<*mut u8, <Wgpu as Backend>::Error> {
        let ptr = match &mut buffer.mapped_slice {
            Some(slice) => slice.as_mut_ptr(),
            None => {
                // Fuck it. Transmute is to not worry about lifetimes.
                // Current wgpu implementation only need the lifetime for
                // the reference to the inner buffer, not needing mut.
                // Since the inner buffer lives as long as the WgpuBuffer,
                // and will never be used mutably until it is destroyed
                // with the WgpuBuffer itself, this should be safe.
                unsafe {
                    buffer.slice = Some(std::mem::transmute::<
                        wgpu::BufferSlice<'_>,
                        wgpu::BufferSlice<'static>,
                    >(buffer.inner.slice(..)));
                }
                buffer.slice.as_ref().unwrap().map_async(
                    if buffer.map_mut.unwrap() {
                        wgpu::MapMode::Write
                    } else {
                        wgpu::MapMode::Read
                    },
                    |_| (),
                );
                // In theory map_async will go through after doing this kind of blocking wait.
                // This might change in the future, making wgpu a volatile backend.
                // Also, this is dumb as shit.
                self.device.poll(wgpu::PollType::Wait).unwrap();
                // Now that we know that the slice will "live forever", we can get its mapped range which
                // will likewise "live forever". I told you I knew what I was doing, borrow checker!
                buffer.mapped_slice = Some(buffer.slice.as_mut().unwrap().get_mapped_range_mut());
                buffer.mapped_slice.as_mut().unwrap().as_mut_ptr()
            }
        };

        Ok(ptr)
    }

    #[tracing::instrument]
    unsafe fn unmap_buffer(
        &mut self,
        buffer: &mut <Wgpu as Backend>::Buffer,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        buffer.mapped_slice = None;
        buffer.slice = None;
        buffer.inner.unmap();
        Ok(())
    }

    #[tracing::instrument]
    unsafe fn create_bind_group(
        &mut self,
        kernel: &mut <Wgpu as Backend>::Kernel,
        resources: &[HalBufferSlice<Wgpu>],
    ) -> Result<<Wgpu as Backend>::BindGroup, <Wgpu as Backend>::Error> {
        let entries: Vec<wgpu::BindGroupEntry<'_>> = resources
            .iter()
            .enumerate()
            .map(|(i, slice)| wgpu::BindGroupEntry {
                binding: i as u32,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &slice.buffer.inner,
                    offset: slice.offset,
                    size: NonZero::<u64>::new(slice.len),
                }),
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
        resources: &[HalBufferSlice<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        unsafe {
            let new_bg = self.create_bind_group(kernel, resources)?;
            self.destroy_bind_group(kernel, std::mem::replace(bg, new_bg))?;
        }
        // TODO: work on bindless
        Ok(())
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
            inner: Mutex::new(None),
            device: self.device.clone(),
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
#[derive(Debug)]
pub struct WgpuKernel {
    _kernel: wgpu::ShaderModule,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}
impl Kernel<Wgpu> for WgpuKernel {}
#[derive(Debug)]
pub struct WgpuBuffer {
    /// This can't be moved for fear of UB lol. This is because buffer
    /// mapping expands the lifetime of a reference to it to 'static.
    inner: Box<wgpu::Buffer>,
    slice: Option<wgpu::BufferSlice<'static>>,
    mapped_slice: Option<wgpu::BufferViewMut<'static>>,
    map_mut: Option<bool>,
    create_info: HalBufferDescriptor,
}
unsafe impl Send for WgpuBuffer {}
unsafe impl Sync for WgpuBuffer {}
impl Buffer<Wgpu> for WgpuBuffer {
    unsafe fn share_to_device(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
        external_device: &dyn Any,
    ) -> Result<Box<dyn Any>, <Wgpu as Backend>::Error> {
        #[cfg(feature = "external_wgpu")]
        if let Some(info) = external_device.downcast_ref::<crate::WgpuDeviceExportInfo>() {
            let memory_obj = unsafe { self.export(instance)? };
            return Ok(Box::new(unsafe {
                info.import_external_memory(memory_obj, self.create_info)
            }));
        }
        Err(WgpuError::ExternalMemoryExport)
    }
    unsafe fn export(
        &mut self,
        _instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<ExternalMemoryObject, <Wgpu as Backend>::Error> {
        Err(WgpuError::ExternalMemoryExport)
    }
}
#[derive(Debug)]
pub struct WgpuCommandRecorder {
    inner: Option<wgpu::CommandEncoder>,
}
impl CommandRecorder<Wgpu> for WgpuCommandRecorder {
    #[tracing::instrument]
    unsafe fn record_commands(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
        commands: &mut [BufferCommand<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let r = self.inner.as_mut().unwrap();
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
                BufferCommand::ZeroMemory { buffer } => {
                    r.clear_buffer(&buffer.buffer.inner, buffer.offset, Some(buffer.len));
                }
                BufferCommand::DispatchKernel {
                    kernel,
                    bind_group,
                    push_constants: _,
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
                    drop(pass);
                }
                BufferCommand::MemoryTransfer { .. } => todo!(),
                BufferCommand::UpdateBindGroup { .. } => {
                    unreachable!()
                }
                BufferCommand::MemoryBarrier { .. } | BufferCommand::PipelineBarrier { .. } => (),
                BufferCommand::Dummy => (),
            }
        }

        Ok(())
    }
    #[tracing::instrument]
    unsafe fn record_dag(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
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
    inner: Mutex<Option<wgpu::SubmissionIndex>>,
    device: wgpu::Device,
}
unsafe impl Send for WgpuSemaphore {}
unsafe impl Sync for WgpuSemaphore {}
impl Debug for WgpuSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WgpuSemaphore")
    }
}
impl Semaphore<Wgpu> for WgpuSemaphore {
    #[tracing::instrument]
    unsafe fn wait(&self) -> Result<(), <Wgpu as Backend>::Error> {
        if let Some(a) = (*self.inner.lock().unwrap()).clone() {
            self.device
                .poll(wgpu::PollType::WaitForSubmissionIndex(a.clone()))
                .map_err(|_| WgpuError::PollTimeout)?;
        }
        *self.inner.lock().unwrap() = None;
        Ok(())
    }
    #[tracing::instrument]
    unsafe fn is_signalled(&self) -> Result<bool, <Wgpu as Backend>::Error> {
        if self.inner.lock().unwrap().is_some() {
            Ok(self
                .device
                .poll(wgpu::PollType::Poll)
                .is_ok_and(|a| a.wait_finished()))
        } else {
            Ok(true)
        }
    }
    #[tracing::instrument]
    unsafe fn signal(&self) -> Result<(), <Wgpu as Backend>::Error> {
        unreachable!()
    }
    unsafe fn reset(&self) -> Result<(), <Wgpu as Backend>::Error> {
        *self.inner.lock().unwrap() = None;
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
    #[error("Timeout when polling")]
    PollTimeout,
    #[error("A buffer export was attempted under invalid conditions")]
    ExternalMemoryExport,
}
impl Error<Wgpu> for WgpuError {
    fn is_out_of_device_memory(&self) -> bool {
        matches!(self, Self::WgpuError(wgpu::Error::OutOfMemory { .. }))
    }
    fn is_out_of_host_memory(&self) -> bool {
        false
    }
    fn is_timeout(&self) -> bool {
        matches!(self, Self::PollTimeout)
    }
}
