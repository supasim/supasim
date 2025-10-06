/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
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
    type Device = WgpuDevice;
    type Stream = WgpuStream;

    type Kernel = WgpuKernel;
    type Buffer = WgpuBuffer;
    type CommandRecorder = WgpuCommandRecorder;
    type BindGroup = WgpuBindGroup;
    type Semaphore = WgpuSemaphore;

    type Error = WgpuError;

    fn setup_default_descriptor() -> Result<InstanceDescriptor<Self>, Self::Error> {
        Self::create_instance(true, wgpu::Backends::PRIMARY, None)
    }
}
impl Wgpu {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    pub fn create_instance(
        advanced_dbg: bool,
        backends: wgpu::Backends,
        preset_unified_memory: Option<bool>,
    ) -> Result<InstanceDescriptor<Wgpu>, WgpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags: if advanced_dbg {
                wgpu::InstanceFlags::advanced_debugging()
            } else {
                wgpu::InstanceFlags::empty()
            },
            backends,
            backend_options: wgpu::BackendOptions {
                dx12: wgpu::Dx12BackendOptions {
                    shader_compiler: wgpu::Dx12Compiler::default_dynamic_dxc(),
                    ..Default::default()
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
        let unified_memory = preset_unified_memory.unwrap_or(false);

        let mut features = wgpu::Features::SHADER_INT64;
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
                experimental_features: wgpu::ExperimentalFeatures::disabled(),
            }))?;
        Ok(InstanceDescriptor {
            instance: WgpuInstance {
                _instance: instance,
                device: device.clone(),
            },
            devices: vec![DeviceDescriptor {
                device: WgpuDevice {
                    device: device.clone(),
                    unified_memory,
                },
                streams: vec![StreamDescriptor {
                    stream: WgpuStream { queue, device },
                    stream_type: crate::StreamType::ComputeAndTransfer,
                }],
                group_idx: None,
            }],
        })
    }
}

#[derive(Debug)]
pub struct WgpuDevice {
    device: wgpu::Device,
    unified_memory: bool,
}
impl Device<Wgpu> for WgpuDevice {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_buffer(
        &self,
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
            view: None,
            map_mut: match alloc_info.memory_type {
                HalBufferType::Download => Some(false),
                HalBufferType::Upload => Some(true),
                _ => None,
            },
        })
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn cleanup_cached_resources(
        &self,
        _instance: &<Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    fn get_properties(&self, _instance: &<Wgpu as Backend>::Instance) -> HalDeviceProperties {
        HalDeviceProperties {
            is_unified_memory: self.unified_memory,
            // TODO: look at this
            host_mappable_buffers: false,
        }
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct WgpuStream {
    queue: wgpu::Queue,
    device: wgpu::Device,
}
impl Stream<Wgpu> for WgpuStream {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_recorder(
        &self,
    ) -> Result<<Wgpu as Backend>::CommandRecorder, <Wgpu as Backend>::Error> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Ok(WgpuCommandRecorder {
            inner: Some(encoder),
        })
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
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
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn wait_for_idle(&mut self) -> Result<(), <Wgpu as Backend>::Error> {
        self.device
            .poll(wgpu::PollType::wait_indefinitely())
            .unwrap();
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_bind_group(
        &self,
        _device: &WgpuDevice,
        kernel: &<Wgpu as Backend>::Kernel,
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
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn cleanup_cached_resources(
        &self,
        _instance: &<Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _device: &mut <Wgpu as Backend>::Device,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct WgpuInstance {
    _instance: wgpu::Instance,
    device: wgpu::Device,
}
impl BackendInstance<Wgpu> for WgpuInstance {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    fn get_properties(&self) -> HalInstanceProperties {
        HalInstanceProperties {
            sync_mode: SyncMode::Automatic,
            kernel_lang: KernelTarget::Spirv {
                version: SpirvVersion::V1_0,
            },
            easily_update_bind_groups: false,
            semaphore_signal: false,
            map_buffers: true,
            map_buffer_while_gpu_use: false,
            upload_download_buffers: false,
        }
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn compile_kernel(
        &self,
        desc: KernelDescriptor,
    ) -> Result<<Wgpu as Backend>::Kernel, <Wgpu as Backend>::Error> {
        #[allow(clippy::uninit_vec)]
        let kernel = self
            .device
            .create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::SpirV(
                    if (desc.binary.as_ptr() as *const u32).is_aligned() {
                        Cow::Borrowed(bytemuck::cast_slice(desc.binary))
                    } else {
                        let mut v = Vec::with_capacity(desc.binary.len() / 4);
                        unsafe {
                            v.set_len(desc.binary.len() / 4);
                            std::ptr::copy(
                                desc.binary.as_ptr(),
                                v.as_mut_ptr() as *mut u8,
                                desc.binary.len(),
                            );
                        }
                        Cow::Owned(v)
                    },
                ),
            });
        let mut layout_entries = Vec::new();
        for (i, &b) in desc.reflection.buffers.iter().enumerate() {
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
                cache: None,
            });
        Ok(WgpuKernel {
            _kernel: kernel,
            pipeline,
            bgl,
        })
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_semaphore(
        &self,
    ) -> Result<<Wgpu as Backend>::Semaphore, <Wgpu as Backend>::Error> {
        Ok(WgpuSemaphore {
            inner: Mutex::new(None),
        })
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(self) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }
}
#[derive(Debug)]
pub struct WgpuKernel {
    _kernel: wgpu::ShaderModule,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}
impl Kernel<Wgpu> for WgpuKernel {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _instance: &<Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }
}
#[derive(Debug)]
pub struct WgpuBuffer {
    /// This can't be moved for fear of UB lol. This is because buffer
    /// mapping expands the lifetime of a reference to it to 'static.
    inner: Box<wgpu::Buffer>,
    view: Option<wgpu::BufferViewMut>,
    map_mut: Option<bool>,
}
unsafe impl Send for WgpuBuffer {}
unsafe impl Sync for WgpuBuffer {}
impl Buffer<Wgpu> for WgpuBuffer {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(mut self, _device: &WgpuDevice) -> Result<(), <Wgpu as Backend>::Error> {
        if self.view.is_some() {
            self.view = None;
            self.inner.unmap();
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn write(
        &mut self,
        device: &WgpuDevice,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let was_mapped = self.view.is_some();
        let ptr = unsafe { self.map(device)?.add(offset as usize) };
        unsafe { std::slice::from_raw_parts_mut(ptr, data.len()) }.clone_from_slice(data);
        if !was_mapped {
            unsafe { self.unmap(device)? };
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn read(
        &mut self,
        device: &WgpuDevice,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let was_mapped = self.view.is_some();
        let ptr = unsafe { self.map(device)?.add(offset as usize) };
        unsafe { data.clone_from_slice(std::slice::from_raw_parts(ptr, data.len())) };
        if !was_mapped {
            unsafe { self.unmap(device)? };
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn map(&mut self, device: &WgpuDevice) -> Result<*mut u8, <Wgpu as Backend>::Error> {
        let ptr = match &mut self.view {
            Some(slice) => slice.as_mut_ptr(),
            None => {
                let slice = self.inner.slice(..);

                slice.map_async(
                    if self.map_mut.unwrap() {
                        wgpu::MapMode::Write
                    } else {
                        wgpu::MapMode::Read
                    },
                    |_| (),
                );
                // In theory map_async will go through after doing this kind of blocking wait.
                // This might change in the future, making wgpu a volatile backend.
                // Also, this is dumb as shit.
                device
                    .device
                    .poll(wgpu::PollType::wait_indefinitely())
                    .unwrap();
                // Now that we know that the slice will "live forever", we can get its mapped range which
                // will likewise "live forever". I told you I knew what I was doing, borrow checker!
                self.view = Some(slice.get_mapped_range_mut());
                self.view.as_mut().unwrap().as_mut_ptr()
            }
        };

        Ok(ptr)
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn unmap(&mut self, _device: &WgpuDevice) -> Result<(), <Wgpu as Backend>::Error> {
        self.view = None;
        self.inner.unmap();
        Ok(())
    }
}
#[derive(Debug)]
pub struct WgpuCommandRecorder {
    inner: Option<wgpu::CommandEncoder>,
}
impl CommandRecorder<Wgpu> for WgpuCommandRecorder {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn record_commands(
        &mut self,
        _instance: &WgpuStream,
        commands: &[BufferCommand<Wgpu>],
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
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn clear(
        &mut self,
        stream: &<Wgpu as Backend>::Stream,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        *self = unsafe { stream.create_recorder()? };
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _stream: &<Wgpu as Backend>::Stream,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }
}
#[derive(Debug)]
pub struct WgpuBindGroup {
    inner: wgpu::BindGroup,
}
impl BindGroup<Wgpu> for WgpuBindGroup {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn update(
        &mut self,
        device: &WgpuDevice,
        stream: &<Wgpu as Backend>::Stream,
        kernel: &<Wgpu as Backend>::Kernel,
        buffers: &[HalBufferSlice<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        unsafe {
            let new_bg = stream.create_bind_group(device, kernel, buffers)?;
            std::mem::replace(self, new_bg).destroy(stream, kernel)?;
        }
        // TODO: work on bindless
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _stream: &<Wgpu as Backend>::Stream,
        _kernel: &<Wgpu as Backend>::Kernel,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }
}
pub struct WgpuSemaphore {
    inner: Mutex<Option<wgpu::SubmissionIndex>>,
}
unsafe impl Send for WgpuSemaphore {}
unsafe impl Sync for WgpuSemaphore {}
impl Debug for WgpuSemaphore {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("WgpuSemaphore")
    }
}
impl Semaphore<Wgpu> for WgpuSemaphore {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn wait(&self, device: &WgpuInstance) -> Result<(), <Wgpu as Backend>::Error> {
        if let Some(a) = self.inner.lock().unwrap().clone() {
            device
                .device
                .poll(wgpu::PollType::Wait {
                    submission_index: Some(a.clone()),
                    timeout: None,
                })
                .map_err(|_| WgpuError::PollTimeout)?;
        }
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn is_signalled(&self, device: &WgpuInstance) -> Result<bool, <Wgpu as Backend>::Error> {
        if self.inner.lock().unwrap().is_some() {
            Ok(device
                .device
                .poll(wgpu::PollType::Poll)
                .is_ok_and(|a| a.wait_finished()))
        } else {
            Ok(true)
        }
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn signal(&mut self, _device: &WgpuInstance) -> Result<(), <Wgpu as Backend>::Error> {
        unreachable!()
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn reset(&mut self, _device: &WgpuInstance) -> Result<(), <Wgpu as Backend>::Error> {
        *self.inner.lock().unwrap() = None;
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _device: &<Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
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
