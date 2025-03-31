#![allow(unused_variables, dead_code)]

use std::{borrow::Cow, cell::Cell, num::NonZero};

use crate::*;
use ::wgpu;

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

    type Event = WgpuEvent;

    type Error = WgpuError;
}
impl Wgpu {
    pub fn create_instance(advanced_dbg: bool) -> Result<WgpuInstance, WgpuError> {
        let instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            flags: if advanced_dbg {
                wgpu::InstanceFlags::advanced_debugging()
            } else {
                wgpu::InstanceFlags::debugging()
            },
            ..Default::default()
        });
        let adapter = pollster::block_on(instance.request_adapter(&wgpu::RequestAdapterOptions {
            power_preference: wgpu::PowerPreference::HighPerformance,
            force_fallback_adapter: false,
            compatible_surface: None,
        }))
        .ok_or(WgpuError::NoSuitableAdapters)?;
        let mut features = wgpu::Features::MAPPABLE_PRIMARY_BUFFERS;
        if adapter.features().contains(wgpu::Features::PIPELINE_CACHE) {
            features |= wgpu::Features::PIPELINE_CACHE;
        }
        let (device, queue) = pollster::block_on(adapter.request_device(
            &wgpu::DeviceDescriptor {
                label: None,
                required_features: features,
                required_limits: wgpu::Limits::downlevel_defaults(),
                memory_hints: wgpu::MemoryHints::Performance,
            },
            None,
        ))?;
        Ok(WgpuInstance {
            instance,
            adapter,
            device,
            queue,
        })
    }
}

pub struct WgpuInstance {
    instance: wgpu::Instance,
    adapter: wgpu::Adapter,
    device: wgpu::Device,
    queue: wgpu::Queue,
}
impl BackendInstance<Wgpu> for WgpuInstance {
    fn get_properties(&mut self) -> InstanceProperties {
        InstanceProperties {
            sync_mode: SyncMode::Automatic,
            indirect: false,
            pipeline_cache: self
                .adapter
                .features()
                .contains(wgpu::Features::PIPELINE_CACHE),
            shader_type: ShaderTarget::Spirv {
                version: SpirvVersion::V1_0,
            },
            easily_update_bind_groups: false,
        }
    }

    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::ShaderReflectionInfo,
        cache: Option<&mut <Wgpu as Backend>::KernelCache>,
    ) -> Result<<Wgpu as Backend>::Kernel, <Wgpu as Backend>::Error> {
        let shader = self
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
                module: &shader,
                entry_point: Some(&reflection.entry_name),
                compilation_options: wgpu::PipelineCompilationOptions::default(),
                cache: cache.map(|a| &a.inner),
            });
        let bgl = pipeline.get_bind_group_layout(0);
        Ok(WgpuKernel {
            shader,
            pipeline,
            bgl,
        })
    }

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

    unsafe fn destroy_kernel_cache(
        &mut self,
        cache: <Wgpu as Backend>::KernelCache,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    unsafe fn get_kernel_cache_data(
        &mut self,
        cache: &mut <Wgpu as Backend>::KernelCache,
    ) -> Result<Vec<u8>, <Wgpu as Backend>::Error> {
        Ok(cache.inner.get_data().unwrap())
    }

    unsafe fn destroy_kernel(
        &mut self,
        kernel: <Wgpu as Backend>::Kernel,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    unsafe fn wait_for_idle(&mut self) -> Result<(), <Wgpu as Backend>::Error> {
        self.device.poll(wgpu::MaintainBase::Wait);
        Ok(())
    }

    unsafe fn create_recorder(
        &mut self,
    ) -> Result<<Wgpu as Backend>::CommandRecorder, <Wgpu as Backend>::Error> {
        let encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        Ok(WgpuCommandRecorder::Unrecorded(encoder))
    }

    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let idx = self.queue.submit(infos.iter_mut().map(|a| {
            if let WgpuCommandRecorder::Recorded(b) = a.command_recorder {
                b.clone()
            } else {
                // This stupid pointer nonsense, idk if its correct
                let recorded = match unsafe { std::ptr::read(a.command_recorder as *const _) } {
                    WgpuCommandRecorder::Unrecorded(r) => r.finish(),
                    WgpuCommandRecorder::Recorded(_) => unreachable!(),
                };
                unsafe {
                    std::ptr::write(
                        a.command_recorder,
                        WgpuCommandRecorder::Recorded(recorded.clone()),
                    )
                }
                recorded
            }
        }));
        for info in infos {
            if let Some(signal) = info.signal_semaphore {
                signal.inner.set(Some(idx.clone()));
            }
        }
        Ok(())
    }

    unsafe fn destroy_recorder(
        &mut self,
        recorder: <Wgpu as Backend>::CommandRecorder,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    unsafe fn clear_recorders(
        &mut self,
        buffers: &mut [&mut <Wgpu as Backend>::CommandRecorder],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        for b in buffers {
            let r = match b {
                WgpuCommandRecorder::Recorded(b) => unsafe {
                    std::ptr::read(b as *const _); // Call destructor
                    self.create_recorder()?
                },
                WgpuCommandRecorder::Unrecorded(r) => {
                    WgpuCommandRecorder::Unrecorded(unsafe { std::ptr::read(r as *const _) })
                }
            };
            unsafe { std::ptr::write(*b, r) };
        }
        Ok(())
    }

    unsafe fn create_buffer(
        &mut self,
        alloc_info: &BufferDescriptor,
    ) -> Result<<Wgpu as Backend>::Buffer, <Wgpu as Backend>::Error> {
        let buffer = self.device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: alloc_info.size,
            usage: {
                let mut usage = wgpu::BufferUsages::STORAGE;
                if alloc_info.indirect_capable {
                    usage |= wgpu::BufferUsages::INDIRECT;
                }
                if alloc_info.transfer_dst {
                    usage |= wgpu::BufferUsages::COPY_DST;
                }
                if alloc_info.transfer_src {
                    usage |= wgpu::BufferUsages::COPY_SRC;
                }
                if alloc_info.uniform {
                    usage |= wgpu::BufferUsages::UNIFORM;
                }
                if !matches!(alloc_info.memory_type, MemoryType::GpuOnly) {
                    usage |= wgpu::BufferUsages::MAP_WRITE;
                }
                usage
            },
            mapped_at_creation: alloc_info.mapped_at_creation,
        });
        Ok(WgpuBuffer {
            inner: buffer.clone(),
            mapped_ptr: Cell::new(if alloc_info.mapped_at_creation {
                Some(buffer.slice(..).get_mapped_range_mut().as_mut_ptr())
            } else {
                None
            }),
        })
    }

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

    unsafe fn map_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
    ) -> Result<*mut u8, <Wgpu as Backend>::Error> {
        let ptr = match buffer.mapped_ptr.get() {
            Some(ptr) => ptr,
            None => {
                let slice = buffer.inner.slice(..);
                slice.map_async(wgpu::MapMode::Write, |_| ());
                self.device.poll(wgpu::Maintain::Poll);
                slice.get_mapped_range_mut().as_mut_ptr()
            }
        };
        buffer.mapped_ptr.set(Some(ptr));
        Ok(ptr)
    }

    unsafe fn unmap_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        buffer.inner.unmap();
        buffer.mapped_ptr.set(None);
        Ok(())
    }

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
                        size,
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

    unsafe fn update_bind_group(
        &mut self,
        bg: &mut <Wgpu as Backend>::BindGroup,
        kernel: &mut <Wgpu as Backend>::Kernel,
        resources: &[GpuResource<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        *bg = unsafe { self.create_bind_group(kernel, resources)? };
        todo!()
    }

    unsafe fn destroy_bind_group(
        &mut self,
        kernel: &mut <Wgpu as Backend>::Kernel,
        bind_group: <Wgpu as Backend>::BindGroup,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    unsafe fn create_semaphore(
        &mut self,
    ) -> Result<<Wgpu as Backend>::Semaphore, <Wgpu as Backend>::Error> {
        Ok(WgpuSemaphore {
            inner: Cell::new(None),
        })
    }

    unsafe fn destroy_semaphore(
        &mut self,
        semaphore: <Wgpu as Backend>::Semaphore,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }

    unsafe fn create_event(
        &mut self,
    ) -> Result<<Wgpu as Backend>::Event, <Wgpu as Backend>::Error> {
        unreachable!()
    }

    unsafe fn destroy_event(
        &mut self,
        event: <Wgpu as Backend>::Event,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        unreachable!()
    }

    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Wgpu as Backend>::Error> {
        Ok(())
    }

    unsafe fn destroy(self) -> Result<(), <Wgpu as Backend>::Error> {
        // Destroyed on drop
        Ok(())
    }
}
pub struct WgpuKernel {
    shader: wgpu::ShaderModule,
    pipeline: wgpu::ComputePipeline,
    bgl: wgpu::BindGroupLayout,
}
impl Kernel<Wgpu> for WgpuKernel {}
pub struct WgpuBuffer {
    inner: wgpu::Buffer,
    mapped_ptr: Cell<Option<*mut u8>>,
}
impl Buffer<Wgpu> for WgpuBuffer {}
pub enum WgpuCommandRecorder {
    Unrecorded(wgpu::CommandEncoder),
    Recorded(wgpu::CommandBuffer),
}
impl CommandRecorder<Wgpu> for WgpuCommandRecorder {
    unsafe fn record_commands(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
        commands: &mut [BufferCommand<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        let r = match self {
            Self::Unrecorded(r) => r,
            Self::Recorded(_) => unreachable!(),
        };
        for command in commands {
            match command {
                BufferCommand::CopyBuffer {
                    src_buffer,
                    dst_buffer,
                    src_offset,
                    dst_offset,
                    size,
                } => {
                    r.copy_buffer_to_buffer(
                        &src_buffer.inner,
                        *src_offset,
                        &dst_buffer.inner,
                        *dst_offset,
                        *size,
                    );
                }
                BufferCommand::DispatchKernel {
                    shader,
                    bind_group,
                    push_constants,
                    workgroup_dims,
                } => {
                    let mut pass = r.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&shader.pipeline);
                    pass.set_bind_group(0, &bind_group.inner, &[]);
                    pass.dispatch_workgroups(
                        workgroup_dims[0],
                        workgroup_dims[1],
                        workgroup_dims[2],
                    );
                }
                BufferCommand::DispatchKernelIndirect {
                    shader,
                    bind_group,
                    push_constants,
                    indirect_buffer,
                    buffer_offset,
                    validate,
                } => {
                    if *validate {
                        unimplemented!("Indirect validation isn't implemented for wgpu");
                    }
                    let mut pass = r.begin_compute_pass(&wgpu::ComputePassDescriptor {
                        label: None,
                        timestamp_writes: None,
                    });
                    pass.set_pipeline(&shader.pipeline);
                    pass.set_bind_group(0, &bind_group.inner, &[]);
                    pass.dispatch_workgroups_indirect(&indirect_buffer.inner, *buffer_offset);
                }
                BufferCommand::MemoryBarrier { .. }
                | BufferCommand::PipelineBarrier { .. }
                | BufferCommand::SetEvent { .. }
                | BufferCommand::WaitEvent { .. } => (),
            }
        }

        Ok(())
    }
    unsafe fn record_dag(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
        resources: &[&GpuResource<Wgpu>],
        dag: &mut Dag<BufferCommand<Wgpu>>,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        unreachable!()
    }
}
pub struct WgpuBindGroup {
    inner: wgpu::BindGroup,
}
impl BindGroup<Wgpu> for WgpuBindGroup {}
pub struct WgpuKernelCache {
    inner: wgpu::PipelineCache,
}
impl KernelCache<Wgpu> for WgpuKernelCache {}
pub struct WgpuSemaphore {
    inner: Cell<Option<wgpu::SubmissionIndex>>,
}
impl Semaphore<Wgpu> for WgpuSemaphore {
    unsafe fn wait(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        if let Some(a) = self.inner.get_mut() {
            instance
                .device
                .poll(wgpu::Maintain::WaitForSubmissionIndex(a.clone()));
        }
        self.inner.set(None);
        Ok(())
    }
    unsafe fn is_signalled(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<bool, <Wgpu as Backend>::Error> {
        if self.inner.get_mut().is_some() {
            Ok(instance.device.poll(wgpu::Maintain::Poll).is_queue_empty())
        } else {
            Ok(true)
        }
    }
}
pub struct WgpuEvent;
impl Event<Wgpu> for WgpuEvent {}
#[derive(thiserror::Error, Debug)]
pub enum WgpuError {
    #[error("Wgpu internal error: {0}")]
    WgpuError(#[from] wgpu::Error),
    #[error("No suitable adapters found")]
    NoSuitableAdapters,
    #[error("Error requesting wgpu device: {0}")]
    RequestDevice(#[from] wgpu::RequestDeviceError),
}
impl Error<Wgpu> for WgpuError {
    fn is_out_of_device_memory(&self) -> bool {
        matches!(self, Self::WgpuError(wgpu::Error::OutOfMemory { .. }))
    }
    fn is_out_of_host_memory(&self) -> bool {
        false
    }
    fn is_timeout(&self) -> bool {
        false
    }
}
