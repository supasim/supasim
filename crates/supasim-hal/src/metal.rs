/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

#[link(name = "CoreGraphics", kind = "framework")]
unsafe extern "C" {}

use std::{ops::Deref, sync::Mutex};

use objc2::{
    rc::Retained,
    runtime::{NSObjectProtocol, ProtocolObject},
};
use objc2_foundation::{NSError, NSRange, NSString};
use objc2_metal::{
    MTLBlitCommandEncoder, MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue,
    MTLCompileOptions, MTLComputeCommandEncoder, MTLComputePipelineState, MTLDevice, MTLEvent,
    MTLFunction, MTLLibrary, MTLResource, MTLResourceOptions, MTLSharedEvent, MTLSize,
};
use thiserror::Error;
use types::{
    HalBufferType, HalInstanceProperties, KernelReflectionInfo, KernelTarget, MetalVersion,
};

use crate::{
    Backend, BackendInstance, BindGroup, Buffer, BufferCommand, CommandRecorder, Device, Kernel,
    KernelDescriptor, Semaphore, Stream,
};

struct UniqueObject<T: ?Sized + NSObjectProtocol>(Retained<ProtocolObject<T>>);
unsafe impl<T: ?Sized + NSObjectProtocol> Send for UniqueObject<T> {}
impl<T: ?Sized + NSObjectProtocol> UniqueObject<T> {
    pub fn new(inner: Retained<ProtocolObject<T>>) -> Self {
        Self(inner)
    }
    #[allow(dead_code)]
    pub fn into_inner(self) -> Retained<ProtocolObject<T>> {
        self.0
    }
    /// # Safety
    /// This should only be called if the cloned value will automatically be synced or won't be used.
    pub unsafe fn unsafe_clone(&self) -> Self {
        Self(self.0.clone())
    }
}
impl<T: ?Sized + NSObjectProtocol> Deref for UniqueObject<T> {
    type Target = ProtocolObject<T>;
    fn deref(&self) -> &ProtocolObject<T> {
        &self.0
    }
}
impl<T: ?Sized + NSObjectProtocol> std::fmt::Debug for UniqueObject<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

#[derive(Debug, Clone)]
pub struct Metal;
impl Metal {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    pub fn create_instance() -> Result<MetalInstance, MetalError> {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceCreate)?;
        Ok(MetalInstance {
            device: UniqueObject::new(device),
        })
    }
}

#[derive(Debug)]
pub struct MetalDevice {
    device: UniqueObject<dyn MTLDevice>,
}
impl Device<Metal> for MetalDevice {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn cleanup_cached_resources(&self, _instance: &MetalInstance) -> Result<(), MetalError> {
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_buffer(
        &self,
        alloc_info: &types::HalBufferDescriptor,
    ) -> Result<MetalBuffer, MetalError> {
        let options = match alloc_info.memory_type {
            HalBufferType::Storage => MTLResourceOptions::StorageModePrivate,
            HalBufferType::Upload => {
                MTLResourceOptions::StorageModeShared
                    | MTLResourceOptions::CPUCacheModeWriteCombined
            }
            HalBufferType::Download | HalBufferType::UploadDownload => {
                MTLResourceOptions::StorageModeShared
            }
        };
        let buffer = self
            .device
            .newBufferWithLength_options(alloc_info.size as usize, options)
            .ok_or(MetalError::BufferAllocate)?;
        Ok(MetalBuffer {
            buffer: UniqueObject::new(buffer),
        })
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    fn get_properties(&self, _instance: &MetalInstance) -> types::HalDeviceProperties {
        let is_unified_memory = self.device.hasUnifiedMemory();
        types::HalDeviceProperties {
            is_unified_memory,
            host_mappable_buffers: is_unified_memory,
        }
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _instance: &mut <Metal as Backend>::Instance,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct MetalStream {
    inner: UniqueObject<dyn MTLCommandQueue>,
}
impl Stream<Metal> for MetalStream {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_bind_group(
        &self,
        _device: &MetalDevice,
        _kernel: &<Metal as Backend>::Kernel,
        slices: &[crate::HalBufferSlice<Metal>],
    ) -> Result<<Metal as Backend>::BindGroup, <Metal as Backend>::Error> {
        let mut buffers = Vec::new();
        for slice in slices {
            buffers.push((
                unsafe { slice.buffer.buffer.unsafe_clone() },
                slice.offset,
                slice.len,
            ));
        }
        Ok(MetalBindGroup {
            buffers: Mutex::new(buffers),
        })
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_recorder(
        &self,
    ) -> Result<<Metal as Backend>::CommandRecorder, <Metal as Backend>::Error> {
        let command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>> =
            self.inner.commandBuffer().ok_or(MetalError::ObjectCreate)?;
        Ok(MetalCommandRecorder {
            command_buffer: UniqueObject::new(command_buffer),
        })
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [crate::RecorderSubmitInfo<Metal>],
    ) -> Result<(), <Metal as Backend>::Error> {
        for info in &*infos {
            for wait_semaphore in info.wait_semaphores {
                let as_event: &ProtocolObject<dyn MTLEvent> = wait_semaphore.event.0.as_ref();
                info.command_recorder
                    .command_buffer
                    .encodeWaitForEvent_value(as_event, wait_semaphore.current_value);
            }
            if let Some(signal_semaphore) = info.signal_semaphore {
                let as_event: &ProtocolObject<dyn MTLEvent> = signal_semaphore.event.0.as_ref();
                info.command_recorder
                    .command_buffer
                    .encodeSignalEvent_value(as_event, signal_semaphore.current_value);
            }
            info.command_recorder.command_buffer.commit();
        }
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn wait_for_idle(&mut self) -> Result<(), <Metal as Backend>::Error> {
        let cb = self.inner.commandBuffer().ok_or(MetalError::ObjectCreate)?;
        cb.commit();
        unsafe {
            cb.waitUntilCompleted();
        }
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn cleanup_cached_resources(
        &self,
        _instance: &<Metal as Backend>::Instance,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _device: &mut <Metal as Backend>::Device,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
}

impl Backend for Metal {
    type Instance = MetalInstance;
    type Device = MetalDevice;
    type Stream = MetalStream;

    type Semaphore = MetalSemaphore;
    type BindGroup = MetalBindGroup;
    type Kernel = MetalKernel;
    type Buffer = MetalBuffer;
    type CommandRecorder = MetalCommandRecorder;

    type Error = MetalError;

    #[cfg_attr(feature = "trace", tracing::instrument)]
    fn setup_default_descriptor() -> Result<crate::InstanceDescriptor<Self>, Self::Error> {
        let instance = Metal::create_instance()?;
        let device = MetalDevice {
            device: unsafe { instance.device.unsafe_clone() },
        };
        let stream = MetalStream {
            inner: UniqueObject::new(
                device
                    .device
                    .newCommandQueue()
                    .ok_or(MetalError::ObjectCreate)?,
            ),
        };
        Ok(crate::InstanceDescriptor {
            instance,
            devices: vec![crate::DeviceDescriptor {
                device,
                streams: vec![crate::StreamDescriptor {
                    stream,
                    stream_type: crate::StreamType::ComputeAndTransfer,
                }],
                group_idx: None,
            }],
        })
    }
}

#[derive(Debug)]
pub struct MetalInstance {
    // It is important that they aren't `Retain`'d because `Retain` requires
    // the inner type to be send + sync
    device: UniqueObject<dyn MTLDevice>,
}
impl BackendInstance<Metal> for MetalInstance {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    fn get_properties(&self) -> HalInstanceProperties {
        // TODO: check for supported metal version
        //let metal_version = MetalVersion::V2_3;
        //self.device.supportsFamily(MTLGPUFamily::Metal3);
        HalInstanceProperties {
            // Technically it's serial streams, but because there is so little
            // potential for overlap (at least on Apple silicon), I figured I would
            // mark it as automatic and then have the backend possibly split
            // transfers into separate streams behind the scenes
            sync_mode: types::SyncMode::Automatic,
            kernel_lang: KernelTarget::Msl {
                // Required for ulong in buffers apparently
                version: MetalVersion::V2_3,
            },
            easily_update_bind_groups: true,
            semaphore_signal: true,
            map_buffers: true,
            map_buffer_while_gpu_use: true,
            upload_download_buffers: true,
        }
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn create_semaphore(
        &self,
    ) -> Result<<Metal as Backend>::Semaphore, <Metal as Backend>::Error> {
        Ok(MetalSemaphore {
            event: UniqueObject::new(
                self.device
                    .newSharedEvent()
                    .ok_or(MetalError::ObjectCreate)?,
            ),
            current_value: 0,
        })
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn compile_kernel(
        &self,
        descriptor: KernelDescriptor,
    ) -> Result<<Metal as Backend>::Kernel, <Metal as Backend>::Error> {
        let str =
            std::str::from_utf8(descriptor.binary).map_err(|_| MetalError::NonUtf8KernelString)?;
        let library_options = MTLCompileOptions::new();
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(str), Some(&library_options))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(&descriptor.reflection.entry_point_name))
            .ok_or(MetalError::WrongMslFunctionName)?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)?;
        let mut revised_layout = vec![0; descriptor.reflection.buffers.len()];
        {
            let mut new_index = 0;
            for (old_index, &is_writable) in descriptor.reflection.buffers.iter().enumerate() {
                if is_writable {
                    revised_layout[old_index] = new_index;
                    new_index += 1;
                }
            }

            // Then, collect indices of readonly buffers
            for (old_index, &is_writable) in descriptor.reflection.buffers.iter().enumerate() {
                if !is_writable {
                    revised_layout[old_index] = new_index;
                    new_index += 1;
                }
            }
        }
        Ok(MetalKernel {
            _library: UniqueObject::new(library),
            _function: UniqueObject::new(function),
            pipeline: UniqueObject::new(pipeline),
            reflection: descriptor.reflection.clone(),
            revised_buffer_indices: revised_layout,
        })
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(self) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
}

#[must_use]
#[derive(Error, Debug)]
pub enum MetalError {
    #[error("SupaSim requires metal version ")]
    BadMetalVersion,
    #[error("Failed to create metal device or queue. No other information.")]
    DeviceCreate,
    #[error("Failed to create a metal object. This should be reported if it ever arises.")]
    ObjectCreate,
    #[error("Internal metal error: {0}")]
    MetalError(#[from] Retained<NSError>),
    #[error("A nonexistent entry point name was provided in kernel reflection")]
    WrongMslFunctionName,
    #[error("A buffer failed to allocate")]
    BufferAllocate,
    #[error("An invalid utf8 string was passed to compile_kernel")]
    NonUtf8KernelString,
}
impl crate::Error<Metal> for MetalError {
    fn is_out_of_device_memory(&self) -> bool {
        false
    }
    fn is_out_of_host_memory(&self) -> bool {
        false
    }
    fn is_timeout(&self) -> bool {
        false
    }
}

#[derive(Debug)]
pub struct MetalSemaphore {
    event: UniqueObject<dyn MTLSharedEvent>,
    current_value: u64,
}
impl Semaphore<Metal> for MetalSemaphore {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn is_signalled(
        &self,
        _device: &MetalInstance,
    ) -> Result<bool, <Metal as Backend>::Error> {
        let value = unsafe { self.event.signaledValue() };
        Ok(value == self.current_value + 1)
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn reset(&mut self, _device: &MetalInstance) -> Result<(), <Metal as Backend>::Error> {
        self.current_value += 1;
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn signal(&mut self, _device: &MetalInstance) -> Result<(), <Metal as Backend>::Error> {
        unsafe {
            self.event.setSignaledValue(self.current_value + 1);
        }
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn wait(&self, _device: &MetalInstance) -> Result<(), <Metal as Backend>::Error> {
        unsafe {
            self.event
                .waitUntilSignaledValue_timeoutMS(self.current_value + 1, u64::MAX);
        }
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(self, _device: &MetalInstance) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct MetalBindGroup {
    #[allow(clippy::type_complexity)]
    buffers: Mutex<Vec<(UniqueObject<dyn MTLBuffer>, u64, u64)>>,
}
impl BindGroup<Metal> for MetalBindGroup {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn update(
        &mut self,
        _device: &MetalDevice,
        _stream: &<Metal as Backend>::Stream,
        _kernel: &<Metal as Backend>::Kernel,
        buffers: &[crate::HalBufferSlice<Metal>],
    ) -> Result<(), <Metal as Backend>::Error> {
        let mut buffers_lock = self.buffers.lock().unwrap();
        for (i, slice) in buffers.iter().enumerate() {
            buffers_lock[i] = (
                unsafe { slice.buffer.buffer.unsafe_clone() },
                slice.offset,
                slice.len,
            );
        }
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _instance: &<Metal as Backend>::Stream,
        _kernel: &<Metal as Backend>::Kernel,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct MetalKernel {
    _library: UniqueObject<dyn MTLLibrary>,
    _function: UniqueObject<dyn MTLFunction>,
    pipeline: UniqueObject<dyn MTLComputePipelineState>,
    reflection: KernelReflectionInfo,
    /// For user passed buffer #i, `revised_buffer_indices[i]` is the index
    /// in the kernel's reordered layout
    revised_buffer_indices: Vec<usize>,
}
impl Kernel<Metal> for MetalKernel {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _instance: &<Metal as Backend>::Instance,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
}

#[derive(Debug)]
pub struct MetalBuffer {
    buffer: UniqueObject<dyn MTLBuffer>,
}
impl Buffer<Metal> for MetalBuffer {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn map(
        &mut self,
        _instance: &MetalDevice,
    ) -> Result<*mut u8, <Metal as Backend>::Error> {
        assert!((self.buffer.storageMode().0 & MTLResourceOptions::StorageModePrivate.0) == 0);
        Ok(self.buffer.contents().as_ptr() as *mut u8)
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn read(
        &mut self,
        _instance: &MetalDevice,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Metal as Backend>::Error> {
        assert!((self.buffer.storageMode().0 & MTLResourceOptions::StorageModePrivate.0) == 0);
        let ptr = self.buffer.contents().as_ptr() as *const u8;
        let slice = unsafe { std::slice::from_raw_parts(ptr.add(offset as usize), data.len()) };
        data.copy_from_slice(slice);
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn write(
        &mut self,
        _instance: &MetalDevice,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Metal as Backend>::Error> {
        assert!((self.buffer.storageMode().0 & MTLResourceOptions::StorageModePrivate.0) == 0);
        let ptr = self.buffer.contents().as_ptr() as *mut u8;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(offset as usize), data.len()) };
        slice.copy_from_slice(data);
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn unmap(
        &mut self,
        _instance: &<Metal as Backend>::Device,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _instance: &<Metal as Backend>::Device,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
}

#[derive(Debug)]
enum CurrentEncoder {
    None,
    Blit(Retained<ProtocolObject<dyn MTLBlitCommandEncoder>>),
    Compute(Retained<ProtocolObject<dyn MTLComputeCommandEncoder>>),
}
impl CurrentEncoder {
    fn blit(
        &mut self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<&ProtocolObject<dyn MTLBlitCommandEncoder>, MetalError> {
        Ok(match self {
            Self::Blit(b) => b,
            _ => {
                self.finish();
                *self =
                    CurrentEncoder::Blit(cb.blitCommandEncoder().ok_or(MetalError::ObjectCreate)?);
                let Self::Blit(blit) = self else {
                    unreachable!()
                };
                blit
            }
        })
    }
    fn compute(
        &mut self,
        cb: &ProtocolObject<dyn MTLCommandBuffer>,
    ) -> Result<&ProtocolObject<dyn MTLComputeCommandEncoder>, MetalError> {
        Ok(match self {
            Self::Compute(c) => c,
            _ => {
                self.finish();
                // We can configure it but that is unnecessary. None of the options are relevant.
                *self = CurrentEncoder::Compute(
                    cb.computeCommandEncoder().ok_or(MetalError::ObjectCreate)?,
                );
                let Self::Compute(compute) = self else {
                    unreachable!()
                };
                compute
            }
        })
    }
    fn finish(&mut self) {
        match self {
            Self::Blit(b) => b.endEncoding(),
            Self::Compute(c) => c.endEncoding(),
            Self::None => (),
        }
        *self = Self::None;
    }
}

#[derive(Debug)]
pub struct MetalCommandRecorder {
    command_buffer: UniqueObject<dyn MTLCommandBuffer>,
}
impl CommandRecorder<Metal> for MetalCommandRecorder {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn clear(&mut self, stream: &MetalStream) -> Result<(), <Metal as Backend>::Error> {
        *self = unsafe { stream.create_recorder()? };
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn record_commands(
        &mut self,
        _stream: &MetalStream,
        commands: &[crate::BufferCommand<Metal>],
    ) -> Result<(), <Metal as Backend>::Error> {
        let mut encoder = CurrentEncoder::None;
        for cmd in commands {
            match *cmd {
                BufferCommand::CopyBuffer {
                    src_buffer,
                    dst_buffer,
                    src_offset,
                    dst_offset,
                    len,
                } => {
                    let blit = encoder.blit(&self.command_buffer)?;
                    unsafe {
                        blit.copyFromBuffer_sourceOffset_toBuffer_destinationOffset_size(
                            &src_buffer.buffer,
                            src_offset as usize,
                            &dst_buffer.buffer,
                            dst_offset as usize,
                            len as usize,
                        );
                    }
                }
                BufferCommand::DispatchKernel {
                    kernel,
                    bind_group,
                    push_constants,
                    workgroup_dims,
                } => {
                    let comp = encoder.compute(&self.command_buffer)?;
                    comp.setComputePipelineState(&kernel.pipeline);
                    assert!(kernel.reflection.push_constants_size == push_constants.len() as u64);
                    let buffers_lock = bind_group.buffers.lock().unwrap();
                    for (i, item) in buffers_lock.iter().enumerate() {
                        unsafe {
                            comp.setBuffer_offset_atIndex(
                                Some(&item.0),
                                item.1 as usize,
                                kernel.revised_buffer_indices[i],
                            );
                        }
                    }
                    drop(buffers_lock);
                    let grid_size = MTLSize {
                        width: workgroup_dims[0] as usize,
                        height: workgroup_dims[1] as usize,
                        depth: workgroup_dims[2] as usize,
                    };
                    let group_size = MTLSize {
                        width: kernel.reflection.workgroup_size[0] as usize,
                        height: kernel.reflection.workgroup_size[1] as usize,
                        depth: kernel.reflection.workgroup_size[2] as usize,
                    };
                    comp.dispatchThreadgroups_threadsPerThreadgroup(grid_size, group_size);
                }
                BufferCommand::UpdateBindGroup {
                    bg,
                    kernel: _,
                    buffers,
                } => {
                    let mut buffers_lock = bg.buffers.lock().unwrap();
                    buffers_lock.clear();
                    for b in buffers {
                        buffers_lock.push((
                            unsafe { b.buffer.buffer.unsafe_clone() },
                            b.offset,
                            b.len,
                        ));
                    }
                }
                BufferCommand::ZeroMemory { ref buffer } => {
                    let blit = encoder.blit(&self.command_buffer)?;
                    blit.fillBuffer_range_value(
                        &buffer.buffer.buffer,
                        NSRange {
                            location: buffer.offset as usize,
                            length: buffer.len as usize,
                        },
                        0,
                    );
                }
                _ => (),
            }
        }
        encoder.finish();
        Ok(())
    }
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(
        self,
        _stream: &<Metal as Backend>::Stream,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
}
