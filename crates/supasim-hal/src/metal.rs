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
    Backend, BackendInstance, BindGroup, Buffer, BufferCommand, CommandRecorder, Kernel,
    KernelCache, Semaphore,
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
    pub fn create_instance() -> Result<MetalInstance, MetalError> {
        let device = objc2_metal::MTLCreateSystemDefaultDevice().ok_or(MetalError::DeviceCreate)?;
        let queue = device.newCommandQueue().ok_or(MetalError::DeviceCreate)?;
        Ok(MetalInstance {
            device: UniqueObject::new(device),
            command_queue: UniqueObject::new(queue),
        })
    }
}
impl Backend for Metal {
    type Instance = MetalInstance;
    type Error = MetalError;
    type Semaphore = MetalSemaphore;
    type BindGroup = MetalBindGroup;
    type Kernel = MetalKernel;
    type Buffer = MetalBuffer;
    type KernelCache = MetalKernelCache;
    type CommandRecorder = MetalCommandRecorder;
}

#[derive(Debug)]
pub struct MetalInstance {
    // It is important that they aren't `Retain`'d because `Retain` requires
    // the inner type to be send + sync
    device: UniqueObject<dyn MTLDevice>,
    command_queue: UniqueObject<dyn MTLCommandQueue>,
}
impl BackendInstance<Metal> for MetalInstance {
    fn get_properties(&mut self) -> HalInstanceProperties {
        // TODO: check for supported metal version
        //let metal_version = MetalVersion::V2_3;
        //self.device.supportsFamily(MTLGPUFamily::Metal3);
        HalInstanceProperties {
            // Technically it's serial streams, but because there is so little
            // potential for overlap (at least on Apple silicon), I figured I would
            // mark it as automatic and then have the backend possibly split
            // transfers into separate streams behind the scenes
            sync_mode: types::SyncMode::Automatic,
            pipeline_cache: false,
            kernel_lang: KernelTarget::Msl {
                // Required for ulong in buffers apparently
                version: MetalVersion::V2_3,
            },
            easily_update_bind_groups: true,
            semaphore_signal: true,
            map_buffers: true,
            is_unified_memory: self.device.hasUnifiedMemory(),
            map_buffer_while_gpu_use: true,
            upload_download_buffers: true,
        }
    }
    unsafe fn get_kernel_cache_data(
        &mut self,
        _cache: &mut <Metal as Backend>::KernelCache,
    ) -> Result<Vec<u8>, <Metal as Backend>::Error> {
        unreachable!()
    }
    unsafe fn create_semaphore(
        &mut self,
    ) -> Result<<Metal as Backend>::Semaphore, <Metal as Backend>::Error> {
        Ok(MetalSemaphore {
            event: UniqueObject::new(
                self.device
                    .newSharedEvent()
                    .ok_or(MetalError::ObjectCreate)?,
            ),
            current_value: Mutex::new(0),
        })
    }
    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::KernelReflectionInfo,
        _cache: Option<&mut <Metal as Backend>::KernelCache>,
    ) -> Result<<Metal as Backend>::Kernel, <Metal as Backend>::Error> {
        let str = std::str::from_utf8(binary).map_err(|_| MetalError::NonUtf8KernelString)?;
        let library_options = MTLCompileOptions::new();
        let library = self
            .device
            .newLibraryWithSource_options_error(&NSString::from_str(str), Some(&library_options))?;
        let function = library
            .newFunctionWithName(&NSString::from_str(&reflection.entry_point_name))
            .ok_or(MetalError::WrongMslFunctionName)?;
        let pipeline = self
            .device
            .newComputePipelineStateWithFunction_error(&function)?;
        let mut revised_layout = vec![0; reflection.buffers.len()];
        {
            let mut new_index = 0;
            for (old_index, &is_writable) in reflection.buffers.iter().enumerate() {
                if is_writable {
                    revised_layout[old_index] = new_index;
                    new_index += 1;
                }
            }

            // Then, collect indices of readonly buffers
            for (old_index, &is_writable) in reflection.buffers.iter().enumerate() {
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
            reflection: reflection.clone(),
            revised_buffer_indices: revised_layout,
        })
    }
    unsafe fn create_buffer(
        &mut self,
        alloc_info: &types::HalBufferDescriptor,
    ) -> Result<<Metal as Backend>::Buffer, <Metal as Backend>::Error> {
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
    unsafe fn create_recorder(
        &mut self,
    ) -> Result<<Metal as Backend>::CommandRecorder, <Metal as Backend>::Error> {
        let command_buffer: Retained<ProtocolObject<dyn MTLCommandBuffer>> = self
            .command_queue
            .commandBuffer()
            .ok_or(MetalError::ObjectCreate)?;
        Ok(MetalCommandRecorder {
            command_buffer: UniqueObject::new(command_buffer),
        })
    }
    unsafe fn create_bind_group(
        &mut self,
        _kernel: &mut <Metal as Backend>::Kernel,
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
    unsafe fn create_kernel_cache(
        &mut self,
        _initial_data: &[u8],
    ) -> Result<<Metal as Backend>::KernelCache, <Metal as Backend>::Error> {
        unreachable!("Metal doesn't support kernel caches")
    }
    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [crate::RecorderSubmitInfo<Metal>],
    ) -> Result<(), <Metal as Backend>::Error> {
        for info in &*infos {
            /*if let Some(wait_semaphore) = info.wait_semaphore {
                let as_event: &ProtocolObject<dyn MTLEvent> = wait_semaphore.event.0.as_ref();
                info.command_recorder
                    .command_buffer
                    .encodeWaitForEvent_value(
                        as_event,
                        *wait_semaphore.current_value.lock().unwrap(),
                    );
            }*/
            if let Some(signal_semaphore) = info.signal_semaphore {
                let as_event: &ProtocolObject<dyn MTLEvent> = signal_semaphore.event.0.as_ref();
                info.command_recorder
                    .command_buffer
                    .encodeSignalEvent_value(
                        as_event,
                        *signal_semaphore.current_value.lock().unwrap() + 1,
                    );
            }
            info.command_recorder.command_buffer.commit();
        }
        Ok(())
    }
    unsafe fn map_buffer(
        &mut self,
        buffer: &mut <Metal as Backend>::Buffer,
    ) -> Result<*mut u8, <Metal as Backend>::Error> {
        assert!((buffer.buffer.storageMode().0 & MTLResourceOptions::StorageModePrivate.0) == 0);
        Ok(buffer.buffer.contents().as_ptr() as *mut u8)
    }
    unsafe fn read_buffer(
        &mut self,
        buffer: &mut <Metal as Backend>::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Metal as Backend>::Error> {
        assert!((buffer.buffer.storageMode().0 & MTLResourceOptions::StorageModePrivate.0) == 0);
        let ptr = buffer.buffer.contents().as_ptr() as *const u8;
        let slice = unsafe { std::slice::from_raw_parts(ptr.add(offset as usize), data.len()) };
        data.copy_from_slice(slice);
        Ok(())
    }
    unsafe fn write_buffer(
        &mut self,
        buffer: &mut <Metal as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Metal as Backend>::Error> {
        assert!((buffer.buffer.storageMode().0 & MTLResourceOptions::StorageModePrivate.0) == 0);
        let ptr = buffer.buffer.contents().as_ptr() as *mut u8;
        let slice = unsafe { std::slice::from_raw_parts_mut(ptr.add(offset as usize), data.len()) };
        slice.copy_from_slice(data);
        Ok(())
    }
    unsafe fn unmap_buffer(
        &mut self,
        _buffer: &mut <Metal as Backend>::Buffer,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    unsafe fn update_bind_group(
        &mut self,
        bg: &mut <Metal as Backend>::BindGroup,
        _kernel: &mut <Metal as Backend>::Kernel,
        buffers: &[crate::HalBufferSlice<Metal>],
    ) -> Result<(), <Metal as Backend>::Error> {
        let mut buffers_lock = bg.buffers.lock().unwrap();
        for (i, slice) in buffers.iter().enumerate() {
            buffers_lock[i] = (
                unsafe { slice.buffer.buffer.unsafe_clone() },
                slice.offset,
                slice.len,
            );
        }
        Ok(())
    }
    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    unsafe fn wait_for_idle(&mut self) -> Result<(), <Metal as Backend>::Error> {
        // Apparently this gets mostly abstracted away and has little penalty. I prefer
        // it to having to unsafely clone a `UniqueObject`
        let cb = self
            .command_queue
            .commandBuffer()
            .ok_or(MetalError::ObjectCreate)?;
        cb.commit();
        unsafe {
            cb.waitUntilCompleted();
        }
        Ok(())
    }
    unsafe fn destroy(self) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    unsafe fn destroy_bind_group(
        &mut self,
        _kernel: &mut <Metal as Backend>::Kernel,
        _bind_group: <Metal as Backend>::BindGroup,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    unsafe fn destroy_buffer(
        &mut self,
        _buffer: <Metal as Backend>::Buffer,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    unsafe fn destroy_kernel(
        &mut self,
        _kernel: <Metal as Backend>::Kernel,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    unsafe fn destroy_kernel_cache(
        &mut self,
        _cache: <Metal as Backend>::KernelCache,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    unsafe fn destroy_recorder(
        &mut self,
        _recorder: <Metal as Backend>::CommandRecorder,
    ) -> Result<(), <Metal as Backend>::Error> {
        Ok(())
    }
    unsafe fn destroy_semaphore(
        &mut self,
        _semaphore: <Metal as Backend>::Semaphore,
    ) -> Result<(), <Metal as Backend>::Error> {
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

pub struct MetalSemaphore {
    event: UniqueObject<dyn MTLSharedEvent>,
    current_value: Mutex<u64>,
}
impl Semaphore<Metal> for MetalSemaphore {
    unsafe fn is_signalled(&self) -> Result<bool, <Metal as Backend>::Error> {
        let value = unsafe { self.event.signaledValue() };
        Ok(value == *self.current_value.lock().unwrap() + 1)
    }
    unsafe fn reset(&self) -> Result<(), <Metal as Backend>::Error> {
        *self.current_value.lock().unwrap() += 1;
        Ok(())
    }
    unsafe fn signal(&self) -> Result<(), <Metal as Backend>::Error> {
        unsafe {
            self.event
                .setSignaledValue(*self.current_value.lock().unwrap() + 1);
        }
        Ok(())
    }
    unsafe fn wait(&self) -> Result<(), <Metal as Backend>::Error> {
        let value = *self.current_value.lock().unwrap();
        unsafe {
            self.event
                .waitUntilSignaledValue_timeoutMS(value + 1, u64::MAX);
        }
        Ok(())
    }
}

pub struct MetalBindGroup {
    #[allow(clippy::type_complexity)]
    buffers: Mutex<Vec<(UniqueObject<dyn MTLBuffer>, u64, u64)>>,
}
impl BindGroup<Metal> for MetalBindGroup {}

pub struct MetalKernel {
    _library: UniqueObject<dyn MTLLibrary>,
    _function: UniqueObject<dyn MTLFunction>,
    pipeline: UniqueObject<dyn MTLComputePipelineState>,
    reflection: KernelReflectionInfo,
    /// For user passed buffer #i, `revised_buffer_indices[i]` is the index
    /// in the kernel's reordered layout
    revised_buffer_indices: Vec<usize>,
}
impl Kernel<Metal> for MetalKernel {}

pub struct MetalBuffer {
    buffer: UniqueObject<dyn MTLBuffer>,
}
impl Buffer<Metal> for MetalBuffer {}

pub struct MetalKernelCache {}
impl KernelCache<Metal> for MetalKernelCache {}

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

pub struct MetalCommandRecorder {
    command_buffer: UniqueObject<dyn MTLCommandBuffer>,
}
impl CommandRecorder<Metal> for MetalCommandRecorder {
    unsafe fn clear(
        &mut self,
        instance: &mut <Metal as Backend>::Instance,
    ) -> Result<(), <Metal as Backend>::Error> {
        *self = unsafe { instance.create_recorder()? };
        Ok(())
    }
    unsafe fn record_commands(
        &mut self,
        _instance: &mut <Metal as Backend>::Instance,
        commands: &mut [crate::BufferCommand<Metal>],
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
}
