/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! Issues this must handle:
//!
//! * Sharing references/multithreading
//! * Moving buffers in and out of GPU memory when OOM is hit
//! * Synchronization/creation and synchronization of command buffers
//! * Lazy operations
//! * Combine/optimize allocations and creation of things

pub extern crate kernels;

#[cfg(test)]
mod tests;

#[macro_use]
mod api_type;
mod buffer;
mod record;
mod sync;

use crate::buffer::residency::{BufferResidency, BufferResidencyRef};
use crate::buffer::{BufferAccess, BufferInner, BufferRange, BufferWeak};
use crate::sync::Semaphore;
use crate::sync::stream_thread::{StreamThreadHandle, StreamThreadMessage, create_sync_thread};
use anyhow::anyhow;
use hal::{
    BackendInstance as _, CommandRecorder as _, Device as _, Kernel as _, Semaphore as _,
    Stream as _, StreamDescriptor, StreamType,
};
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use smallvec::{SmallVec, smallvec};
use std::{
    ops::{Deref, DerefMut},
    sync::Arc,
};
use thiserror::Error;
use thunderdome::{Arena, Index};
use types::HalDeviceProperties;

pub use crate::buffer::{Buffer, BufferDescriptor, BufferSlice, access::MappedBuffer};
pub use hal;
pub use hal::{DeviceDescriptor, InstanceDescriptor};
pub use types::{
    Backend, ExternalBufferDescriptor, ExternalSemaphoreDescriptor, HalBufferType,
    KernelReflectionInfo, KernelTarget, MetalVersion, ShaderModel, SpirvVersion,
};

pub(crate) const DEVICE_SMALLVEC_SIZE: usize = 8;
pub(crate) const STREAM_SMALLVEC_SIZE: usize = 16;

struct InnerRef<'a, T>(RwLockReadGuard<'a, T>);
impl<T> Deref for InnerRef<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

struct InnerRefMut<'a, T>(RwLockWriteGuard<'a, T>);
impl<T> Deref for InnerRefMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        &self.0
    }
}
impl<T> DerefMut for InnerRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.0
    }
}

#[must_use]
#[derive(Error, Debug)]
pub enum SupaSimError<B: hal::Backend> {
    // Rust thinks that B::Error could be SupaSimError. Never mind that this would be a recursive definition
    HalError(B::Error),
    Other(anyhow::Error),
    AlreadyDestroyed(String),
    BufferRegionNotValid,
    ValidateIndirectUnsupported,
    UserClosure(anyhow::Error),
    KernelCompileError(#[from] kernels::KernelCompileError),
    SyncThreadPanic(String),
    BufferExportError(String),
    ZeroMemoryWrongAlignment,
}
impl<B: hal::Backend> std::fmt::Display for SupaSimError<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}

pub type SupaSimResult<B, T> = Result<T, SupaSimError<B>>;

trait MapSupasimError<T, B: hal::Backend> {
    fn map_supasim(self) -> Result<T, SupaSimError<B>>;
}
impl<T, B: hal::Backend> MapSupasimError<T, B> for Result<T, B::Error> {
    fn map_supasim(self) -> Result<T, SupaSimError<B>> {
        match self {
            Ok(t) => Ok(t),
            Err(e) => Err(SupaSimError::HalError(e)),
        }
    }
}

#[derive(Clone, Copy, Debug)]
pub struct InstanceProperties {
    pub kernel_lang: KernelTarget,
}

struct Stream<B: hal::Backend> {
    inner: Mutex<Option<B::Stream>>,
    /// Hal command recorders not currently in use
    unused_hal_command_recorders: Mutex<Vec<B::CommandRecorder>>,
    /// Is an option because this must be created after the instance and so
    /// starts uninitialized.
    stream_handle: Option<RwLock<StreamThreadHandle<B>>>,
}

struct Device<B: hal::Backend> {
    inner: Mutex<Option<B::Device>>,
    streams: SmallVec<[Stream<B>; STREAM_SMALLVEC_SIZE]>,
    _properties: HalDeviceProperties,
}

api_type!(Instance, {
    /// The inner hal instance
    hal_instance: RwLock<Option<B::Instance>>,
    /// The inner hal devices
    hal_devices: SmallVec<[Device<B>; DEVICE_SMALLVEC_SIZE]>,
    /// The hal instance properties
    hal_instance_properties: types::HalInstanceProperties,
    /// All created kernels
    kernels: RwLock<Arena<KernelWeak<B>>>,
    /// All created buffers
    buffers: RwLock<Arena<Option<BufferWeak<B>>>>,
    /// All wait handles created
    wait_handles: RwLock<Arena<WaitHandleWeak<B>>>,
    /// All created command recorders
    command_recorders: RwLock<Arena<Option<CommandRecorderWeak<B>>>>,
    unused_semaphores: Mutex<Vec<B::Semaphore>>,
    /// User accessible kernel compiler state
    kernel_compiler: Mutex<kernels::GlobalState>,
    /// A weak reference to self
    self_weak: Option<InstanceWeak<B>>,
    is_destroyed: bool,
},);

impl<B: hal::Backend> InstanceInner<B> {
    fn check_destroyed(&self) -> SupaSimResult<B, ()> {
        if self.is_destroyed {
            return Err(SupaSimError::AlreadyDestroyed("Instance".into()));
        }
        Ok(())
    }

    fn get_semaphore(&self) -> SupaSimResult<B, B::Semaphore> {
        Ok(if let Some(s) = self.unused_semaphores.lock().pop() {
            s
        } else {
            unsafe {
                self.hal_instance
                    .read()
                    .as_ref()
                    .unwrap()
                    .create_semaphore()
                    .map_supasim()?
            }
        })
    }
}

impl<B: hal::Backend> Instance<B> {
    pub fn from_hal(desc: hal::InstanceDescriptor<B>) -> Self {
        let instance = desc.instance;
        let device = desc.devices.into_iter().next().unwrap();
        let StreamDescriptor {
            stream,
            stream_type: StreamType::ComputeAndTransfer,
        } = device.streams.into_iter().next().unwrap()
        else {
            panic!("Expected compute and transfer queue to be first")
        };

        let device = device.device;
        let instance_properties = instance.get_properties();
        let device_properties = device.get_properties(&instance);
        let s = Self::from_inner(InstanceInner {
            _phantom: Default::default(),
            is_destroyed: false,
            hal_instance: RwLock::new(Some(instance)),
            hal_devices: smallvec![Device {
                inner: Mutex::new(Some(device)),
                streams: smallvec![Stream {
                    inner: Mutex::new(Some(stream)),
                    unused_hal_command_recorders: Mutex::new(Vec::new()),
                    stream_handle: None,
                }],
                _properties: device_properties,
            }],
            hal_instance_properties: instance_properties,
            kernels: RwLock::new(Arena::default()),
            buffers: RwLock::new(Arena::default()),
            wait_handles: RwLock::new(Arena::default()),
            command_recorders: RwLock::new(Arena::default()),
            unused_semaphores: Mutex::new(Vec::new()),
            kernel_compiler: Mutex::new(kernels::GlobalState::new_from_env().unwrap()),
            self_weak: None,
        });

        {
            let mut inner_mut = s.inner_mut().unwrap();
            inner_mut.self_weak = Some(s.downgrade());
            // Pass a WEAK instance handle: a strong clone here would form a
            // ref-cycle (thread -> Instance -> stream_handle -> thread) that
            // prevents `InstanceInner::drop` from ever running. See
            // `create_sync_thread`.
            inner_mut.hal_devices[0].streams[0].stream_handle =
                Some(RwLock::new(create_sync_thread(s.downgrade(), 0, 0)));
        }
        s
    }

    pub fn properties(&self) -> SupaSimResult<B, InstanceProperties> {
        let s = self.inner()?;
        s.check_destroyed()?;
        let v = s.hal_instance_properties;
        Ok(InstanceProperties {
            kernel_lang: v.kernel_lang,
        })
    }

    pub fn compile_raw_kernel(
        &self,
        binary: &[u8],
        reflection: types::KernelReflectionInfo,
    ) -> SupaSimResult<B, Kernel<B>> {
        let s = self.inner()?;
        s.check_destroyed()?;
        let kernel = unsafe {
            s.hal_instance
                .write()
                .as_mut()
                .unwrap()
                .compile_kernel(hal::KernelDescriptor {
                    binary,
                    reflection: reflection.clone(),
                })
        }
        .map_supasim()?;
        let k = Kernel::from_inner(KernelInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner: Some(kernel),
            id: Index::DANGLING,
            reflection_info: reflection,
        });
        k.inner_mut()?.id = s.kernels.write().insert(k.downgrade());
        Ok(k)
    }
    pub fn compile_slang_kernel(&self, slang: &str, entry: &str) -> SupaSimResult<B, Kernel<B>> {
        let s = self.inner()?;
        s.check_destroyed()?;
        let mut binary = Vec::new();
        let reflection_info =
            s.kernel_compiler
                .lock()
                .compile_kernel(kernels::KernelCompileOptions {
                    target: s.hal_instance_properties.kernel_lang,
                    source: kernels::KernelSource::Memory(slang.as_bytes()),
                    dest: kernels::KernelDest::Memory(&mut binary),
                    entry,
                    include: None,
                    fp_mode: kernels::KernelFpMode::Precise,
                    opt_level: kernels::OptimizationLevel::Standard,
                    stability: kernels::StabilityGuarantee::Stable,
                    minify: true,
                })?;
        drop(s);
        self.compile_raw_kernel(&binary, reflection_info)
    }

    pub fn create_recorder(&self) -> SupaSimResult<B, CommandRecorder<B>> {
        let s = self.inner()?;
        s.check_destroyed()?;
        let r = CommandRecorder::from_inner(CommandRecorderInner {
            _phantom: Default::default(),
            instance: self.clone(),
            // thunderdome::Index is POD (two u32s, no destructor); constructing with
            // Index::DANGLING then immediately overwriting via the arena insert below is sound.
            id: Index::DANGLING,
            commands: Vec::new(),
            is_alive: true,
            writes_slice: Vec::new(),
            sem_waits: vec![],
            sem_signals: vec![],
        });
        r.inner_mut()?.id = s.command_recorders.write().insert(Some(r.downgrade()));
        Ok(r)
    }

    pub fn create_buffer(&self, desc: &BufferDescriptor) -> SupaSimResult<B, Buffer<B>> {
        let s = self.inner()?;
        s.check_destroyed()?;
        let b = Buffer::from_inner(BufferInner {
            _phantom: Default::default(),
            instance: self.clone(),
            // thunderdome::Index is POD (two u32s, no destructor); constructing with
            // Index::DANGLING then immediately overwriting via the arena insert below is sound.
            id: Index::DANGLING,
            create_info: *desc,
            residency: BufferResidencyRef(RwLock::new(BufferResidency::new(
                1,
                desc.size,
                desc.contents_align as u32,
            ))),
            is_alive: true,
        });
        b.inner_mut()?.id = s.buffers.write().insert(Some(b.downgrade()));
        Ok(b)
    }

    pub fn submit_commands(
        &self,
        recorders: &[crate::CommandRecorder<B>],
    ) -> SupaSimResult<B, WaitHandle<B>> {
        sync::submit_command_recorders(self, recorders)
    }

    pub fn wait_for_idle(&self, _timeout: f32) -> SupaSimResult<B, ()> {
        // Take a *read* lock, not `inner_mut()`. The per-stream sync thread also
        // needs `instance.inner()` (a read lock) to make progress and to advance
        // the completion counter; a write lock here would starve it and deadlock,
        // while multiple concurrent readers are fine. The blocking wait itself
        // only touches the handle's own condvar, not the instance lock.
        let inner = self.inner()?;
        for stream in inner.hal_devices.iter().flat_map(|a| &a.streams) {
            let handle = stream.stream_handle.as_ref().unwrap().read();
            handle.wait_for_submission(handle.current_submitted_count - 1);
        }
        Ok(())
    }

    /// Do work which is queued but not yet started for batching reasons. Currently a NOOP
    pub fn do_busywork(&self) -> SupaSimResult<B, ()> {
        // Not implemented
        Ok(())
    }

    /// Clear cached resources. This would be useful if the usage requirements have just
    /// dramatically changed, for example after startup or between different simulations.
    /// Currently a NOOP.
    pub fn clear_cached_resources(&self) -> SupaSimResult<B, ()> {
        // Not fully implemented

        // This is about exclusive access not mutable access
        let mut _s = self.inner_mut()?;
        let s = &mut *_s;
        s.check_destroyed()?;
        let mut _hal_instance = s.hal_instance.write();
        let hal_instance = _hal_instance.as_mut().unwrap();
        unsafe {
            hal_instance.cleanup_cached_resources().map_supasim()?;
            for d in &mut s.hal_devices {
                d.inner
                    .get_mut()
                    .as_ref()
                    .unwrap()
                    .cleanup_cached_resources(hal_instance)
                    .map_supasim()?;
            }
        }
        Ok(())
    }

    /// # Safety
    /// Importing memory is inherently unsafe. You must manually use external semaphores and memory ownership transfers to synchronize.
    /// The descriptor must be valid.
    pub unsafe fn import_buffer(
        &self,
        _desc: &ExternalBufferDescriptor,
    ) -> SupaSimResult<B, Buffer<B>> {
        todo!()
    }

    /// # Safety
    /// Importing semaphores is inherently unsafe. The descriptor must be valid.
    pub unsafe fn import_semaphore(
        &self,
        _desc: &ExternalSemaphoreDescriptor,
    ) -> SupaSimResult<B, ExternalSemaphore<B>> {
        todo!()
    }

    pub fn destroy(&mut self) -> SupaSimResult<B, ()> {
        let mut s = self.inner_mut()?;
        s.check_destroyed()?;
        s.is_destroyed = true;
        Ok(())
    }
}

impl<B: hal::Backend> InstanceInner<B> {
    fn destroy(&mut self) {
        // Tell all streams to shutdown
        for stream in self.hal_devices.iter().flat_map(|a| &a.streams) {
            // This will only not occur if this an error occurs during creation
            if let Some(sync_handle) = stream.stream_handle.as_ref() {
                sync_handle
                    .write()
                    .sender
                    .send(StreamThreadMessage::ShutDown)
                    .unwrap()
            }
        }
        // Wait for all streams to shutdown
        for stream in self.hal_devices.iter_mut().flat_map(|a| &mut a.streams) {
            if let Some(handle) = stream.stream_handle.take() {
                handle.into_inner().thread.join().unwrap();
            }
        }
        for (_, cr) in std::mem::take(&mut *self.command_recorders.write()) {
            if let Some(cr) = cr
                && let Ok(cr) = cr.upgrade()
                && let Ok(mut cr2) = cr.inner_mut()
            {
                cr2.destroy(self);
            }
        }
        for (_, wh) in std::mem::take(&mut *self.wait_handles.write()) {
            if let Ok(wh) = wh.upgrade()
                && let Ok(mut wh2) = wh.inner_mut()
            {
                wh2.destroy(self);
            }
        }
        for (_, b) in std::mem::take(&mut *self.buffers.write()) {
            if let Some(b) = b
                && let Ok(b) = b.upgrade()
                && let Ok(mut b2) = b.inner_mut()
            {
                b2.destroy(self);
            }
        }
        for (_, k) in std::mem::take(&mut *self.kernels.write()) {
            if let Ok(k) = k.upgrade()
                && let Ok(mut k2) = k.inner_mut()
            {
                k2.destroy(self);
            }
        }
        for thing in std::mem::take(&mut *self.unused_semaphores.lock()) {
            unsafe {
                thing
                    .destroy(self.hal_instance.read().as_ref().unwrap())
                    .unwrap();
            }
        }
        unsafe {
            let mut instance = self.hal_instance.get_mut().take().unwrap();
            for mut device in std::mem::take(&mut self.hal_devices) {
                let mut dev = device.inner.get_mut().take().unwrap();
                for mut stream in device.streams {
                    let st = stream.inner.get_mut().take().unwrap();
                    for cr in std::mem::take(&mut *stream.unused_hal_command_recorders.lock()) {
                        cr.destroy(&st).unwrap();
                    }
                    st.destroy(&mut dev).unwrap();
                }
                dev.destroy(&mut instance).unwrap();
            }
            instance.destroy().unwrap();
        }
    }
}

impl<B: hal::Backend> Drop for InstanceInner<B> {
    fn drop(&mut self) {
        self.destroy();
    }
}

api_type!(Kernel, {
    instance: Instance<B>,
    inner: Option<B::Kernel>,
    reflection_info: KernelReflectionInfo,
    id: Index,
},);

impl<B: hal::Backend> std::fmt::Debug for Kernel<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Kernel(undecorated)")
    }
}

impl<B: hal::Backend> KernelInner<B> {
    pub fn destroy(&mut self, instance: &InstanceInner<B>) {
        instance.kernels.write().remove(self.id).unwrap();
        if let Some(inner) = std::mem::take(&mut self.inner) {
            unsafe {
                MapSupasimError::<(), B>::map_supasim(
                    inner.destroy(instance.hal_instance.read().as_ref().unwrap()),
                )
                .unwrap();
            }
        }
    }
}

impl<B: hal::Backend> Drop for KernelInner<B> {
    fn drop(&mut self) {
        if self.inner.is_some()
            && let Ok(instance) = self.instance.clone().inner()
        {
            self.destroy(&instance);
        }
    }
}

#[derive(Debug)]
struct BufferCommand<B: hal::Backend> {
    inner: BufferCommandInner<B>,
    buffers: Vec<BufferSlice<B>>,
}

#[derive(Debug)]
enum BufferCommandInner<B: hal::Backend> {
    CopyBufferToBuffer,
    ZeroBuffer,
    CopyFromTemp {
        src_offset: u64,
    },
    KernelDispatch {
        kernel: Kernel<B>,
        workgroup_dims: [u32; 3],
    },
    MemoryTransfer {
        import: bool,
    },
    CommandRecorderEnd,
}

struct _SubmittedCommandRecorder<B: hal::Backend> {
    command_recorders: Vec<B::CommandRecorder>,
    used_semaphore: B::Semaphore,
    /// Buffers that are waiting for this to be destroyed
    buffers_to_destroy: Vec<B::Buffer>,
    kernels_to_destroy: Vec<Index>,
    bind_groups: Vec<(B::BindGroup, Index)>,
    // used_buffer_ranges: Vec<(BufferUserId, BufferWeak<B>)>,
    used_buffers: Vec<BufferWeak<B>>,
}

api_type!(CommandRecorder, {
    instance: Instance<B>,
    is_alive: bool,
    id: Index,
    commands: Vec<BufferCommand<B>>,
    writes_slice: Vec<u8>,
    sem_waits: Vec<ExternalSemaphore<B>>,
    sem_signals: Vec<ExternalSemaphore<B>>,
},);

impl<B: hal::Backend> CommandRecorder<B> {
    fn check_destroyed(&self) -> SupaSimResult<B, ()> {
        if !self.inner()?.is_alive {
            return Err(SupaSimError::AlreadyDestroyed("CommandRecorder".into()));
        }
        Ok(())
    }

    /// Valid copies (automatic can replace any of these)
    /// * Upload -> download
    /// * Upload -> gpu
    /// * Gpu -> download
    /// * Gpu -> gpu
    ///
    /// So, basically anything that isn't starting from download or landing in upload
    pub fn copy_buffer(
        &self,
        src_buffer: &Buffer<B>,
        dst_buffer: &Buffer<B>,
        src_offset: u64,
        dst_offset: u64,
        length: u64,
    ) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        let src_slice = BufferSlice {
            buffer: src_buffer.clone(),
            access: BufferAccess {
                needs_mut: false,
                range: BufferRange {
                    start: src_offset,
                    length,
                },
            },
        };
        let dst_slice = BufferSlice {
            buffer: dst_buffer.clone(),
            access: BufferAccess {
                needs_mut: true,
                range: BufferRange {
                    start: dst_offset,
                    length,
                },
            },
        };
        src_slice.validate()?;
        dst_slice.validate()?;
        {
            let mut lock = self.inner_mut()?;
            lock.commands.push(BufferCommand {
                inner: BufferCommandInner::CopyBufferToBuffer,
                buffers: vec![src_slice, dst_slice],
            });
        }
        Ok(())
    }

    /// Offset and length must be multiples of 4
    pub fn zero_memory(&self, buffer: &Buffer<B>, offset: u64, size: u64) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        if !offset.is_multiple_of(4) || !size.is_multiple_of(4) {
            return Err(SupaSimError::ZeroMemoryWrongAlignment);
        }
        let slice = BufferSlice {
            buffer: buffer.clone(),
            access: BufferAccess {
                range: BufferRange {
                    start: offset,
                    length: size,
                },
                needs_mut: true,
            },
        };
        slice.validate()?;
        {
            let mut lock = self.inner_mut()?;
            lock.commands.push(BufferCommand {
                inner: BufferCommandInner::ZeroBuffer,
                buffers: vec![slice],
            });
        }
        Ok(())
    }

    pub fn write_buffer<T: bytemuck::Pod>(
        &self,
        buffer: &Buffer<B>,
        offset: u64,
        data: &[T],
    ) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        let data = bytemuck::cast_slice(data);
        let length = data.len() as u64;
        let dst_slice = BufferSlice {
            buffer: buffer.clone(),
            access: BufferAccess {
                range: BufferRange {
                    start: offset,
                    length,
                },
                needs_mut: true,
            },
        };
        dst_slice.validate()?;
        let mut s = self.inner_mut()?;
        let src_offset = s.writes_slice.len() as u64;
        s.commands.push(BufferCommand {
            inner: BufferCommandInner::CopyFromTemp { src_offset },
            buffers: vec![dst_slice],
        });
        s.writes_slice.extend_from_slice(data);
        Ok(())
    }

    pub fn dispatch_kernel(
        &self,
        kernel: &Kernel<B>,
        buffers: &[&BufferSlice<B>],
        workgroup_dims: [u32; 3],
    ) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        let reflection_info = kernel.inner()?.reflection_info.clone();
        if buffers.len() != reflection_info.buffers.len() {
            return Err(SupaSimError::Other(anyhow!(
                "Incorrect number of buffers passed to dispatch_kernel"
            )));
        }
        for (i, &b) in reflection_info.buffers.iter().enumerate() {
            if buffers[i].access.needs_mut != b {
                return Err(SupaSimError::Other(anyhow!(
                    "Buffer at index {i} in dispatch_kernel does not have the correct mutability"
                )));
            }
        }
        for b in buffers {
            b.validate()?;
        }
        let mut s = self.inner_mut()?;
        s.commands.push(BufferCommand {
            inner: BufferCommandInner::KernelDispatch {
                kernel: kernel.clone(),
                workgroup_dims,
            },
            buffers: buffers.iter().map(|&b| b.clone()).collect(),
        });
        Ok(())
    }

    pub fn transfer_memory(&self, buffer: &BufferSlice<B>, import: bool) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        buffer.validate()?;
        self.inner_mut()?.commands.push(BufferCommand {
            inner: BufferCommandInner::MemoryTransfer { import },
            buffers: vec![buffer.clone()],
        });
        Ok(())
    }

    /// Applies to the **entire** command recorder. All commands execute after the external semaphore has been signalled.
    pub fn wait_for_semaphore(&self, s: ExternalSemaphore<B>) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        self.inner_mut()?.sem_waits.push(s);
        Ok(())
    }

    /// Applies to the **entire** command recorder. After all commands execute after the external semaphore will be signalled.
    pub fn signal_semaphore(&self, s: ExternalSemaphore<B>) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        self.inner_mut()?.sem_signals.push(s);
        Ok(())
    }
}

impl<B: hal::Backend> CommandRecorderInner<B> {
    fn destroy(&mut self, instance: &InstanceInner<B>) {
        instance.command_recorders.write().remove(self.id);
        self.is_alive = false;
    }
}

impl<B: hal::Backend> Drop for CommandRecorderInner<B> {
    fn drop(&mut self) {
        if self.is_alive
            && let Ok(instance) = self.instance.clone().inner()
        {
            self.destroy(&instance);
        }
    }
}

api_type!(WaitHandle, {
    instance: Instance<B>,
    semaphore: Arc<Semaphore<B>>,
    id: Index,
    is_alive: bool,
},);

impl<B: hal::Backend> WaitHandleInner<B> {
    fn destroy(&mut self, instance: &InstanceInner<B>) {
        instance.wait_handles.write().remove(self.id);
        self.is_alive = false;
    }
}

impl<B: hal::Backend> Drop for WaitHandleInner<B> {
    fn drop(&mut self) {
        if self.is_alive
            && let Ok(instance) = self.instance.clone().inner()
        {
            self.destroy(&instance);
        }
    }
}

impl<B: hal::Backend> WaitHandle<B> {
    pub fn wait(&self) -> SupaSimResult<B, ()> {
        self.inner()?.semaphore.wait()
    }
    pub fn is_complete(&self) -> SupaSimResult<B, bool> {
        self.inner()?.semaphore.is_signalled()
    }
}

api_type!(ExternalSemaphore, {
    _instance: Instance<B>,
    _id: Index,
},);
