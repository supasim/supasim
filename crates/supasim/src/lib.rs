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

#![allow(dead_code)]

pub extern crate kernels;

#[cfg(test)]
mod tests;

mod record;
mod residency;
mod sync;

use anyhow::anyhow;
use hal::{
    BackendInstance as _, Buffer as _, CommandRecorder as _, Device as _, Kernel as _, Stream as _,
    StreamDescriptor, StreamType,
};
use parking_lot::{Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use smallvec::{SmallVec, smallvec};
use std::ops::Bound;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};
use thiserror::Error;
use thunderdome::{Arena, Index};
use types::HalDeviceProperties;

pub use hal;
pub use hal::{DeviceDescriptor, InstanceDescriptor};
pub use types::{
    Backend, ExternalBufferDescriptor, ExternalSemaphoreDescriptor, HalBufferType,
    KernelReflectionInfo, KernelTarget, MetalVersion, ShaderModel, SpirvVersion,
};

use crate::residency::BufferResidency;
use crate::sync::Semaphore;

pub(crate) const DEVICE_SMALLVEC_SIZE: usize = 4;
pub(crate) const STREAM_SMALLVEC_SIZE: usize = 16;

pub type UserBufferAccessClosure<'a, B> =
    Box<dyn FnOnce(&mut [MappedBuffer<'a, B>]) -> anyhow::Result<()>>;
pub type SendableUserBufferAccessClosure<B> =
    Box<dyn Send + FnOnce(&mut [MappedBuffer<'_, B>]) -> anyhow::Result<()>>;

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
#[derive(Clone, Copy, Debug)]
pub struct BufferDescriptor {
    /// The size needed in bytes
    pub size: u64,
    /// The value that the contents of the buffer must be aligned to. This is important for when supasim must detect
    pub contents_align: u64,
    /// Currently unused. In the future this may be used to prefer keeping some buffers in memory when device runs out of memory and swapping becomes necessary
    pub priority: f32,
    /// If `Some`, the device given will be preferred for operations using this buffer. This is useful for example when exporting memory.
    pub preferred_device_index: Option<usize>,
}
impl Default for BufferDescriptor {
    fn default() -> Self {
        Self {
            size: 0,
            contents_align: 0,
            priority: 1.0,
            preferred_device_index: None,
        }
    }
}

#[derive(Clone, Debug)]
pub struct BufferSlice<B: hal::Backend> {
    pub buffer: Buffer<B>,
    pub start: u64,
    pub len: u64,
    pub needs_mut: bool,
}
impl<B: hal::Backend> BufferSlice<B> {
    pub fn validate(&self) -> SupaSimResult<B, ()> {
        let b = self.buffer.inner()?;
        if self.start.is_multiple_of(b.create_info.contents_align)
            && self.len.is_multiple_of(b.create_info.contents_align)
        {
            Ok(())
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }
    pub fn validate_with_align(&self, align: u64) -> SupaSimResult<B, ()> {
        let b = self.buffer.inner()?;
        // This is explained in MappedBuffer associated methods
        if align.is_multiple_of(b.create_info.contents_align)
            && self.start.is_multiple_of(align)
            && self.len.is_multiple_of(align)
        {
            Ok(())
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }
    pub fn entire_buffer(buffer: &Buffer<B>, needs_mut: bool) -> SupaSimResult<B, Self> {
        Ok(Self {
            buffer: buffer.clone(),
            start: 0,
            len: buffer.inner()?.create_info.size,
            needs_mut,
        })
    }
    fn range(&self) -> BufferRange {
        BufferRange {
            start: self.start,
            len: self.len,
            needs_mut: self.needs_mut,
        }
    }
}

trait AsId<T> {}
/// The size must be >0 or equality comparison is undefined
macro_rules! api_type {
    ($name: ident, { $($field:tt)* }, $($attr: meta),*) => {
        paste::paste! {
            // Inner type
            pub(crate) struct [<$name Inner>] <B: hal::Backend> {
                _phantom: PhantomData<B>, // Ensures B is always used
                _is_destroyed: bool,
                $($field)*
            }

            #[derive(Clone)]
            pub(crate) struct [<$name Weak>] <B: hal::Backend>(std::sync::Weak<RwLock<[<$name Inner>]<B>>>);
            #[allow(dead_code)]
            impl<B: hal::Backend> [<$name Weak>] <B> {
                pub(crate) fn upgrade(&self) -> SupaSimResult<B, $name<B>> {
                    Ok($name(self.0.upgrade().ok_or(SupaSimError::AlreadyDestroyed(stringify!($name).to_owned()))?))
                }
            }

            // Outer type, with some helper methods
            #[derive(Clone)]
            $(
                #[$attr]
            )*
            pub struct $name <B: hal::Backend> (std::sync::Arc<RwLock<[<$name Inner>]<B>>>);
            #[allow(dead_code)]
            impl<B: hal::Backend> $name <B> {
                pub(crate) fn from_inner(inner: [<$name Inner>]<B>) -> Self {
                    Self(Arc::new(RwLock::new(inner)))
                }
                pub(crate) fn inner(&'_ self) -> SupaSimResult<B, InnerRef<'_, [<$name Inner>]<B>>> {
                    let r = self.0.read();
                    Ok(InnerRef(r))
                }
                pub(crate) fn inner_mut(&'_ self) -> SupaSimResult<B, InnerRefMut<'_, [<$name Inner>]<B>>> {
                    let r = self.0.write();
                    Ok(InnerRefMut(r))
                }
                pub(crate) fn downgrade(&self) -> [<$name Weak>]<B> {
                    [<$name Weak>](Arc::downgrade(&self.0))
                }
                pub(crate) fn check_destroyed(&self) -> SupaSimResult<B, ()> {
                    if self.0.read()._is_destroyed {
                        Err(SupaSimError::AlreadyDestroyed(stringify!($name).to_owned()))
                    } else {
                        Ok(())
                    }
                }
                pub(crate) fn _destroy(&self) -> SupaSimResult<B, ()> {
                    let mut r = self.0.write();
                    if r._is_destroyed {
                        return Err(SupaSimError::AlreadyDestroyed(stringify!($name).to_owned()));
                    }
                    // Destroy shouldn't get called, thats only on drop
                    r._is_destroyed = true;
                    Ok(())
                }
            }
            impl<B: hal::Backend> PartialEq for $name <B> {
                fn eq(&self, other: &Self) -> bool {
                    std::ptr::eq(self.0.as_ref(), other.0.as_ref())
                }
            }
            impl<B: hal::Backend> Eq for $name <B> {}
            unsafe impl<B: hal::Backend> Send for $name <B> {}
            unsafe impl<B: hal::Backend> Sync for $name <B> {}
            impl<B: hal::Backend> AsId<$name <B>> for [<$name Inner>] <B> {}
            impl<B: hal::Backend> AsId<[<$name Inner>] <B>> for $name <B> {}
            impl<B: hal::Backend> AsId<$name <B>> for [<$name Weak>] <B> {}
            impl<B: hal::Backend> AsId<[<$name Weak>] <B>> for $name <B> {}
            impl<B: hal::Backend> AsId<$name <B>> for Option<[<$name Weak>] <B>> {}
            impl<B: hal::Backend> AsId<Option<[<$name Weak>] <B>>> for $name <B> {}
        }
    };
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
impl<B: hal::Backend> std::fmt::Display for SupaSimError<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{self:?}")
    }
}
pub type SupaSimResult<B, T> = Result<T, SupaSimError<B>>;

#[derive(Clone, Copy, Debug)]
pub struct InstanceProperties {
    pub kernel_lang: KernelTarget,
}
struct Stream<B: hal::Backend> {
    inner: Mutex<Option<B::Stream>>,
}
struct Device<B: hal::Backend> {
    inner: Mutex<Option<B::Device>>,
    streams: SmallVec<[Stream<B>; STREAM_SMALLVEC_SIZE]>,
    properties: HalDeviceProperties,
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
    /// Hal command recorders not currently in use
    hal_command_recorders: RwLock<Vec<B::CommandRecorder>>,
    /// User accessible kernel compiler state
    kernel_compiler: Mutex<kernels::GlobalState>,
    /// A weak reference to self
    self_weak: Option<InstanceWeak<B>>,
},);
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
            _is_destroyed: false,
            hal_instance: RwLock::new(Some(instance)),
            hal_devices: smallvec![Device {
                inner: Mutex::new(Some(device)),
                streams: smallvec![Stream {
                    inner: Mutex::new(Some(stream))
                }],
            }],
            hal_instance_properties: instance_properties,
            hal_device_properties: device_properties,
            kernels: RwLock::new(Arena::default()),
            buffers: RwLock::new(Arena::default()),
            wait_handles: RwLock::new(Arena::default()),
            command_recorders: RwLock::new(Arena::default()),
            hal_command_recorders: RwLock::new(Vec::new()),
            kernel_compiler: Mutex::new(kernels::GlobalState::new_from_env().unwrap()),
            self_weak: None,
        });

        {
            let mut inner_mut = s.inner_mut().unwrap();
            inner_mut.self_weak = Some(s.downgrade());
        }
        s
    }
    pub fn properties(&self) -> SupaSimResult<B, InstanceProperties> {
        self.check_destroyed()?;
        let v = self.inner()?.hal_instance_properties;
        Ok(InstanceProperties {
            kernel_lang: v.kernel_lang,
        })
    }
    pub fn compile_raw_kernel(
        &self,
        binary: &[u8],
        reflection: types::KernelReflectionInfo,
    ) -> SupaSimResult<B, Kernel<B>> {
        self.check_destroyed()?;

        let s = self.inner()?;
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
            _is_destroyed: false,
            instance: self.clone(),
            per_device: smallvec![Some(kernel)],
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
            id: Index::DANGLING,
            reflection_info: reflection,
            last_used_per_device: smallvec![Vec::new()],
        });
        k.inner_mut()?.id = s.kernels.write().insert(k.downgrade());
        Ok(k)
    }
    pub fn compile_slang_kernel(&self, slang: &str, entry: &str) -> SupaSimResult<B, Kernel<B>> {
        self.check_destroyed()?;
        let mut binary = Vec::new();
        let s = self.inner()?;
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
        self.check_destroyed()?;
        let s = self.inner()?;
        let r = CommandRecorder::from_inner(CommandRecorderInner {
            _phantom: Default::default(),
            _is_destroyed: false,
            instance: self.clone(),
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
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
        self.check_destroyed()?;
        let s = self.inner()?;
        let b = Buffer::from_inner(BufferInner {
            _phantom: Default::default(),
            _is_destroyed: false,
            instance: self.clone(),
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
            id: Index::DANGLING,
            create_info: *desc,
            is_currently_external: false,
            residency: BufferResidency::new(1),
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
        self.check_destroyed()?;
        let s = self.inner()?;
        s.sync_thread().wait_for_idle()?;
        Ok(())
    }
    /// Do work which is queued but not yet started for batching reasons. Currently a NOOP
    pub fn do_busywork(&self) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        Ok(())
    }
    /// Clear cached resources. This would be useful if the usage requirements have just
    /// dramatically changed, for example after startup or between different simulations.
    /// Currently a NOOP.
    pub fn clear_cached_resources(&self) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        let s = self.inner()?;
        unsafe {
            s.hal_instance
                .write()
                .as_mut()
                .unwrap()
                .cleanup_cached_resources()
        }
        .map_supasim()?;
        Ok(())
    }

    /// If the closure panics, memory issues may occur.
    /// Also, calling any supasim related functions in
    /// the closure may cause deadlocks.
    #[allow(clippy::type_complexity)]
    pub fn access_buffers(
        &self,
        closure: UserBufferAccessClosure<B>,
        buffers: &[&BufferSlice<B>],
    ) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        let properties = self.inner()?.hal_instance_properties;
        let mut mapped_buffers = Vec::with_capacity(buffers.len());
        let mut ids = Vec::new();
        for b in buffers {
            b.validate()?;
            ids.push(b.acquire(BufferUser::Cpu, false)?.id);
        }
        if properties.map_buffers {
            let instance = self.inner()?;
            #[allow(clippy::never_loop)]
            for (i, b) in buffers.iter().enumerate() {
                let mut buffer_inner = b.buffer.inner_mut()?;
                let mapping = unsafe {
                    buffer_inner
                        .inner
                        .as_mut()
                        .unwrap()
                        .map(instance.device.lock().as_mut().unwrap())
                        .map_supasim()?
                };
                mapped_buffers.push(MappedBuffer {
                    instance: self.clone(),
                    inner: unsafe { mapping.add(b.start as usize) },
                    len: b.len,
                    buffer_align: buffer_inner.create_info.contents_align,
                    has_mut: b.needs_mut,
                    was_used_mut: false,
                    in_buffer_offset: b.start,
                    buffer: b.buffer.clone(),
                    vec_capacity: None,
                    user_id: ids[i],
                    _p: Default::default(),
                });
            }
            drop(instance);
            // Memory issues if we don't unmap I guess
            let error = closure(&mut mapped_buffers).map_err(|e| SupaSimError::UserClosure(e));
            drop(mapped_buffers);
            error?;
        } else {
            let mut buffer_contents = Vec::new();
            for b in buffers {
                let mut buffer = b.buffer.inner_mut()?;
                let instance = buffer.instance.clone();
                let mut data;
                #[allow(clippy::uninit_vec)]
                {
                    data = Vec::with_capacity(b.len as usize);
                    unsafe {
                        data.set_len(b.len as usize);
                        buffer
                            .inner
                            .as_mut()
                            .unwrap()
                            .read(
                                instance.inner()?.device.lock().as_ref().unwrap(),
                                b.start,
                                &mut data,
                            )
                            .map_supasim()?;
                    };
                };
                buffer_contents.push(data);
            }
            for (i, a) in buffer_contents.iter_mut().enumerate() {
                let b = &buffers[i];
                let buffer_inner = b.buffer.inner()?;
                let mapped = MappedBuffer {
                    instance: self.clone(),
                    inner: a.as_mut_ptr(),
                    len: b.len,
                    buffer_align: buffer_inner.create_info.contents_align,
                    has_mut: b.needs_mut,
                    was_used_mut: false,
                    in_buffer_offset: b.start,
                    buffer: b.buffer.clone(),
                    vec_capacity: Some(a.capacity()),
                    user_id: ids[i],
                    _p: Default::default(),
                };
                mapped_buffers.push(mapped);
            }
            // Memory issues if we don't unmap I guess
            let error = closure(&mut mapped_buffers).map_err(|e| SupaSimError::UserClosure(e));
            for b in buffer_contents {
                std::mem::forget(b);
            }
            drop(mapped_buffers);
            error?;
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
        self._destroy()
    }
}
impl<B: hal::Backend> InstanceInner<B> {
    fn destroy(&mut self) {
        println!("Instance destroying");
        // We unsafely pass a mutable reference to the sync thread, knowing that this thread will "join" it,
        // meaning that at no point is the mutable reference used by both at the same time.
        let mut sync_thread = self.sync_thread.take().unwrap();
        let _ = sync_thread.wait_for_idle();
        let _ = sync_thread
            .sender
            .get_mut()
            .send(sync::SendSyncThreadEvent::WaitFinishAndShutdown);
        let _ = sync_thread.thread.join();
        if sync_thread.shared_thread.0.lock().error.is_some() {
            println!("Instance attempting to shutdown gracefully despite sync thread error");
        }
        for (_, cr) in std::mem::take(&mut *self.command_recorders.write()) {
            if let Some(cr) = cr
                && let Ok(cr) = cr.upgrade()
            {
                if let Ok(mut cr2) = cr.inner_mut() {
                    cr2.destroy(self);
                }
                let _ = cr._destroy();
            }
        }
        for (_, wh) in std::mem::take(&mut *self.wait_handles.write()) {
            if let Ok(wh) = wh.upgrade() {
                if let Ok(mut wh2) = wh.inner_mut() {
                    wh2.destroy(self);
                }
                let _ = wh._destroy();
            }
        }
        for (_, b) in std::mem::take(&mut *self.buffers.write()) {
            if let Some(b) = b
                && let Ok(b) = b.upgrade()
            {
                if let Ok(mut b2) = b.inner_mut() {
                    b2.destroy(self);
                }
                let _ = b._destroy();
            }
        }
        for (_, k) in std::mem::take(&mut *self.kernels.write()) {
            if let Ok(k) = k.upgrade() {
                if let Ok(mut k2) = k.inner_mut() {
                    k2.destroy(self);
                }
                let _ = k._destroy();
            }
        }
        unsafe {
            let mut instance = self.hal_instance.get_mut().take().unwrap();
            for mut device in std::mem::take(&mut self.hal_devices) {
                let mut dev = device.inner.get_mut().take().unwrap();
                for mut stream in device.streams {
                    stream.inner.get_mut().take().unwrap().destroy(&mut dev);
                }
                dev.destroy(&mut instance);
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
    per_device: SmallVec<[Option<B::Kernel>; DEVICE_SMALLVEC_SIZE]>,
    reflection_info: KernelReflectionInfo,
    last_used_per_device: SmallVec<[Vec<Semaphore<B>>; DEVICE_SMALLVEC_SIZE]>,
    id: Index,
},);
impl<B: hal::Backend> std::fmt::Debug for Kernel<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Kernel(undecorated)")
    }
}
impl<B: hal::Backend> KernelInner<B> {
    pub fn destroy(&mut self, instance: &InstanceState<B>) {
        let inner = std::mem::take(&mut self.inner).unwrap();
        instance.kernels.lock().remove(self.id).unwrap();
        unsafe {
            inner
                .destroy(instance.instance.lock().as_ref().unwrap())
                .unwrap();
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
    Dummy,
}
struct SubmittedCommandRecorder<B: hal::Backend> {
    command_recorders: Vec<B::CommandRecorder>,
    used_semaphore: B::Semaphore,
    /// Buffers that are waiting for this to be destroyed
    buffers_to_destroy: Vec<B::Buffer>,
    kernels_to_destroy: Vec<Index>,
    bind_groups: Vec<(B::BindGroup, Index)>,
    used_buffer_ranges: Vec<(BufferUserId, BufferWeak<B>)>,
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
        len: u64,
    ) -> SupaSimResult<B, ()> {
        self.check_destroyed()?;
        let src_slice = BufferSlice {
            buffer: src_buffer.clone(),
            start: src_offset,
            len,
            needs_mut: false,
        };
        let dst_slice = BufferSlice {
            buffer: dst_buffer.clone(),
            start: dst_offset,
            len,
            needs_mut: true,
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
            start: offset,
            len: size,
            needs_mut: true,
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
        let len = data.len() as u64;
        let dst_slice = BufferSlice {
            buffer: buffer.clone(),
            start: offset,
            len,
            needs_mut: true,
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
            if buffers[i].needs_mut != b {
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
    fn destroy(&mut self, instance: &InstanceState<B>) {
        instance.command_recorders.lock().remove(self.id);
        self.is_alive = true;
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

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
struct BufferRange {
    start: u64,
    len: u64,
    needs_mut: bool,
}
impl<B: hal::Backend> From<&BufferSlice<B>> for BufferRange {
    fn from(s: &BufferSlice<B>) -> Self {
        Self {
            start: s.start,
            len: s.len,
            needs_mut: s.needs_mut,
        }
    }
}
impl BufferRange {
    pub fn overlaps(&self, other: &Self) -> bool {
        // Starts before the end of the other
        // and the other starts before the end of this
        // and at least one of them is mutable
        self.start < other.start + other.len
            && other.start < self.start + self.len
            && (self.needs_mut || other.needs_mut)
    }
    pub fn overlaps_ignore_mut(&self, other: &Self) -> bool {
        self.start < other.start + other.len && other.start < self.start + self.len
    }
    pub fn contains(&self, other: &Self) -> bool {
        self.start <= other.start
            && self.start + self.len >= other.start + other.len
            && (!other.needs_mut || self.needs_mut)
    }
    pub fn try_join(&self, other: &Self) -> Option<Self> {
        if self.overlaps_ignore_mut(other) {
            if self.needs_mut == other.needs_mut {
                let start = self.start.min(other.start);
                let end = (self.start + self.len).max(other.start + other.len);
                let len = end - start;
                Some(Self {
                    needs_mut: self.needs_mut,
                    start,
                    len,
                })
            } else if self.contains(other) {
                Some(*self)
            } else if other.contains(self) {
                Some(*other)
            } else {
                None
            }
        } else {
            None
        }
    }
    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if self.overlaps(other) {
            let start = self.start.min(other.start);
            let end = (self.start + self.len).max(other.start + other.len);
            Some(Self {
                start,
                len: end - start,
                needs_mut: true,
            })
        } else {
            None
        }
    }
}
enum BufferBacking<B: hal::Backend> {
    HostBacked(B::Buffer),
    GpuBacked(B::Buffer),
    NoBacking,
}
impl<B: hal::Backend> BufferBacking<B> {
    pub fn buffer_mut(&mut self) -> Option<&mut B::Buffer> {
        match self {
            Self::HostBacked(b) => Some(b),
            Self::GpuBacked(b) => Some(b),
            Self::NoBacking => None,
        }
    }
}
api_type!(Buffer, {
    instance: Instance<B>,
    id: Index,
    create_info: BufferDescriptor,
    residency: BufferResidency<B>,
    is_currently_external: bool,
},);
impl<B: hal::Backend> Buffer<B> {
    pub fn write<T: bytemuck::Pod>(&self, offset: u64, data: &[T]) -> SupaSimResult<B, ()> {
        let buffer_slice = BufferSlice {
            buffer: self.clone(),
            start: offset,
            len: data.len() as u64,
            needs_mut: true,
        };
        buffer_slice.validate_with_align(size_of::<T>() as u64)?;
        let id = buffer_slice.acquire(BufferUser::Cpu, false)?.id;
        let mut s = self.inner_mut()?;
        let _instance = s.instance.clone();
        let instance = _instance.inner()?;
        let slice = bytemuck::cast_slice::<T, u8>(data);
        unsafe {
            s.inner
                .as_mut()
                .unwrap()
                .write(instance.device.lock().as_ref().unwrap(), offset, slice)
                .map_supasim()?;
        }
        drop(s);
        buffer_slice.release(id)?;
        Ok(())
    }
    pub fn read<T: bytemuck::Pod>(&self, offset: u64, out: &mut [T]) -> SupaSimResult<B, ()> {
        let slice = BufferSlice {
            buffer: self.clone(),
            start: offset,
            len: out.len() as u64,
            needs_mut: false,
        };
        slice.validate_with_align(size_of::<T>() as u64)?;
        let id = slice.acquire(BufferUser::Cpu, false)?;
        let mut s = self.inner_mut()?;
        let _instance = s.instance.clone();
        let instance = _instance.inner()?;
        let slice = bytemuck::cast_slice_mut::<T, u8>(out);
        unsafe {
            s.inner
                .as_mut()
                .unwrap()
                .read(instance.device.lock().as_ref().unwrap(), offset, slice)
                .map_supasim()?;
        }
        drop(s);
        self.inner()?.slice_tracker.release(id);
        Ok(())
    }
    pub fn access(
        &'_ self,
        offset: u64,
        len: u64,
        needs_mut: bool,
    ) -> SupaSimResult<B, MappedBuffer<'_, B>> {
        let slice = BufferSlice {
            buffer: self.clone(),
            start: offset,
            len,
            needs_mut,
        };
        slice.validate()?;
        let id = slice.acquire(BufferUser::Cpu, false)?.id;
        let _instance = self.inner()?.instance.clone();
        let instance = _instance.inner()?;
        let mut s = self.inner_mut()?;
        let buffer_align = s.create_info.contents_align;
        if instance.hal_instance_properties.map_buffers {
            let mapped_ptr = unsafe {
                s.inner
                    .as_mut()
                    .unwrap()
                    .map(instance.device.lock().as_ref().unwrap())
                    .map_supasim()?
                    .add(offset as usize)
            };
            drop(instance);
            Ok(MappedBuffer {
                instance: _instance,
                inner: mapped_ptr,
                len,
                has_mut: needs_mut,
                buffer: slice.buffer,
                buffer_align,
                in_buffer_offset: offset,
                was_used_mut: false,
                vec_capacity: None,
                user_id: id,
                _p: Default::default(),
            })
        } else {
            let mut data = Vec::with_capacity(len as usize);
            unsafe {
                s.inner
                    .as_mut()
                    .unwrap()
                    .read(instance.device.lock().as_ref().unwrap(), offset, &mut data)
                    .map_supasim()?;
            }
            drop(instance);
            let out = Ok(MappedBuffer {
                instance: _instance,
                inner: data.as_mut_ptr(),
                len,
                buffer_align,
                in_buffer_offset: offset,
                has_mut: needs_mut,
                was_used_mut: false,
                buffer: slice.buffer,
                vec_capacity: Some(data.capacity()),
                user_id: id,
                _p: Default::default(),
            });
            std::mem::forget(data);
            out
        }
    }
    pub fn slice(&self, range: impl std::ops::RangeBounds<u64>, needs_mut: bool) -> BufferSlice<B> {
        let start = match range.start_bound() {
            Bound::Included(v) => *v,
            Bound::Excluded(v) => *v + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(v) => *v + 1,
            Bound::Excluded(v) => *v,
            Bound::Unbounded => self.inner().unwrap().create_info.size,
        };
        BufferSlice {
            buffer: self.clone(),
            start,
            len: end - start,
            needs_mut,
        }
    }
}
impl<B: hal::Backend> std::fmt::Debug for Buffer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Buffer(undecorated)")
    }
}
impl<B: hal::Backend> BufferInner<B> {
    fn destroy(&mut self, instance: &InstanceInner<B>) {
        instance.buffers.write().remove(self.id);
        unsafe {
            std::mem::take(&mut self.inner)
                .unwrap()
                .destroy(instance.device.lock().as_ref().unwrap())
                .unwrap();
        }
    }
}
impl<B: hal::Backend> Drop for BufferInner<B> {
    fn drop(&mut self) {
        if self.inner.is_some()
            && let Ok(instance) = self.instance.clone().inner()
        {
            self.destroy(&instance);
        }
    }
}

/// If the entire buffer isn't the same type you are trying to read, read as bytes first then cast yourself.
/// SupaSim does checks for alignments and validates offsets with the size of types
pub struct MappedBuffer<'a, B: hal::Backend> {
    instance: Instance<B>,
    inner: *mut u8,
    len: u64,
    buffer_align: u64,
    in_buffer_offset: u64,
    has_mut: bool,
    was_used_mut: bool,
    buffer: Buffer<B>,
    user_id: u64,
    /// Size of the vector in which the data is allocated for non memory mapping scenarios
    vec_capacity: Option<usize>,
    _p: PhantomData<&'a ()>,
}
impl<B: hal::Backend> MappedBuffer<'_, B> {
    pub fn readable<T: bytemuck::Pod>(&self) -> SupaSimResult<B, &[T]> {
        let s = unsafe { std::slice::from_raw_parts(self.inner, self.len as usize) };
        // Length of the slice is a multiple of the length of the type
        if self.len.is_multiple_of(size_of::<T>() as u64)
        // Length of the type is a multiple of the buffer alignment
        && ((size_of::<T>() as u64).is_multiple_of(self.buffer_align) || self.buffer_align.is_multiple_of(size_of::<T>() as u64))
            // The offset is reasonable given the size of T
            && self.in_buffer_offset.is_multiple_of(size_of::<T>() as u64)
        {
            Ok(bytemuck::cast_slice(s))
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }
    pub fn writeable<T: bytemuck::Pod>(&mut self) -> SupaSimResult<B, &mut [T]> {
        if !self.has_mut {
            return Err(SupaSimError::BufferRegionNotValid);
        }
        self.was_used_mut = true;
        let s = unsafe { std::slice::from_raw_parts_mut(self.inner, self.len as usize) };
        // Length of the slice is a multiple of the length of the type
        if self.len.is_multiple_of(size_of::<T>() as u64)
        // Length of the type is a multiple of the buffer alignment
        && ((size_of::<T>() as u64).is_multiple_of(self.buffer_align) || self.buffer_align.is_multiple_of(size_of::<T>() as u64))
            // The offset is reasonable given the size of T
            && self.in_buffer_offset.is_multiple_of(size_of::<T>() as u64)
        {
            Ok(bytemuck::cast_slice_mut(s))
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }
}
impl<B: hal::Backend> Drop for MappedBuffer<'_, B> {
    fn drop(&mut self) {
        let instance = self.instance.inner().unwrap();
        let slice = BufferSlice {
            buffer: self.buffer.clone(),
            start: self.in_buffer_offset,
            len: self.len,
            needs_mut: self.has_mut,
        };
        slice.release(self.user_id).unwrap();
        let mut s = self.buffer.inner_mut().unwrap();
        if instance.hal_instance_properties.map_buffers {
            // Nothing needs to be done for now
        } else {
            unsafe {
                let vec =
                    Vec::from_raw_parts(self.inner, self.len as usize, self.vec_capacity.unwrap());
                if self.was_used_mut {
                    s.inner
                        .as_mut()
                        .unwrap()
                        .write(
                            instance.device.lock().as_ref().unwrap(),
                            self.in_buffer_offset,
                            &vec,
                        )
                        .unwrap();
                }
                // And then the vec gets dropped. Tada!
            }
        }
    }
}

api_type!(WaitHandle, {
    instance: Instance<B>,
    semaphore: Semaphore<B>,
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
        let s = self.inner()?;
        let _i = s.instance.clone();
        let instance = _i.inner()?;
        instance.sync_thread().wait_for(s.index, true)?;
        Ok(())
    }
    pub fn is_complete(&self) -> SupaSimResult<B, bool> {
        let s = self.inner()?;
        let _i = s.instance.clone();
        let instance = _i.inner()?;
        instance.sync_thread().wait_for(s.index, false)
    }
}
#[allow(clippy::type_complexity)]
pub type CpuCallback<B> = (
    Box<dyn Fn(Vec<MappedBuffer<B>>) -> Result<(), SupaSimError<B>>>,
    Vec<Buffer<B>>,
);

api_type!(ExternalSemaphore, {
    instance: Instance<B>,
    id: Index,
},);
