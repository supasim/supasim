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

mod sync;

use anyhow::anyhow;
use hal::{BackendInstance as _, CommandRecorder as _};
use parking_lot::{Condvar, Mutex, RwLock, RwLockReadGuard, RwLockWriteGuard};
use std::cell::UnsafeCell;
use std::collections::{HashMap, HashSet};
use std::ops::Bound;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::Arc,
};
use thiserror::Error;
use thunderdome::{Arena, Index};

pub use bytemuck;
pub use hal;
pub use types::{
    Backend, HalBufferType, KernelReflectionInfo, KernelTarget, MetalVersion, ShaderModel,
    SpirvVersion,
};

use crate::sync::{GpuSubmissionInfo, SyncThreadHandle, create_sync_thread};

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
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferType {
    /// Used by kernels
    Gpu,
    /// Used to upload data to GPU
    Upload,
    /// Used to download data from GPU
    Download,
    /// Automatically converted between CPU/GPU depending on usage. Currently not implemented
    Automatic,
}
impl BufferType {
    pub fn can_be_on_cpu(&self) -> bool {
        match self {
            Self::Gpu => false,
            Self::Upload | Self::Download | Self::Automatic => true,
        }
    }
    pub fn can_be_on_gpu(&self) -> bool {
        match self {
            Self::Gpu | Self::Automatic => true,
            Self::Upload | Self::Download => false,
        }
    }
}
#[derive(Clone, Copy, Debug)]
pub struct BufferDescriptor {
    /// The size needed in bytes
    pub size: u64,
    /// The type of the buffer
    pub buffer_type: BufferType,
    /// The value that the contents of the buffer must be aligned to. This is important for when supasim must detect
    pub contents_align: u64,
    /// Currently unused. In the future this may be used to prefer keeping some buffers in memory when device runs out of memory and swapping becomes necessary
    pub priority: f32,
    /// Whether the memory can be exported
    pub can_export: bool,
}
impl Default for BufferDescriptor {
    fn default() -> Self {
        Self {
            size: 0,
            buffer_type: BufferType::Gpu,
            contents_align: 0,
            priority: 1.0,
            can_export: false,
        }
    }
}
impl From<BufferDescriptor> for types::HalBufferDescriptor {
    fn from(s: BufferDescriptor) -> types::HalBufferDescriptor {
        types::HalBufferDescriptor {
            size: s.size,
            memory_type: match s.buffer_type {
                BufferType::Gpu => types::HalBufferType::Storage,
                BufferType::Upload => types::HalBufferType::Upload,
                BufferType::Download => types::HalBufferType::Download,
                BufferType::Automatic => types::HalBufferType::Storage,
            },
            min_alignment: 16,
            can_export: false,
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
        if (self.start % b.create_info.contents_align) == 0
            && (self.len % b.create_info.contents_align) == 0
        {
            Ok(())
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }
    pub fn validate_with_align(&self, align: u64) -> SupaSimResult<B, ()> {
        let b = self.buffer.inner()?;
        // This is explained in MappedBuffer associated methods
        if (align % b.create_info.contents_align) == 0
            && (self.start % align) == 0
            && (self.len % align) == 0
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
    fn acquire(&self, user: BufferUser, bypass_gpu: bool) -> SupaSimResult<B, BufferUserId> {
        let s = self.buffer.inner()?;
        let instance = s.instance.clone();

        s.slice_tracker.acquire(
            &*instance.inner()?,
            BufferRange {
                start: self.start,
                len: self.len,
                needs_mut: self.needs_mut,
            },
            user,
            bypass_gpu,
        )
    }
    fn release(&self, id: u64) -> SupaSimResult<B, ()> {
        let s = self.buffer.inner()?;
        // Clippy generates a confusing warning and nonsense solution
        s.slice_tracker.release(BufferUserId {
            range: self.into(),
            id,
        });
        Ok(())
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
                pub(crate) fn inner(&self) -> SupaSimResult<B, InnerRef<[<$name Inner>]<B>>> {
                    let r = self.0.read();
                    Ok(InnerRef(r))
                }
                pub(crate) fn inner_mut(&self) -> SupaSimResult<B, InnerRefMut<[<$name Inner>]<B>>> {
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
    /// Buffer attempted to be used from wrong side of cpu/gpu on non-unified memory system
    BufferLocalityViolated,
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
    pub supports_pipeline_cache: bool,
    pub kernel_lang: KernelTarget,
    pub is_unified_memory: bool,
    pub export_buffers: bool,
}
struct InstanceState<B: hal::Backend> {
    /// The inner hal instance
    inner: Mutex<Option<B::Instance>>,
    /// The hal instance properties
    inner_properties: types::HalInstanceProperties,
    /// All created kernels
    kernels: Mutex<Arena<KernelWeak<B>>>,
    /// All created kernel caches
    kernel_caches: Mutex<Arena<Option<KernelCacheWeak<B>>>>,
    /// All created buffers
    buffers: Mutex<Arena<Option<BufferWeak<B>>>>,
    /// All wait handles created
    wait_handles: Mutex<Arena<WaitHandleWeak<B>>>,
    /// All created command recorders
    command_recorders: Mutex<Arena<Option<CommandRecorderWeak<B>>>>,
    /// Hal command recorders not currently in use
    hal_command_recorders: Mutex<Vec<B::CommandRecorder>>,
    /// User accessible kernel compiler state
    kernel_compiler: Mutex<kernels::GlobalState>,
    /// A handle to the thread used for syncing
    sync_thread: UnsafeCell<Option<SyncThreadHandle<B>>>,
    /// A weak reference to self
    myself: UnsafeCell<Option<SupaSimInstanceWeak<B>>>,
}
impl<B: hal::Backend> InstanceState<B> {
    pub fn sync_thread(&self) -> &SyncThreadHandle<B> {
        unsafe { &*self.sync_thread.get() }.as_ref().unwrap()
    }
}
impl<B: hal::Backend> Drop for InstanceState<B> {
    fn drop(&mut self) {
        let instance = std::mem::take(&mut *self.inner.lock()).unwrap();
        unsafe {
            instance.destroy().unwrap();
        }
    }
}
api_type!(SupaSimInstance, {
    _inner: Arc<InstanceState<B>>,
},);
impl<B: hal::Backend> Deref for SupaSimInstanceInner<B> {
    type Target = InstanceState<B>;
    fn deref(&self) -> &Self::Target {
        &self._inner
    }
}
impl<B: hal::Backend> SupaSimInstance<B> {
    pub fn from_hal(mut hal: B::Instance) -> Self {
        let inner_properties = hal.get_properties();
        let s = Self::from_inner(SupaSimInstanceInner {
            _phantom: Default::default(),
            _is_destroyed: false,
            _inner: Arc::new(InstanceState {
                inner: Mutex::new(Some(hal)),
                inner_properties,
                kernels: Mutex::new(Arena::default()),
                kernel_caches: Mutex::new(Arena::default()),
                buffers: Mutex::new(Arena::default()),
                wait_handles: Mutex::new(Arena::default()),
                command_recorders: Mutex::new(Arena::default()),
                hal_command_recorders: Mutex::new(Vec::new()),
                kernel_compiler: Mutex::new(kernels::GlobalState::new_from_env().unwrap()),
                sync_thread: UnsafeCell::new(None),
                myself: UnsafeCell::new(None),
            }),
        });

        let handle = create_sync_thread(s.clone()).unwrap();

        unsafe {
            let inner_mut = s.inner().unwrap();
            *inner_mut.sync_thread.get() = Some(handle);
            *inner_mut.myself.get() = Some(s.downgrade());
        }

        s
    }
    pub fn properties(&self) -> SupaSimResult<B, InstanceProperties> {
        self.check_destroyed()?;
        let v = self.inner()?.inner_properties;
        Ok(InstanceProperties {
            supports_pipeline_cache: v.pipeline_cache,
            kernel_lang: v.kernel_lang,
            is_unified_memory: v.is_unified_memory,
            export_buffers: v.export_memory,
        })
    }
    pub fn compile_raw_kernel(
        &self,
        binary: &[u8],
        reflection_info: types::KernelReflectionInfo,
        cache: Option<&KernelCache<B>>,
    ) -> SupaSimResult<B, Kernel<B>> {
        self.check_destroyed()?;
        let s = self.inner()?;
        let mut cache_lock = if let Some(cache) = cache {
            Some(cache.inner_mut()?)
        } else {
            None
        };
        let kernel = unsafe {
            s.inner.lock().as_mut().unwrap().compile_kernel(
                binary,
                &reflection_info,
                if let Some(lock) = cache_lock.as_mut() {
                    Some(lock.inner.as_mut().unwrap())
                } else {
                    None
                },
            )
        }
        .map_supasim()?;
        let k = Kernel::from_inner(KernelInner {
            _phantom: Default::default(),
            _is_destroyed: false,
            instance: self.clone(),
            inner: Some(kernel),
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
            id: Index::DANGLING,
            reflection_info,
            last_used: 0,
        });
        k.inner_mut()?.id = s.kernels.lock().insert(k.downgrade());
        Ok(k)
    }
    pub fn compile_slang_kernel(
        &self,
        slang: &str,
        entry: &str,
        cache: Option<&KernelCache<B>>,
    ) -> SupaSimResult<B, Kernel<B>> {
        self.check_destroyed()?;
        let mut binary = Vec::new();
        let s = self.inner()?;
        let reflection_info =
            s.kernel_compiler
                .lock()
                .compile_kernel(kernels::KernelCompileOptions {
                    target: s.inner_properties.kernel_lang,
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
        self.compile_raw_kernel(&binary, reflection_info, cache)
    }
    pub fn create_kernel_cache(&self, data: &[u8]) -> SupaSimResult<B, KernelCache<B>> {
        self.check_destroyed()?;
        let s = self.inner()?;
        let inner =
            unsafe { s.inner.lock().as_mut().unwrap().create_kernel_cache(data) }.map_supasim()?;
        let k = KernelCache::from_inner(KernelCacheInner {
            _phantom: Default::default(),
            _is_destroyed: false,
            instance: self.clone(),
            inner: Some(inner),
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
            id: Index::DANGLING,
        });
        k.inner_mut()?.id = s.kernel_caches.lock().insert(Some(k.downgrade()));
        Ok(k)
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
        });
        r.inner_mut()?.id = s.command_recorders.lock().insert(Some(r.downgrade()));
        Ok(r)
    }
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> SupaSimResult<B, Buffer<B>> {
        self.check_destroyed()?;
        if desc.can_export && desc.buffer_type != BufferType::Gpu {
            return Err(SupaSimError::BufferExportError(
                "Exportable buffers can only be of `Gpu` type".to_string(),
            ));
        }
        let s = self.inner()?;
        let inner = unsafe {
            s.inner
                .lock()
                .as_mut()
                .unwrap()
                .create_buffer(&(*desc).into())
        }
        .map_supasim()?;
        let buffer_type_is_cpu = match desc.buffer_type {
            BufferType::Download | BufferType::Upload => true,
            BufferType::Gpu => false,
            BufferType::Automatic => unimplemented!(),
        };
        let cpu_available = s.inner_properties.is_unified_memory || buffer_type_is_cpu;
        let gpu_available = s.inner_properties.is_unified_memory || !buffer_type_is_cpu;
        let b = Buffer::from_inner(BufferInner {
            _phantom: Default::default(),
            _is_destroyed: false,
            instance: self.clone(),
            inner: Some(inner),
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
            id: Index::DANGLING,
            _semaphores: Vec::new(),
            create_info: *desc,
            last_used: 0,
            slice_tracker: SliceTracker::new(gpu_available, cpu_available),
            is_currently_external: false,
        });
        b.inner_mut()?.id = s.buffers.lock().insert(Some(b.downgrade()));
        Ok(b)
    }
    pub fn submit_commands(
        &self,
        recorders: &mut [crate::CommandRecorder<B>],
    ) -> SupaSimResult<B, WaitHandle<B>> {
        self.check_destroyed()?;
        let submission_idx;
        {
            let s = self.inner()?;

            let mut recorder_locks = Vec::new();
            for r in recorders.iter_mut() {
                r.check_destroyed()?;
                recorder_locks.push(r.inner_mut()?);
            }
            let mut recorder_inners = Vec::new();
            for r in &mut recorder_locks {
                recorder_inners.push(&mut **r);
            }

            let mut recorder = if let Some(mut r) = s.hal_command_recorders.lock().pop() {
                unsafe {
                    r.clear(s.inner.lock().as_mut().unwrap()).map_supasim()?;
                }
                r
            } else {
                unsafe { s.inner.lock().as_mut().unwrap().create_recorder() }.map_supasim()?
            };
            let mut used_buffers = HashSet::new();
            let mut used_buffer_ranges = Vec::new();
            let mut used_kernels = Vec::new();
            let (dag, sync_info, resources) =
                sync::assemble_dag(&mut recorder_inners, &mut used_kernels, &s)?;
            for (&buf_id, ranges) in &sync_info {
                let b = s
                    .buffers
                    .lock()
                    .get(buf_id)
                    .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                    .as_ref()
                    .unwrap()
                    .upgrade()?;
                let _b = b.clone();
                let b_mut = _b.inner()?;
                for &range in ranges {
                    let id = b_mut.slice_tracker.acquire(
                        &s,
                        range,
                        if b_mut.slice_tracker.mutex.lock().gpu_available {
                            BufferUser::Gpu(u64::MAX)
                        } else {
                            BufferUser::Cross(u64::MAX)
                        },
                        false,
                    )?;
                    used_buffer_ranges.push((id, b.clone()))
                }
                used_buffers.insert(buf_id);
            }
            let sync_mode = s.inner_properties.sync_mode;
            drop(s);
            let bind_groups = match sync_mode {
                types::SyncMode::Dag => sync::record_dag(&dag, &mut recorder)?,
                types::SyncMode::VulkanStyle => {
                    let streams = sync::dag_to_command_streams(&dag, true)?;
                    sync::record_command_streams(
                        &streams,
                        self.clone(),
                        &mut recorder,
                        &resources.temp_copy_buffer,
                    )?
                }
                types::SyncMode::Automatic => {
                    let streams = sync::dag_to_command_streams(&dag, false)?;
                    sync::record_command_streams(
                        &streams,
                        self.clone(),
                        &mut recorder,
                        &resources.temp_copy_buffer,
                    )?
                }
            };
            let s = self.inner()?;
            let used_buffers: Vec<_> = used_buffers
                .iter()
                .map(|a| {
                    s.buffers
                        .lock()
                        .get(*a)
                        .unwrap()
                        .as_ref()
                        .unwrap()
                        .upgrade()
                        .unwrap()
                })
                .collect();
            submission_idx = s.sync_thread().submit_gpu(GpuSubmissionInfo {
                command_recorder: Some(recorder),
                bind_groups,
                used_buffer_ranges: used_buffer_ranges.clone(),
                used_buffers,
                used_resources: resources,
            })?;

            for kernel in &used_kernels {
                kernel.inner_mut()?.last_used = submission_idx;
            }
            for &buf_id in sync_info.keys() {
                let b = s
                    .buffers
                    .lock()
                    .get(buf_id)
                    .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                    .as_ref()
                    .unwrap()
                    .upgrade()?;
                let mut b_mut = b.inner_mut()?;
                b_mut.last_used = submission_idx;
            }
            for (id, b) in used_buffer_ranges {
                b.inner()?
                    .slice_tracker
                    .update_user_submission(id, submission_idx, &s);
            }
        }

        for recorder in recorders {
            recorder._destroy()?;
        }
        Ok(WaitHandle::from_inner(WaitHandleInner {
            _phantom: Default::default(),
            _is_destroyed: false,
            instance: self.clone(),
            index: submission_idx,
            id: Index::DANGLING,
            is_alive: true,
        }))
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
        unsafe { s.inner.lock().as_mut().unwrap().cleanup_cached_resources() }.map_supasim()?;
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
        let properties = self.inner()?.inner_properties;
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
                    instance
                        .inner
                        .lock()
                        .as_mut()
                        .unwrap()
                        .map_buffer(buffer_inner.inner.as_mut().unwrap())
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
            let mut buffer_datas = Vec::new();
            for b in buffers {
                let mut buffer = b.buffer.inner_mut()?;
                let _instance = buffer.instance.clone();
                let mut data;
                #[allow(clippy::uninit_vec)]
                {
                    data = Vec::with_capacity(b.len as usize);
                    unsafe {
                        data.set_len(b.len as usize);
                        _instance
                            .inner()?
                            .inner
                            .lock()
                            .as_mut()
                            .unwrap()
                            .read_buffer(buffer.inner.as_mut().unwrap(), b.start, &mut data)
                            .map_supasim()?;
                    };
                };
                buffer_datas.push(data);
            }
            for (i, a) in buffer_datas.iter_mut().enumerate() {
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
            for b in buffer_datas {
                std::mem::forget(b);
            }
            drop(mapped_buffers);
            error?;
        }
        Ok(())
    }
    pub fn destroy(&mut self) -> SupaSimResult<B, ()> {
        self._destroy()
    }
}
impl<B: hal::Backend> SupaSimInstanceInner<B> {
    fn destroy(&mut self) {
        println!("Instance destroying");
        // We unsafely pass a mutable reference to the sync thread, knowing that this thread will "join" it,
        // meaning that at no point is the mutable reference used by both at the same time.
        let mut sync_thread =
            std::mem::take::<Option<SyncThreadHandle<B>>>(unsafe { &mut *self.sync_thread.get() })
                .unwrap();
        let _ = sync_thread.wait_for_idle();
        let _ = sync_thread
            .sender
            .get_mut()
            .send(sync::SendSyncThreadEvent::WaitFinishAndShutdown);
        let _ = sync_thread.thread.join();
        if sync_thread.shared_thread.0.lock().error.is_some() {
            println!("Instance attempting to shutdown gracefully despite sync thread error");
        }
        for (_, cr) in std::mem::take(&mut *self.command_recorders.lock()) {
            if let Some(cr) = cr {
                if let Ok(cr) = cr.upgrade() {
                    if let Ok(mut cr2) = cr.inner_mut() {
                        cr2.destroy(self);
                    }
                    let _ = cr._destroy();
                }
            }
        }
        for (_, wh) in std::mem::take(&mut *self.wait_handles.lock()) {
            if let Ok(wh) = wh.upgrade() {
                if let Ok(mut wh2) = wh.inner_mut() {
                    wh2.destroy(self);
                }
                let _ = wh._destroy();
            }
        }
        for (_, kc) in std::mem::take(&mut *self.kernel_caches.lock()) {
            if let Some(kc) = kc {
                if let Ok(kc) = kc.upgrade() {
                    if let Ok(mut kc2) = kc.inner_mut() {
                        kc2.destroy(self);
                    }
                    let _ = kc._destroy();
                }
            }
        }
        for (_, b) in std::mem::take(&mut *self.buffers.lock()) {
            if let Some(b) = b {
                if let Ok(b) = b.upgrade() {
                    if let Ok(mut b2) = b.inner_mut() {
                        b2.destroy(self);
                    }
                    let _ = b._destroy();
                }
            }
        }
        for (_, k) in std::mem::take(&mut *self.kernels.lock()) {
            if let Ok(k) = k.upgrade() {
                if let Ok(mut k2) = k.inner_mut() {
                    k2.destroy(self);
                }
                let _ = k._destroy();
            }
        }
        unsafe {
            let _ = self.inner.lock().as_mut().unwrap().wait_for_idle().is_ok();
        }
    }
}
impl<B: hal::Backend> Drop for SupaSimInstanceInner<B> {
    fn drop(&mut self) {
        self.destroy();
    }
}
api_type!(Kernel, {
    instance: SupaSimInstance<B>,
    inner: Option<B::Kernel>,
    reflection_info: KernelReflectionInfo,
    last_used: u64,
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
            instance
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .destroy_kernel(inner)
                .unwrap();
        }
    }
}
impl<B: hal::Backend> Drop for KernelInner<B> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            if let Ok(instance) = self.instance.clone().inner() {
                self.destroy(&instance);
            }
        }
    }
}
api_type!(KernelCache, {
    instance: SupaSimInstance<B>,
    inner: Option<B::KernelCache>,
    id: Index,
},);
impl<B: hal::Backend> KernelCache<B> {
    pub fn get_data(self) -> SupaSimResult<B, Vec<u8>> {
        let mut inner = self.inner_mut()?;
        let instance = inner.instance.clone();
        let data = unsafe {
            instance
                .inner()?
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .get_kernel_cache_data(inner.inner.as_mut().unwrap())
        }
        .map_supasim()?;
        Ok(data)
    }
}
impl<B: hal::Backend> KernelCacheInner<B> {
    fn destroy(&mut self, instance: &InstanceState<B>) {
        instance.kernel_caches.lock().remove(self.id);
        let _ = unsafe {
            instance
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .destroy_kernel_cache(std::mem::take(&mut self.inner).unwrap())
        };
    }
}
impl<B: hal::Backend> Drop for KernelCacheInner<B> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            if let Ok(instance) = self.instance.clone().inner() {
                self.destroy(&instance);
            }
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
    instance: SupaSimInstance<B>,
    is_alive: bool,
    id: Index,
    commands: Vec<BufferCommand<B>>,
    writes_slice: Vec<u8>,
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
        if matches!(
            (
                src_buffer.inner()?.create_info.buffer_type,
                dst_buffer.inner()?.create_info.buffer_type,
            ),
            (BufferType::Download, _) | (_, BufferType::Upload)
        ) {
            return Err(SupaSimError::BufferLocalityViolated);
        }
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
        if offset % 4 != 0 || size % 4 != 0 {
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
        if buffer.inner()?.create_info.buffer_type == BufferType::Upload {
            return Err(SupaSimError::BufferLocalityViolated);
        }
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
            if !b.buffer.inner()?.create_info.buffer_type.can_be_on_gpu() {
                return Err(SupaSimError::BufferLocalityViolated);
            }
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
}
impl<B: hal::Backend> CommandRecorderInner<B> {
    fn destroy(&mut self, instance: &InstanceState<B>) {
        instance.command_recorders.lock().remove(self.id);
        self.is_alive = true;
    }
}
impl<B: hal::Backend> Drop for CommandRecorderInner<B> {
    fn drop(&mut self) {
        if self.is_alive {
            if let Ok(instance) = self.instance.clone().inner() {
                self.destroy(&instance);
            }
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
    instance: SupaSimInstance<B>,
    inner: Option<B::Buffer>,
    id: Index,
    _semaphores: Vec<(Index, BufferRange)>,
    create_info: BufferDescriptor,
    last_used: u64,
    slice_tracker: SliceTracker,
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
            instance
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .write_buffer(s.inner.as_mut().unwrap(), offset, slice)
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
            instance
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .read_buffer(s.inner.as_mut().unwrap(), offset, slice)
                .map_supasim()?;
        }
        drop(s);
        self.inner()?.slice_tracker.release(id);
        Ok(())
    }
    pub fn access(
        &self,
        offset: u64,
        len: u64,
        needs_mut: bool,
    ) -> SupaSimResult<B, MappedBuffer<B>> {
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
        if instance.inner_properties.map_buffers {
            let mapped_ptr = unsafe {
                instance
                    .inner
                    .lock()
                    .as_mut()
                    .unwrap()
                    .map_buffer(s.inner.as_mut().unwrap())
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
                instance
                    .inner
                    .lock()
                    .as_mut()
                    .unwrap()
                    .read_buffer(s.inner.as_mut().unwrap(), offset, &mut data)
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
    /// # Safety
    /// * The exported buffer must be destroyed before `external_wgpu` buffer is destroyed
    /// * Synchronization must be guaranteed by the user.
    #[cfg(feature = "wgpu")]
    pub unsafe fn export_to_wgpu(
        &self,
        device: WgpuDeviceExportInfo,
    ) -> SupaSimResult<B, hal::wgpu_dep::Buffer> {
        if !self.inner()?.create_info.can_export {
            return Err(SupaSimError::BufferExportError(
                "Attempted to export buffer not made for exporting".to_string(),
            ));
        }
        {
            let mut buffer_inner = self.inner_mut()?;
            let _instance = buffer_inner.instance.clone();
            let instance_inner = _instance.inner()?;
            let mut instance_lock = instance_inner.inner.lock();
            unsafe {
                if instance_lock
                    .as_mut()
                    .unwrap()
                    .can_share_memory_to_device(&device)
                    .map_supasim()?
                {
                    use hal::Buffer;
                    let res = buffer_inner
                        .inner
                        .as_mut()
                        .unwrap()
                        .share_to_device(instance_lock.as_mut().unwrap(), &device)
                        .map_supasim()?;
                    Ok(*res
                        .downcast::<hal::wgpu_dep::Buffer>()
                        .expect("Hal error: backend didn't downcast to wgpu::Buffer"))
                } else {
                    Err(SupaSimError::BufferExportError(
                        "Backend refused to export buffer".to_string(),
                    ))
                }
            }
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
    fn destroy(&mut self, instance: &InstanceState<B>) {
        instance.buffers.lock().remove(self.id);
        unsafe {
            instance
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .destroy_buffer(std::mem::take(&mut self.inner).unwrap())
                .unwrap();
        }
    }
}
impl<B: hal::Backend> Drop for BufferInner<B> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            if let Ok(instance) = self.instance.clone().inner() {
                self.destroy(&instance);
            }
        }
    }
}

/// If the entire buffer isn't the same type you are trying to read, read as bytes first then cast yourself.
/// SupaSim does checks for alignments and validates offsets with the size of types
pub struct MappedBuffer<'a, B: hal::Backend> {
    instance: SupaSimInstance<B>,
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
        if (self.len % size_of::<T>() as u64) == 0
        // Length of the type is a multiple of the buffer alignment
        && (((size_of::<T>() as u64 % self.buffer_align) == 0) || ((self.buffer_align % size_of::<T>() as u64) == 0))
            // The offset is reasonable given the size of T
            && (self.in_buffer_offset % size_of::<T>() as u64) == 0
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
        if (self.len % size_of::<T>() as u64) == 0
        // Length of the type is a multiple of the buffer alignment
        && (((size_of::<T>() as u64 % self.buffer_align) == 0) || ((self.buffer_align % size_of::<T>() as u64) == 0))
            // The offset is reasonable given the size of T
            && (self.in_buffer_offset % size_of::<T>() as u64) == 0
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
        if instance.inner_properties.map_buffers {
            // Nothing needs to be done for now
        } else {
            unsafe {
                let vec =
                    Vec::from_raw_parts(self.inner, self.len as usize, self.vec_capacity.unwrap());
                if self.was_used_mut {
                    instance
                        .inner
                        .lock()
                        .as_mut()
                        .unwrap()
                        .write_buffer(s.inner.as_mut().unwrap(), self.in_buffer_offset, &vec)
                        .unwrap();
                }
                // And then the vec gets dropped. Tada!
            }
        }
    }
}

api_type!(WaitHandle, {
    instance: SupaSimInstance<B>,
    /// Index of the submission
    index: u64,
    id: Index,
    is_alive: bool,
},);
impl<B: hal::Backend> WaitHandleInner<B> {
    fn destroy(&mut self, instance: &InstanceState<B>) {
        instance.wait_handles.lock().remove(self.id);
        self.is_alive = false;
    }
}
impl<B: hal::Backend> Drop for WaitHandleInner<B> {
    fn drop(&mut self) {
        if self.is_alive {
            if let Ok(instance) = self.instance.clone().inner() {
                self.destroy(&instance);
            }
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
struct SliceTrackerInner {
    uses: HashMap<BufferUserId, BufferUser>,
    current_id: u64,
    cpu_locked: Option<u64>,
    gpu_available: bool,
    cpu_available: bool,
}
struct SliceTracker {
    condvar: Condvar,
    /// Contains a submission index.
    ///
    /// Also, the higher level u64 is for the current id. Bool is for whether cpu work is prevented(used for mapping logic)
    mutex: Mutex<SliceTrackerInner>,
}
impl SliceTracker {
    pub fn new(gpu_available: bool, cpu_available: bool) -> Self {
        Self {
            condvar: Condvar::new(),
            mutex: Mutex::new(SliceTrackerInner {
                uses: HashMap::new(),
                current_id: 0,
                cpu_locked: None,
                gpu_available,
                cpu_available,
            }),
        }
    }
    pub fn acquire<B: hal::Backend>(
        &self,
        instance: &InstanceState<B>,
        range: BufferRange,
        user: BufferUser,
        bypass_gpu: bool,
    ) -> SupaSimResult<B, BufferUserId> {
        let mut lock = self.mutex.lock();
        if match user {
            BufferUser::Cpu => !lock.cpu_available,
            BufferUser::Cross(_) => !lock.cpu_available,
            BufferUser::Gpu(_) => !lock.gpu_available,
        } {
            return Err(SupaSimError::BufferLocalityViolated);
        }
        if !bypass_gpu {
            let mut cont = true;
            let mut gpu_submissions = Vec::new();
            while cont {
                let mut has_cpu = false;
                cont = false;
                gpu_submissions.clear();
                if user.submission_id().is_none() && lock.cpu_locked.is_some() {
                    cont = true;
                    gpu_submissions.push(lock.cpu_locked.unwrap());
                }
                for (&a, &submission) in &lock.uses {
                    if a.range.overlaps(&range) {
                        // If this is part of the same GPU submission, don't try to wait
                        if submission.submission_id() == user.submission_id()
                            && submission.submission_id().is_some()
                        {
                            continue;
                        }
                        cont = true;
                        if let Some(sub) = submission.submission_id() {
                            gpu_submissions.push(sub);
                        } else {
                            has_cpu = true;
                        }
                        break;
                    }
                }
                if cont && has_cpu {
                    self.condvar.wait(&mut lock);
                } else if cont {
                    let sub = *gpu_submissions.iter().min().unwrap();
                    if sub == u64::MAX {
                        self.condvar.wait(&mut lock);
                    } else {
                        drop(lock);
                        instance.sync_thread().wait_for(sub, true)?;
                        lock = self.mutex.lock();
                    }
                } else {
                    break;
                }
            }
        }
        let id = BufferUserId {
            range,
            id: lock.current_id,
        };
        lock.current_id += 1;
        lock.uses.insert(id, user);
        Ok(id)
    }
    pub fn update_user_submission<B: hal::Backend>(
        &self,
        user: BufferUserId,
        submission_id: u64,
        instance: &InstanceState<B>,
    ) {
        let mut lock = self.mutex.lock();
        if instance
            .sync_thread()
            .wait_for(submission_id, false)
            .unwrap()
        {
            lock.uses.remove(&user).unwrap();
        } else {
            lock.uses
                .get_mut(&user)
                .unwrap()
                .set_submission_id(submission_id);
        }
        self.condvar.notify_all();
    }
    pub fn release(&self, range: BufferUserId) {
        self.mutex.lock().uses.remove(&range);
        self.condvar.notify_all();
    }
    pub fn acquire_cpu<B: hal::Backend>(&self, submission_id: u64) -> SupaSimResult<B, ()> {
        let mut lock = self.mutex.lock();
        let mut cont = true;
        while cont {
            cont = false;
            for &submission in lock.uses.values() {
                if submission.submission_id().is_none() {
                    cont = true;
                    break;
                }
            }
            if cont {
                self.condvar.wait(&mut lock);
            }
        }
        lock.cpu_locked = Some(submission_id);
        Ok(())
    }
    pub fn release_cpu(&self) {
        self.mutex.lock().cpu_locked = None;
        self.condvar.notify_all();
    }
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
struct BufferUserId {
    range: BufferRange,
    id: u64,
}
#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum BufferUser {
    Gpu(u64),
    Cpu,
    Cross(u64),
}
impl BufferUser {
    fn submission_id(&self) -> Option<u64> {
        match self {
            Self::Gpu(a) | Self::Cross(a) => Some(*a),
            Self::Cpu => None,
        }
    }
    fn set_submission_id(&mut self, id: u64) {
        match self {
            Self::Gpu(v) => *v = id,
            Self::Cross(v) => *v = id,
            Self::Cpu => panic!(),
        }
    }
}

#[cfg(feature = "wgpu")]
pub use hal::WgpuDeviceExportInfo;
#[cfg(feature = "wgpu")]
pub use hal::wgpu_dep as wgpu;
