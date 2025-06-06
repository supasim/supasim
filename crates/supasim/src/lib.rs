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

#[cfg(test)]
mod tests;

mod sync;

use anyhow::anyhow;
use hal::{BackendInstance as _, RecorderSubmitInfo, Semaphore};
use std::collections::VecDeque;
use std::sync::Mutex;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use thiserror::Error;
use thunderdome::{Arena, Index};

pub use bytemuck;
pub use hal;
pub use kernels;
pub use types::{
    HalBufferType, KernelReflectionInfo, KernelTarget, MetalVersion, ShaderModel, SpirvVersion,
};

pub type UserBufferAccessClosure<'a, B> =
    Box<dyn FnOnce(&mut [MappedBuffer<'a, B>]) -> anyhow::Result<()>>;

struct InnerRef<'a, T>(RwLockReadGuard<'a, Option<T>>);
impl<T> Deref for InnerRef<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}
struct InnerRefMut<'a, T>(RwLockWriteGuard<'a, Option<T>>);
impl<T> Deref for InnerRefMut<'_, T> {
    type Target = T;
    fn deref(&self) -> &Self::Target {
        self.0.as_ref().unwrap()
    }
}
impl<T> DerefMut for InnerRefMut<'_, T> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        self.0.as_mut().unwrap()
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum BufferType {
    /// Used by kernels
    Storage,
    /// Used as an indirect buffer for indirect dispatch calls
    Indirect,
    /// Used for uniform data for kernels
    Uniform,
    /// Used to upload data to GPU
    Upload,
    /// Used to download data from GPU
    Download,
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
}
impl Default for BufferDescriptor {
    fn default() -> Self {
        Self {
            size: 0,
            buffer_type: BufferType::Storage,
            contents_align: 0,
            priority: 1.0,
        }
    }
}
impl From<BufferDescriptor> for types::HalBufferDescriptor {
    fn from(s: BufferDescriptor) -> types::HalBufferDescriptor {
        types::HalBufferDescriptor {
            size: s.size,
            memory_type: match s.buffer_type {
                BufferType::Storage => types::HalBufferType::Storage,
                BufferType::Indirect | BufferType::Uniform => types::HalBufferType::Other,
                BufferType::Upload => types::HalBufferType::Upload,
                BufferType::Download => types::HalBufferType::Download,
            },
            visible_to_renderer: false,
            indirect_capable: s.buffer_type == BufferType::Indirect,
            uniform: s.buffer_type == BufferType::Uniform,
            needs_flush: true,
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
    fn acquire(&self) -> SupaSimResult<B, ()> {
        let mut s = self.buffer.inner_mut()?;
        let _instance = s.instance.clone();
        let mut _instance = _instance.inner_mut()?;
        _instance.wait_for_submission(true, s.last_used)?;

        s.host_using.push(BufferRange {
            start: self.start,
            len: self.len,
            needs_mut: self.needs_mut,
        });
        Ok(())
    }
    fn release(&self) -> SupaSimResult<B, ()> {
        let mut s = self.buffer.inner_mut()?;
        let range = BufferRange {
            start: self.start,
            len: self.len,
            needs_mut: self.needs_mut,
        };
        // Clippy generates a confusing warning and nonsense solution
        #[allow(clippy::unnecessary_filter_map)]
        let (index_to_remove, _) = s
            .host_using
            .iter()
            .enumerate()
            .filter_map(|a| if *a.1 == range { Some(a) } else { None })
            .next()
            .unwrap();
        s.host_using.remove(index_to_remove);
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
                $($field)*
            }

            #[derive(Clone)]
            pub(crate) struct [<$name Weak>] <B: hal::Backend>(std::sync::Weak<RwLock<Option<[<$name Inner>]<B>>>>);
            #[allow(dead_code)]
            impl<B: hal::Backend> [<$name Weak>] <B> {
                pub(crate) fn upgrade(&self) -> SupaSimResult<B, $name<B>> {
                    Ok($name(self.0.upgrade().ok_or(SupaSimError::AlreadyDestroyed)?))
                }
            }

            // Outer type, with some helper methods
            #[derive(Clone)]
            $(
                #[$attr]
            )*
            pub struct $name <B: hal::Backend> (std::sync::Arc<RwLock<Option<[<$name Inner>]<B>>>>);
            #[allow(dead_code)]
            impl<B: hal::Backend> $name <B> {
                pub(crate) fn from_inner(inner: [<$name Inner>]<B>) -> Self {
                    Self(Arc::new(RwLock::new(Some(inner))))
                }
                pub(crate) fn inner(&self) -> SupaSimResult<B, InnerRef<[<$name Inner>]<B>>> {
                    let r = self.0.read().map_err(|e| SupaSimError::Poison(e.to_string()))?;
                    if r.is_some() {
                        Ok(InnerRef(r))
                    } else {
                        Err(SupaSimError::AlreadyDestroyed)
                    }
                }
                pub(crate) fn inner_mut(&self) -> SupaSimResult<B, InnerRefMut<[<$name Inner>]<B>>> {
                    let r = self.0.write().map_err(|e| SupaSimError::Poison(e.to_string()))?;
                    if r.is_some() {
                        Ok(InnerRefMut(r))
                    } else {
                        Err(SupaSimError::AlreadyDestroyed)
                    }
                }
                pub(crate) fn take_inner(&self) -> SupaSimResult<B, [<$name Inner>]<B>> {
                    let mut a = self.0.write().map_err(|e| SupaSimError::Poison(e.to_string()))?;
                    if a.is_some() {
                        Ok(std::mem::take(&mut *a).unwrap())
                    } else {
                        Err(SupaSimError::AlreadyDestroyed)
                    }
                }
                pub fn destroy(&self) -> SupaSimResult<B, ()> {
                    let mut writer = self.0.write().map_err(|e| SupaSimError::Poison(e.to_string()))?;
                    // We want the destructor to be run after the writer lock is lost
                    let _temp = std::mem::take(&mut *writer);
                    drop(writer);
                    Ok(())
                }
                pub(crate) fn downgrade(&self) -> [<$name Weak>]<B> {
                    [<$name Weak>](Arc::downgrade(&self.0))
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
    Poison(String),
    Other(anyhow::Error),
    AlreadyDestroyed,
    BufferRegionNotValid,
    ValidateIndirectUnsupported,
    UserClosure(anyhow::Error),
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
        write!(f, "{:?}", self)
    }
}
pub type SupaSimResult<B, T> = Result<T, SupaSimError<B>>;

#[derive(Clone, Copy, Debug)]
pub struct InstanceProperties {
    pub supports_pipeline_cache: bool,
    pub kernel_lang: KernelTarget,
    pub is_unified_memory: bool,
}
api_type!(SupaSimInstance, {
    /// The inner hal instance
    inner: Option<B::Instance>,
    /// The hal instance properties
    inner_properties: types::HalInstanceProperties,
    /// All created kernels
    kernels: Arena<Option<KernelWeak<B>>>,
    /// All created kernel caches
    kernel_caches: Arena<Option<KernelCacheWeak<B>>>,
    /// All created buffers
    buffers: Arena<Option<BufferWeak<B>>>,
    /// All wait handles created
    wait_handles: Arena<WaitHandleWeak<B>>,
    /// Hal wait handles not currently associated with any submission
    unused_semaphores: Vec<B::Semaphore>,
    /// All created command recorders
    command_recorders: Arena<Option<CommandRecorderWeak<B>>>,
    /// All command recorders that have already been submitted and the associated usage data
    submitted_command_recorders: VecDeque<SubmittedCommandRecorder<B>>,
    /// The number of command recorder sets that have been submitted so far
    submitted_semaphore_count: u64,
    /// The index of the first command recorder in the queue
    submitted_command_recorders_start: u64,
    /// Hal command recorders not currently in use
    hal_command_recorders: Vec<B::CommandRecorder>,
    /// User accessible kernel compiler state
    kernel_compiler: kernels::GlobalState,
    /// Used so that buffers can be prevented from being used multiple threads without blocking the entire instance
    usage_lock: Arc<Mutex<()>>,
},);
impl<B: hal::Backend> SupaSimInstance<B> {
    pub fn from_hal(mut hal: B::Instance) -> Self {
        let inner_properties = hal.get_properties();
        Self::from_inner(SupaSimInstanceInner {
            _phantom: Default::default(),
            inner: Some(hal),
            inner_properties,
            kernels: Arena::default(),
            kernel_caches: Arena::default(),
            buffers: Arena::default(),
            wait_handles: Arena::default(),
            unused_semaphores: Vec::new(),
            command_recorders: Arena::default(),
            submitted_command_recorders: VecDeque::new(),
            // These are 1 so that we can always wait for zero
            submitted_semaphore_count: 1,
            submitted_command_recorders_start: 1,
            hal_command_recorders: Vec::new(),
            kernel_compiler: kernels::GlobalState::new_from_env().unwrap(),
            usage_lock: Arc::new(Mutex::new(())),
        })
    }
    pub fn properties(&self) -> SupaSimResult<B, InstanceProperties> {
        let v = self.inner()?.inner_properties;
        Ok(InstanceProperties {
            supports_pipeline_cache: v.pipeline_cache,
            kernel_lang: v.kernel_lang,
            is_unified_memory: v.is_unified_memory,
        })
    }
    pub fn compile_kernel(
        &self,
        binary: &[u8],
        reflection_info: KernelReflectionInfo,
        cache: Option<&KernelCache<B>>,
    ) -> SupaSimResult<B, Kernel<B>> {
        let mut cache_lock = if let Some(cache) = cache {
            Some(cache.inner_mut()?)
        } else {
            None
        };
        let mut s = self.inner_mut()?;

        let kernel = unsafe {
            s.inner.as_mut().unwrap().compile_kernel(
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
            instance: self.clone(),
            inner: Some(kernel),
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
            id: Index::DANGLING,
            reflection_info,
        });
        k.inner_mut()?.id = s.kernels.insert(Some(k.downgrade()));
        Ok(k)
    }
    pub fn create_kernel_cache(&self, data: &[u8]) -> SupaSimResult<B, KernelCache<B>> {
        let mut s = self.inner_mut()?;
        let inner = unsafe { s.inner.as_mut().unwrap().create_kernel_cache(data) }.map_supasim()?;
        let k = KernelCache::from_inner(KernelCacheInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner: Some(inner),
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
            id: Index::DANGLING,
        });
        k.inner_mut()?.id = s.kernel_caches.insert(Some(k.downgrade()));
        Ok(k)
    }
    pub fn create_recorder(&self) -> SupaSimResult<B, CommandRecorder<B>> {
        let mut s = self.inner_mut()?;
        let r = CommandRecorder::from_inner(CommandRecorderInner {
            _phantom: Default::default(),
            instance: self.clone(),
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
            id: Index::DANGLING,
            commands: Vec::new(),
            is_alive: true,
        });
        r.inner_mut()?.id = s.command_recorders.insert(Some(r.downgrade()));
        Ok(r)
    }
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> SupaSimResult<B, Buffer<B>> {
        let mut s = self.inner_mut()?;
        let inner =
            unsafe { s.inner.as_mut().unwrap().create_buffer(&(*desc).into()) }.map_supasim()?;
        let b = Buffer::from_inner(BufferInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner: Some(inner),
            // Yes its UB, but id doesn't have any destructor and just contains two numbers
            id: Index::DANGLING,
            _semaphores: Vec::new(),
            host_using: Vec::new(),
            create_info: *desc,
            last_used: 0,
            num_used_mappings: 0,
        });
        b.inner_mut()?.id = s.buffers.insert(Some(b.downgrade()));
        Ok(b)
    }
    pub fn submit_commands(
        &self,
        recorders: &mut [CommandRecorder<B>],
    ) -> SupaSimResult<B, WaitHandle<B>> {
        // This code is terrible
        // I sincerely apologize to anyone trying to read(my future self)

        let mut s = self.inner_mut()?;
        {
            let mut recorder_locks = Vec::new();
            for r in recorders.iter_mut() {
                recorder_locks.push(r.inner_mut()?);
            }
            let mut recorder_inners = Vec::new();
            for r in &mut recorder_locks {
                recorder_inners.push(&mut **r);
            }

            let mut recorder = if let Some(r) = s.hal_command_recorders.pop() {
                r
            } else {
                unsafe { s.inner.as_mut().unwrap().create_recorder() }.map_supasim()?
            };
            let (dag, sync_info) = sync::assemble_dag(&mut recorder_inners)?;
            let mut used_buffers = Vec::new();
            for (buf_id, ranges) in sync_info {
                let b = s
                    .buffers
                    .get(buf_id)
                    .ok_or(SupaSimError::AlreadyDestroyed)?
                    .as_ref()
                    .unwrap()
                    .upgrade()?;
                b.inner_mut()?.last_used = s.submitted_semaphore_count;
                for range in ranges {
                    used_buffers.push(BufferSlice {
                        buffer: b.clone(),
                        start: range.start,
                        len: range.len,
                        needs_mut: range.needs_mut,
                    })
                }
            }
            let sync_mode = s.inner_properties.sync_mode;
            drop(s);
            let bind_groups = match sync_mode {
                types::SyncMode::Dag => sync::record_dag(&dag, &mut recorder)?,
                types::SyncMode::VulkanStyle => {
                    let streams = sync::dag_to_command_streams(&dag, true)?;
                    sync::record_command_streams(&streams, self.clone(), &mut recorder)?
                }
                types::SyncMode::Automatic => {
                    let streams = sync::dag_to_command_streams(&dag, false)?;
                    sync::record_command_streams(&streams, self.clone(), &mut recorder)?
                }
            };
            let mut s = self.inner_mut()?;
            let semaphore = if let Some(s) = s.unused_semaphores.pop() {
                s
            } else {
                unsafe { s.inner.as_mut().unwrap().create_semaphore().map_supasim()? }
            };
            let mut submit_info = RecorderSubmitInfo {
                command_recorder: &mut recorder,
                wait_semaphores: &mut [],
                signal_semaphore: Some(&semaphore),
            };
            unsafe {
                s.inner
                    .as_mut()
                    .unwrap()
                    .submit_recorders(std::slice::from_mut(&mut submit_info))
                    .map_supasim()?;
            }
            s.submitted_command_recorders
                .push_back(SubmittedCommandRecorder {
                    command_recorders: vec![recorder],
                    buffers_to_destroy: Vec::new(),
                    bind_groups,
                    used_semaphore: semaphore,
                    used_buffers,
                });
        }
        for recorder in recorders {
            recorder.destroy()?;
        }
        let mut s = self.inner_mut()?;
        let index = s.submitted_semaphore_count;
        s.submitted_semaphore_count += 1;
        Ok(WaitHandle::from_inner(WaitHandleInner {
            _phantom: Default::default(),
            instance: self.clone(),
            index,
            id: Index::DANGLING,
            is_alive: true,
        }))
    }
    pub fn wait_for_idle(&self, _timeout: f32) -> SupaSimResult<B, ()> {
        let mut s = self.inner_mut()?;
        let idx = s.submitted_semaphore_count - 1;
        s.wait_for_submission(true, idx)?;
        Ok(())
    }
    /// Do work which is queued but not yet started for batching reasons. Currently a NOOP
    pub fn do_busywork(&self) -> SupaSimResult<B, ()> {
        Ok(())
    }
    /// Clear cached resources. This would be useful if the usage requirements have just
    /// dramatically changed, for example after startup or between different simulations.
    /// Currently a NOOP.
    pub fn clear_cached_resources(&self) -> SupaSimResult<B, ()> {
        let mut s = self.inner_mut()?;
        unsafe { s.inner.as_mut().unwrap().cleanup_cached_resources() }.map_supasim()?;
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
        let (_lock, properties) = {
            let inner = self.inner()?;
            (inner.usage_lock.clone(), inner.inner_properties)
        };
        let _lock2 = _lock.lock().map_err(|_| SupaSimError::Other(anyhow!("Failed to get usage lock for a buffer - an error has already occured in another thread")))?;
        let mut mapped_buffers = Vec::with_capacity(buffers.len());
        for b in buffers {
            b.validate()?;
            b.acquire()?;
        }
        if properties.map_buffers {
            let mut instance = self.inner_mut()?;
            #[allow(clippy::never_loop)]
            for &b in buffers {
                let mut buffer_inner = b.buffer.inner_mut()?;
                buffer_inner.num_used_mappings += 1;
                let mapping = unsafe {
                    instance
                        .inner
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
                            .inner_mut()?
                            .inner
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
        for b in buffers {
            b.release()?;
        }
        drop(_lock2);
        Ok(())
    }
}
impl<B: hal::Backend> Drop for SupaSimInstanceInner<B> {
    fn drop(&mut self) {
        self.wait_for_submission(true, self.submitted_semaphore_count - 1)
            .unwrap();
        for (_, cr) in std::mem::take(&mut self.command_recorders) {
            if let Some(cr) = cr {
                if let Ok(cr) = cr.upgrade() {
                    if let Ok(mut cr2) = cr.inner_mut() {
                        cr2.destroy(self);
                    }
                    let _ = cr.destroy();
                }
            }
        }
        for (_, wh) in std::mem::take(&mut self.wait_handles) {
            if let Ok(wh) = wh.upgrade() {
                if let Ok(mut wh2) = wh.inner_mut() {
                    wh2.destroy(self);
                }
                let _ = wh.destroy();
            }
        }
        for (_, kc) in std::mem::take(&mut self.kernel_caches) {
            if let Some(kc) = kc {
                if let Ok(kc) = kc.upgrade() {
                    if let Ok(mut kc2) = kc.inner_mut() {
                        kc2.destroy(self);
                    }
                    let _ = kc.destroy();
                }
            }
        }
        for (_, b) in std::mem::take(&mut self.buffers) {
            if let Some(b) = b {
                if let Ok(b) = b.upgrade() {
                    if let Ok(mut b2) = b.inner_mut() {
                        b2.destroy(self);
                    }
                    let _ = b.destroy();
                }
            }
        }
        for (_, k) in std::mem::take(&mut self.kernels) {
            if let Some(k) = k {
                if let Ok(k) = k.upgrade() {
                    if let Ok(mut k2) = k.inner_mut() {
                        k2.destroy(self);
                    }
                    let _ = k.destroy();
                }
            }
        }
        unsafe {
            self.inner.as_mut().unwrap().wait_for_idle().unwrap();
        }
    }
}
impl<B: hal::Backend> SupaSimInstanceInner<B> {
    /// Returns whether the operation has completed
    pub fn wait_for_submission(&mut self, force_wait: bool, id: u64) -> SupaSimResult<B, bool> {
        if id < self.submitted_command_recorders_start {
            return Ok(true);
        }
        while self.submitted_command_recorders_start <= id {
            let thing = &mut self.submitted_command_recorders[0];
            if force_wait {
                unsafe { thing.used_semaphore.wait().map_supasim()? };
            } else if !unsafe { thing.used_semaphore.is_signalled().map_supasim()? } {
                return Ok(false);
            }
            //unsafe { self.inner.wait_for_idle().map_supasim()? };
            let mut item = self.submitted_command_recorders.pop_front().unwrap();
            self.submitted_command_recorders_start += 1;
            for b in item.buffers_to_destroy {
                unsafe {
                    self.inner
                        .as_mut()
                        .unwrap()
                        .destroy_buffer(b)
                        .map_supasim()?
                };
            }
            for (bg, kernel) in item.bind_groups {
                let mut k = kernel.inner_mut()?;
                unsafe {
                    self.inner
                        .as_mut()
                        .unwrap()
                        .destroy_bind_group(k.inner.as_mut().unwrap(), bg)
                        .map_supasim()?;
                }
            }
            self.hal_command_recorders.extend(item.command_recorders);
            unsafe { item.used_semaphore.reset().map_supasim()? };
            self.unused_semaphores.push(item.used_semaphore);
        }
        Ok(true)
    }
}
api_type!(Kernel, {
    instance: SupaSimInstance<B>,
    inner: Option<B::Kernel>,
    id: Index,
    reflection_info: KernelReflectionInfo,
},);
impl<B: hal::Backend> std::fmt::Debug for Kernel<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Kernel(undecorated)")
    }
}
impl<B: hal::Backend> KernelInner<B> {
    pub fn destroy(&mut self, instance: &mut SupaSimInstanceInner<B>) {
        instance.kernels.remove(self.id);
        let kernel = std::mem::take(&mut self.inner).unwrap();
        let _ = unsafe { instance.inner.as_mut().unwrap().destroy_kernel(kernel) };
    }
}
impl<B: hal::Backend> Drop for KernelInner<B> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            if let Ok(mut instance) = self.instance.clone().inner_mut() {
                self.destroy(&mut instance);
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
                .inner_mut()?
                .inner
                .as_mut()
                .unwrap()
                .get_kernel_cache_data(inner.inner.as_mut().unwrap())
        }
        .map_supasim()?;
        Ok(data)
    }
}
impl<B: hal::Backend> KernelCacheInner<B> {
    fn destroy(&mut self, instance: &mut SupaSimInstanceInner<B>) {
        instance.kernel_caches.remove(self.id);
        let _ = unsafe {
            instance
                .inner
                .as_mut()
                .unwrap()
                .destroy_kernel_cache(std::mem::take(&mut self.inner).unwrap())
        };
    }
}
impl<B: hal::Backend> Drop for KernelCacheInner<B> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            if let Ok(mut instance) = self.instance.clone().inner_mut() {
                self.destroy(&mut instance);
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
    CopyBufferToBuffer {
        src_buffer: Buffer<B>,
        dst_buffer: Buffer<B>,
        src_offset: u64,
        dst_offset: u64,
        len: u64,
    },
    /// Kernel, workgroup size
    KernelDispatch {
        kernel: Kernel<B>,
        workgroup_dims: [u32; 3],
    },
    Dummy,
}
struct SubmittedCommandRecorder<B: hal::Backend> {
    used_buffers: Vec<BufferSlice<B>>,
    command_recorders: Vec<B::CommandRecorder>,
    used_semaphore: B::Semaphore,
    /// Buffers that are waiting for this to be destroyed
    buffers_to_destroy: Vec<B::Buffer>,
    bind_groups: Vec<(B::BindGroup, Kernel<B>)>,
}
api_type!(CommandRecorder, {
    instance: SupaSimInstance<B>,
    is_alive: bool,
    id: Index,
    commands: Vec<BufferCommand<B>>,
},);
impl<B: hal::Backend> CommandRecorder<B> {
    pub fn copy_buffer(
        &self,
        src_buffer: Buffer<B>,
        dst_buffer: Buffer<B>,
        src_offset: u64,
        dst_offset: u64,
        len: u64,
    ) -> SupaSimResult<B, ()> {
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
                inner: BufferCommandInner::CopyBufferToBuffer {
                    src_buffer,
                    dst_buffer,
                    src_offset,
                    dst_offset,
                    len,
                },
                buffers: vec![src_slice, dst_slice],
            });
        }
        Ok(())
    }
    pub fn dispatch_kernel(
        &self,
        kernel: Kernel<B>,
        buffers: &[&BufferSlice<B>],
        workgroup_dims: [u32; 3],
    ) -> SupaSimResult<B, ()> {
        if buffers.len() != kernel.inner()?.reflection_info.num_buffers as usize {
            return Err(SupaSimError::Other(anyhow!(
                "Incorrect number of buffers passed to dispatch_kernel"
            )));
        }
        for b in buffers {
            b.validate()?;
        }
        let mut s = self.inner_mut()?;
        s.commands.push(BufferCommand {
            inner: BufferCommandInner::KernelDispatch {
                kernel,
                workgroup_dims,
            },
            buffers: buffers.iter().map(|&b| b.clone()).collect(),
        });
        Ok(())
    }
}
impl<B: hal::Backend> CommandRecorderInner<B> {
    fn destroy(&mut self, instance: &mut SupaSimInstanceInner<B>) {
        instance.command_recorders.remove(self.id);
        self.is_alive = true;
    }
}
impl<B: hal::Backend> Drop for CommandRecorderInner<B> {
    fn drop(&mut self) {
        if self.is_alive {
            if let Ok(mut instance) = self.instance.clone().inner_mut() {
                self.destroy(&mut instance);
            }
        }
    }
}

#[derive(Clone, Copy, PartialEq, Eq)]
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

api_type!(Buffer, {
    instance: SupaSimInstance<B>,
    inner: Option<B::Buffer>,
    id: Index,
    _semaphores: Vec<(Index, BufferRange)>,
    host_using: Vec<BufferRange>,
    create_info: BufferDescriptor,
    last_used: u64,
    num_used_mappings: u64,
},);
impl<B: hal::Backend> Buffer<B> {
    pub fn write<T: bytemuck::Pod>(&self, offset: usize, data: &[T]) -> SupaSimResult<B, ()> {
        let slice = BufferSlice {
            buffer: self.clone(),
            start: offset as u64,
            len: data.len() as u64,
            needs_mut: true,
        };
        slice.validate_with_align(size_of::<T>() as u64)?;
        let mut s = self.inner_mut()?;
        s.num_used_mappings += 1; // This is unnecessary at the moment
        let _instance = s.instance.clone();
        let mut instance = _instance.inner_mut()?;
        let slice = bytemuck::cast_slice::<T, u8>(data);
        unsafe {
            instance
                .inner
                .as_mut()
                .unwrap()
                .write_buffer(s.inner.as_mut().unwrap(), offset as u64, slice)
                .map_supasim()?;
        }
        s.num_used_mappings -= 1;
        Ok(())
    }
    pub fn read<T: bytemuck::Pod>(&self, offset: usize, out: &mut [T]) -> SupaSimResult<B, ()> {
        let slice = BufferSlice {
            buffer: self.clone(),
            start: offset as u64,
            len: out.len() as u64,
            needs_mut: false,
        };
        slice.validate_with_align(size_of::<T>() as u64)?;
        let mut s = self.inner_mut()?;
        let _instance = s.instance.clone();
        let mut instance = _instance.inner_mut()?;
        let slice = bytemuck::cast_slice_mut::<T, u8>(out);
        unsafe {
            instance
                .inner
                .as_mut()
                .unwrap()
                .read_buffer(s.inner.as_mut().unwrap(), offset as u64, slice)
                .map_supasim()?;
        }
        Ok(())
    }
    pub fn access(
        &self,
        offset: usize,
        len: usize,
        needs_mut: bool,
    ) -> SupaSimResult<B, MappedBuffer<B>> {
        let slice = BufferSlice {
            buffer: self.clone(),
            start: offset as u64,
            len: len as u64,
            needs_mut,
        };
        slice.validate()?;
        let _instance = self.inner()?.instance.clone();
        let mut instance = _instance.inner_mut()?;
        let mut s = self.inner_mut()?;
        let buffer_align = s.create_info.contents_align;
        if instance.inner_properties.map_buffers {
            s.num_used_mappings += 1;
            let mapped_ptr = unsafe {
                instance
                    .inner
                    .as_mut()
                    .unwrap()
                    .map_buffer(s.inner.as_mut().unwrap())
                    .map_supasim()?
                    .add(offset)
            };
            drop(instance);
            Ok(MappedBuffer {
                instance: _instance,
                inner: mapped_ptr,
                len: len as u64,
                has_mut: needs_mut,
                buffer: slice.buffer,
                buffer_align,
                in_buffer_offset: offset as u64,
                was_used_mut: false,
                vec_capacity: None,
                _p: Default::default(),
            })
        } else {
            let mut data = Vec::with_capacity(len);
            unsafe {
                instance
                    .inner
                    .as_mut()
                    .unwrap()
                    .read_buffer(s.inner.as_mut().unwrap(), offset as u64, &mut data)
                    .map_supasim()?;
            }
            drop(instance);
            let out = Ok(MappedBuffer {
                instance: _instance,
                inner: data.as_mut_ptr(),
                len: len as u64,
                buffer_align,
                in_buffer_offset: offset as u64,
                has_mut: needs_mut,
                was_used_mut: false,
                buffer: slice.buffer,
                vec_capacity: Some(data.capacity()),
                _p: Default::default(),
            });
            std::mem::forget(data);
            out
        }
    }
}
impl<B: hal::Backend> std::fmt::Debug for Buffer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Buffer(undecorated)")
    }
}
impl<B: hal::Backend> BufferInner<B> {
    fn destroy(&mut self, instance: &mut SupaSimInstanceInner<B>) {
        instance.buffers.remove(self.id);

        if instance.submitted_command_recorders_start <= self.last_used {
            let start = instance.submitted_command_recorders_start;
            instance.submitted_command_recorders[(self.last_used - start) as usize]
                .buffers_to_destroy
                .push(std::mem::take(&mut self.inner).unwrap());
        } else {
            let _ = unsafe {
                instance
                    .inner
                    .as_mut()
                    .unwrap()
                    .destroy_buffer(std::mem::take(&mut self.inner).unwrap())
            };
        }
    }
}
impl<B: hal::Backend> Drop for BufferInner<B> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            if let Ok(mut instance) = self.instance.clone().inner_mut() {
                self.destroy(&mut instance);
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
        && (size_of::<T>() as u64 % self.buffer_align) == 0
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
        && (size_of::<T>() as u64 % self.buffer_align) == 0
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
        let mut instance = self.instance.inner_mut().unwrap();
        let mut s = self.buffer.inner_mut().unwrap();
        if instance.inner_properties.map_buffers {
            s.num_used_mappings -= 1;
            if s.num_used_mappings == 0 {
                unsafe {
                    instance
                        .inner
                        .as_mut()
                        .unwrap()
                        .unmap_buffer(s.inner.as_mut().unwrap())
                        .unwrap();
                }
            }
        } else {
            unsafe {
                let vec =
                    Vec::from_raw_parts(self.inner, self.len as usize, self.vec_capacity.unwrap());
                if self.was_used_mut {
                    instance
                        .inner
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
    fn destroy(&mut self, instance: &mut SupaSimInstanceInner<B>) {
        instance.wait_handles.remove(self.id);
        self.is_alive = false;
    }
}
impl<B: hal::Backend> Drop for WaitHandleInner<B> {
    fn drop(&mut self) {
        if self.is_alive {
            if let Ok(mut instance) = self.instance.clone().inner_mut() {
                self.destroy(&mut instance);
            }
        }
    }
}
impl<B: hal::Backend> WaitHandle<B> {
    pub fn wait(&self) -> SupaSimResult<B, ()> {
        let s = self.inner_mut()?;
        let _i = s.instance.clone();
        let mut instance = _i.inner_mut()?;
        instance.wait_for_submission(true, s.index)?;
        Ok(())
    }
}
#[allow(clippy::type_complexity)]
pub type CpuCallback<B> = (
    Box<dyn Fn(Vec<MappedBuffer<B>>) -> Result<(), SupaSimError<B>>>,
    Vec<Buffer<B>>,
);
