/* BEGIN LICENSE
  SupaSim, a GPUGPU and simulation toolkit.
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

mod sync;

use hal::{BackendInstance as _, RecorderSubmitInfo, Semaphore};
use std::{
    hash::Hash,
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use thiserror::Error;
use types::SyncMode;

pub use bytemuck;
pub use hal;
pub use shaders;
pub use types::{
    BufferType, ShaderModel, ShaderReflectionInfo, ShaderResourceType, ShaderTarget, SpirvVersion,
};

pub type UserBufferAccessClosure<B> = Box<dyn FnOnce(&mut [MappedBuffer<B>]) -> anyhow::Result<()>>;

/// Contains the index, and a certain "random" value to check if a destroyed thing has been replaced
struct Id<T>(u32, u32, PhantomData<T>);
impl<T> Default for Id<T> {
    fn default() -> Self {
        Self(u32::MAX, 0, Default::default())
    }
}
impl<T> Clone for Id<T> {
    fn clone(&self) -> Self {
        *self
    }
}
impl<T> Copy for Id<T> {}
impl<T> std::fmt::Debug for Id<T> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("()")
    }
}
impl<T> PartialEq for Id<T> {
    fn eq(&self, other: &Self) -> bool {
        self.0 == other.0 && self.1 == other.1
    }
}
impl<T> Eq for Id<T> {}
impl<T> Hash for Id<T> {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        state.write_u32(self.0);
        state.write_u32(self.1);
    }
}
pub(crate) fn convert_id<A, B>(id: Id<A>) -> Id<B> {
    Id(id.0, id.1, Default::default())
}
struct Tracker<T> {
    list: Vec<(u32, T)>,
    unused: Vec<u32>,
    current_identifier: u32,
}
impl<T> Default for Tracker<T> {
    fn default() -> Self {
        Self {
            list: Vec::new(),
            unused: Vec::new(),
            current_identifier: 0,
        }
    }
}
impl<T> Tracker<T> {
    pub fn get<O: AsId<T>>(&self, id: Id<O>) -> Option<&T> {
        let value = &self.list[id.0 as usize];
        if value.0 != id.1 {
            return None;
        }
        Some(&value.1)
    }
    pub fn get_mut<O: AsId<T>>(&mut self, id: Id<O>) -> Option<&mut T> {
        let value = &mut self.list[id.0 as usize];
        if value.0 != id.1 {
            return None;
        }
        Some(&mut value.1)
    }
    pub fn add<O: AsId<T>>(&mut self, value: T) -> Id<O> {
        // TODO: currently this overwrites the previous value. Make it soemtimes preserve if that is desired
        let identifier = self.current_identifier;
        self.current_identifier = self.current_identifier.wrapping_add(1);
        let idx = match self.unused.pop() {
            Some(idx) => {
                self.list[idx as usize] = (identifier, value);
                idx
            }
            None => {
                self.list.push((identifier, value));
                self.list.len() as u32 - 1
            }
        };
        Id(idx, identifier, Default::default())
    }
    pub fn remove<O: AsId<T>>(&mut self, id: Id<O>, replace_with: Option<T>) {
        let value = &mut self.list[id.0 as usize];
        if value.0 == id.1 {
            value.0 = u32::MAX;
            if let Some(v) = replace_with {
                value.1 = v;
            }
        }
    }
    pub fn acquire<O: AsId<T>>(&mut self, idx: u32) -> Id<O> {
        let v = self.current_identifier;
        self.current_identifier += 1;
        self.list[idx as usize].0 = v;
        Id(idx, v, Default::default())
    }
}

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

#[derive(Clone, Copy, Debug)]
pub struct BufferDescriptor {
    /// The size needed in bytes
    pub size: u64,
    /// Whether this can be used as an indirect buffer for indirect dispatch calls
    pub indirect_capable: bool,
    /// Whether the buffer is used for uniform data(small amount of data which can be **read** from all threads in a dispatch quickly)
    pub uniform: bool,
    /// The value that the contents of the buffer must be aligned to. This is important for when supasim must detect
    pub contents_align: u64,
    /// Currently unused. In the future this may be used to prefer keeping some buffers in memory when device runs out of memory and swapping becomes necessary
    pub priority: f32,
}
impl Default for BufferDescriptor {
    fn default() -> Self {
        Self {
            size: 0,
            indirect_capable: false,
            uniform: false,
            contents_align: 0,
            priority: 1.0,
        }
    }
}
impl From<BufferDescriptor> for types::BufferDescriptor {
    fn from(s: BufferDescriptor) -> types::BufferDescriptor {
        types::BufferDescriptor {
            size: s.size,
            memory_type: if s.indirect_capable || s.uniform {
                types::BufferType::Other
            } else {
                types::BufferType::Storage
            },
            visible_to_renderer: false,
            indirect_capable: s.indirect_capable,
            uniform: s.uniform,
            needs_flush: true,
        }
    }
}

#[derive(Clone)]
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
        let _instance = _instance.inner_mut()?;

        s.host_using.push(BufferRange {
            start: self.start,
            len: self.len,
            needs_mut: self.needs_mut,
        });
        // We also need to check it isn't in use
        todo!()
    }
    fn release(&self) -> SupaSimResult<B, ()> {
        let s = self.buffer.inner_mut()?;
        let range = BufferRange {
            start: self.start,
            len: self.len,
            needs_mut: self.needs_mut,
        };
        // I don't understand clippy's recommendation here. It gives invalid code that is nonsensical
        #[allow(clippy::unnecessary_filter_map)]
        s.host_using
            .iter()
            .enumerate()
            .filter_map(|a| if *a.1 == range { Some(a) } else { None })
            .next()
            .unwrap();
        // Check it isn't in use
        todo!()
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
                pub(crate) fn as_inner(&self) -> SupaSimResult<B, [<$name Inner>]<B>> {
                    let mut a = self.0.write().map_err(|e| SupaSimError::Poison(e.to_string()))?;
                    if a.is_some() {
                        Ok(std::mem::take(&mut *a).unwrap())
                    } else {
                        Err(SupaSimError::AlreadyDestroyed)
                    }
                }
                pub fn destroy(&self) -> SupaSimResult<B, ()> {
                    *self.0.write().map_err(|e| SupaSimError::Poison(e.to_string()))? = None;
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
    // Rust thinks that B::Error could be SupaSimError. Nevermind that this would be a recursive definition
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
    pub supports_indirect_dispatch: bool,
    pub shader_type: types::ShaderTarget,
    pub is_unified_memory: bool,
}
api_type!(Instance, {
    inner: B::Instance,
    inner_properties: types::InstanceProperties,
    kernels: Tracker<Option<KernelWeak<B>>>,
    kernel_caches: Tracker<Option<KernelCacheWeak<B>>>,
    command_recorders: Tracker<Option<CommandRecorderWeak<B>>>,
    buffers: Tracker<Option<BufferWeak<B>>>,
    wait_handles: Tracker<WaitHandleWeak<B>>,
    hal_command_recorders: Vec<B::CommandRecorder>,
    hal_bind_groups: Vec<B::BindGroup>,
},);
impl<B: hal::Backend> Instance<B> {
    pub fn from_hal(mut hal: B::Instance) -> Self {
        let inner_properties = hal.get_properties();
        Self::from_inner(InstanceInner {
            _phantom: Default::default(),
            inner: hal,
            inner_properties,
            kernels: Tracker::default(),
            kernel_caches: Tracker::default(),
            command_recorders: Tracker::default(),
            buffers: Tracker::default(),
            wait_handles: Tracker::default(),
            hal_command_recorders: Vec::new(),
            hal_bind_groups: Vec::new(),
        })
    }
    pub fn properties(&self) -> SupaSimResult<B, InstanceProperties> {
        let v = self.as_inner()?.inner_properties;
        Ok(InstanceProperties {
            supports_pipeline_cache: v.pipeline_cache,
            supports_indirect_dispatch: v.indirect,
            shader_type: v.shader_type,
            is_unified_memory: v.is_unified_memory,
        })
    }
    pub fn compile_kernel(
        &self,
        binary: &[u8],
        reflection: ShaderReflectionInfo,
        cache: Option<&KernelCache<B>>,
    ) -> SupaSimResult<B, Kernel<B>> {
        let mut cache_lock = if let Some(cache) = cache {
            Some(cache.inner_mut()?)
        } else {
            None
        };
        let mut s = self.inner_mut()?;

        let kernel = unsafe {
            s.inner.compile_kernel(
                binary,
                &reflection,
                if let Some(lock) = cache_lock.as_mut() {
                    Some(&mut lock.inner)
                } else {
                    None
                },
            )
        }
        .map_supasim()?;
        let k = Kernel::from_inner(KernelInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner: kernel,
            id: Default::default(),
        });
        k.inner_mut()?.id = s.kernels.add(Some(k.downgrade()));
        Ok(k)
    }
    pub fn create_kernel_cache(&self, data: &[u8]) -> SupaSimResult<B, KernelCache<B>> {
        let mut s = self.inner_mut()?;
        let inner = unsafe { s.inner.create_kernel_cache(data) }.map_supasim()?;
        let k = KernelCache::from_inner(KernelCacheInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner,
            id: Default::default(),
        });
        k.inner_mut()?.id = s.kernel_caches.add(Some(k.downgrade()));
        Ok(k)
    }
    pub fn create_recorder(&self, reusable: bool) -> SupaSimResult<B, CommandRecorder<B>> {
        let mut s = self.inner_mut()?;
        let r = CommandRecorder::from_inner(CommandRecorderInner {
            _phantom: Default::default(),
            instance: self.clone(),
            id: Default::default(),
            recorded: false,
            cleared: true,
            commands: Vec::new(),
            reusable,
            used_buffers: Vec::new(),
            current_iteration: 0,
            command_recorders: Vec::new(),
            num_semaphores: 0,
            semaphores: Vec::new(),
        });
        r.inner_mut()?.id = s.command_recorders.add(Some(r.downgrade()));
        Ok(r)
    }
    pub fn create_buffer(&self, desc: &BufferDescriptor) -> SupaSimResult<B, Buffer<B>> {
        let mut s = self.inner_mut()?;
        let inner = unsafe { s.inner.create_buffer(&(*desc).into()) }.map_supasim()?;
        let b = Buffer::from_inner(BufferInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner,
            id: Default::default(),
            _semaphores: Vec::new(),
            host_using: Vec::new(),
            create_info: *desc,
        });
        b.inner_mut()?.id = s.buffers.add(Some(b.downgrade()));
        Ok(b)
    }
    pub fn submit_commands(&self, recorders: &[CommandRecorder<B>]) -> SupaSimResult<B, ()> {
        for recorder_ref in recorders {
            let mut recorder = recorder_ref.inner_mut()?;
            let _instance = recorder.instance.clone();
            if !recorder.cleared && !recorder.recorded {
                todo!()
            }
            let needs_record = !recorder.recorded;
            if !recorder.reusable {
                recorder.recorded = false;
            }
            drop(recorder);
            if needs_record {
                recorder_ref.record()?;
            }
        }
        let mut recorded_locks = Vec::new();
        for recorder in recorders {
            recorded_locks.push(recorder.inner_mut()?);
        }
        let mut recorders = Vec::new();
        let mut semaphore_locks = Vec::new();
        let mut submit_infos: Vec<RecorderSubmitInfo<'_, B>> = Vec::new();
        {
            for recorder in &mut recorded_locks {
                let mut recorder_semaphores = Vec::new();
                for _ in 0..recorder.num_semaphores {
                    recorder_semaphores.push(self.acquire_wait_handle()?);
                    semaphore_locks.push(recorder_semaphores.last().unwrap().as_inner()?);
                }
                for cb in &mut recorder.command_recorders {
                    // More semaphore magic
                    recorders.push(cb);
                    submit_infos.push(RecorderSubmitInfo {
                        #[allow(invalid_value)]
                        command_recorder: unsafe { std::mem::zeroed() },
                        wait_semaphore: None,
                        signal_semaphore: None,
                    });
                }
            }
            for (i, r) in recorders.iter_mut().enumerate() {
                unsafe {
                    std::ptr::write(
                        &mut submit_infos[i].command_recorder as *mut _,
                        &mut r.inner,
                    )
                };
            }
            // TODO: finish this by working with semaphores
        }
        unsafe {
            self.inner_mut()?
                .inner
                .submit_recorders(&mut submit_infos)
                .map_supasim()?;
        }
        Ok(())
    }
    pub fn wait_for_idle(&self, _timeout: f32) -> SupaSimResult<B, ()> {
        let mut _s = self.inner_mut()?;
        let _s = &mut *_s;
        todo!()
    }
    pub fn do_busywork(&self) -> SupaSimResult<B, ()> {
        todo!()
    }
    pub fn clear_cached_resources(&self) -> SupaSimResult<B, ()> {
        let mut s = self.inner_mut()?;
        unsafe { s.inner.cleanup_cached_resources() }.map_supasim()?;
        todo!();
        //Ok(())
    }
    fn acquire_wait_handle(&self) -> SupaSimResult<B, WaitHandle<B>> {
        let mut s = self.inner_mut()?;
        let idx = if let Some(idx) = s.wait_handles.unused.pop() {
            s.wait_handles.acquire(idx)
        } else {
            let semaphore = unsafe { s.inner.create_semaphore() }.map_supasim()?;
            let id: Id<WaitHandle<B>> = s.wait_handles.add(
                WaitHandle::from_inner(WaitHandleInner {
                    instance: self.clone(),
                    _phantom: Default::default(),
                    inner: semaphore,
                    id: Id::default(),
                })
                .downgrade(),
            );
            s.wait_handles
                .get_mut(id)
                .unwrap()
                .upgrade()?
                .inner_mut()?
                .id = id;
            id
        };
        s.wait_handles.get(idx).unwrap().upgrade()
    }

    #[allow(clippy::type_complexity)]
    pub fn access_buffers(
        &self,
        closure: UserBufferAccessClosure<B>,
        buffers: &[&BufferSlice<B>],
    ) -> SupaSimResult<B, Option<WaitHandle<B>>> {
        let mut buffer_datas = Vec::new();
        for b in buffers {
            b.validate()?;
            b.acquire()?;
            let buffer = b.buffer.inner()?;
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
                        .read_buffer(&buffer.inner, b.start, &mut data)
                        .map_supasim()?;
                };
            };
            buffer_datas.push(data);
        }
        let mut mapped_buffers = Vec::new();
        for (i, a) in buffer_datas.iter_mut().enumerate() {
            let b = &buffers[i];
            let mapped = MappedBuffer {
                instance: self.clone(),
                inner: a.as_mut_ptr(),
                len: b.len,
                buffer: b.buffer.as_inner()?.id,
                has_mut: b.needs_mut,
                was_used_mut: false,
            };
            mapped_buffers.push(mapped);
        }
        closure(&mut mapped_buffers).map_err(|e| SupaSimError::UserClosure(e))?;
        // TODO: write buffers that need it
        drop(mapped_buffers);
        for b in buffers {
            b.release()?;
        }
        todo!()
    }
}
impl<B: hal::Backend> Drop for InstanceInner<B> {
    fn drop(&mut self) {
        let _ = unsafe { self.inner.wait_for_idle() };
        self.command_recorders.list.clear(); // These will call their destructors, politely taking care of themselves
        self.wait_handles.list.clear();
        self.kernel_caches.list.clear();
        self.buffers.list.clear();
        self.kernels.list.clear();
    }
}
api_type!(Kernel, {
    instance: Instance<B>,
    inner: B::Kernel,
    id: Id<Kernel<B>>,
},);
impl<B: hal::Backend> Kernel<B> {}
impl<B: hal::Backend> Drop for KernelInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.kernels.remove(self.id, Some(None));
            let _ = unsafe { instance.inner.destroy_kernel(std::ptr::read(&self.inner)) };
        }
    }
}
api_type!(KernelCache, {
    instance: Instance<B>,
    inner: B::KernelCache,
    id: Id<KernelCache<B>>,
},);
impl<B: hal::Backend> KernelCache<B> {
    pub fn get_data(self) -> SupaSimResult<B, Vec<u8>> {
        let mut inner = self.inner_mut()?;
        let instance = inner.instance.clone();
        let data = unsafe {
            instance
                .inner_mut()?
                .inner
                .get_kernel_cache_data(&mut inner.inner)
        }
        .map_supasim()?;
        Ok(data)
    }
}
impl<B: hal::Backend> Drop for KernelCacheInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.kernel_caches.remove(self.id, Some(None));
            let _ = unsafe {
                instance
                    .inner
                    .destroy_kernel_cache(std::ptr::read(&self.inner))
            };
        }
    }
}
struct BufferCommand<B: hal::Backend> {
    inner: BufferCommandInner<B>,
    buffers: Vec<BufferSlice<B>>,
    wait_handle: Option<WaitHandle<B>>,
}
enum BufferCommandInner<B: hal::Backend> {
    /// Kernel, workgroup size
    KernelDispatch(Kernel<B>, [u32; 3]),
    /// Kernel, buffer and offset, needs validation
    KernelDispatchIndirect(Kernel<B>, BufferSlice<B>, bool),
    CpuCode(UserBufferAccessClosure<B>),
    Dummy,
}
struct CommandRecorderData<B: hal::Backend> {
    inner: B::CommandRecorder,
    out_semaphore: u32,
    wait_semaphores: Vec<u32>,
}
api_type!(CommandRecorder, {
    instance: Instance<B>,
    id: Id<CommandRecorder<B>>,
    recorded: bool,
    cleared: bool,
    commands: Vec<BufferCommand<B>>,
    /// Used for tracking if the command recorder is cleared so previously used resources don't need to wait
    current_iteration: u64,
    used_buffers: Vec<BufferSlice<B>>,
    command_recorders: Vec<CommandRecorderData<B>>,
    reusable: bool,
    num_semaphores: u32,
    semaphores: Vec<Id<WaitHandle<B>>>,
},);
impl<B: hal::Backend> CommandRecorder<B> {
    pub fn dispatch_kernel(
        &self,
        shader: Kernel<B>,
        buffers: &[&BufferSlice<B>],
        workgroup_dims: [u32; 3],
        return_wait: bool,
    ) -> SupaSimResult<B, Option<WaitHandle<B>>> {
        for b in buffers {
            b.validate()?;
            self.inner_mut()?.used_buffers.push((*b).clone());
        }
        let wait_handle = if return_wait {
            Some(self.as_inner()?.instance.acquire_wait_handle()?)
        } else {
            None
        };
        let mut s = self.inner_mut()?;
        s.commands.push(BufferCommand {
            inner: BufferCommandInner::KernelDispatch(shader, workgroup_dims),
            buffers: buffers.iter().map(|&b| b.clone()).collect(),
            wait_handle: wait_handle.clone(),
        });
        s.recorded = false;
        Ok(wait_handle)
    }
    #[allow(clippy::too_many_arguments)]
    pub fn dispatch_kernel_indirect(
        &self,
        shader: Kernel<B>,
        buffers: &[&BufferSlice<B>],
        indirect_buffer: &BufferSlice<B>,
        validate_dispatch: bool,
        return_wait: bool,
    ) -> SupaSimResult<B, Option<WaitHandle<B>>> {
        for b in buffers {
            b.validate()?;
            self.inner_mut()?.used_buffers.push((*b).clone());
        }
        indirect_buffer.validate()?;
        if (indirect_buffer.start % 4) != 0 || indirect_buffer.len != 12 {
            return Err(SupaSimError::BufferRegionNotValid);
        }
        let wait_handle = if return_wait {
            Some(self.as_inner()?.instance.acquire_wait_handle()?)
        } else {
            None
        };
        if validate_dispatch {
            return Err(SupaSimError::ValidateIndirectUnsupported);
        }
        let mut s = self.inner_mut()?;
        s.commands.push(BufferCommand {
            inner: BufferCommandInner::KernelDispatchIndirect(
                shader,
                indirect_buffer.clone(),
                validate_dispatch,
            ),
            buffers: buffers.iter().map(|&b| b.clone()).collect(),
            wait_handle: wait_handle.clone(),
        });
        s.recorded = false;
        Ok(wait_handle)
    }
    pub fn cpu_code(
        &self,
        closure: UserBufferAccessClosure<B>,
        buffers: &[&BufferSlice<B>],
        return_wait: bool,
    ) -> SupaSimResult<B, Option<WaitHandle<B>>> {
        for b in buffers {
            b.validate()?;
            self.inner_mut()?.used_buffers.push((*b).clone());
        }
        let wait_handle = if return_wait {
            Some(self.as_inner()?.instance.acquire_wait_handle()?)
        } else {
            None
        };
        let mut s = self.inner_mut()?;
        s.commands.push(BufferCommand {
            inner: BufferCommandInner::CpuCode(closure),
            buffers: buffers.iter().map(|&b| b.clone()).collect(),
            wait_handle: wait_handle.clone(),
        });
        s.recorded = false;
        Ok(wait_handle)
    }
    pub fn clear(&self) -> SupaSimResult<B, ()> {
        let mut s = self.inner_mut()?;
        s.commands.clear();
        s.recorded = false;
        s.cleared = true;
        s.current_iteration += 1;
        s.used_buffers.clear();
        s.num_semaphores = 0;
        let instance = s.instance.clone();
        for cb in &mut s.command_recorders {
            unsafe {
                instance
                    .inner_mut()?
                    .inner
                    .clear_recorders(&mut [&mut cb.inner])
                    .map_supasim()?;
            };
        }
        while let Some(cr) = s.command_recorders.pop() {
            instance.inner_mut()?.hal_command_recorders.push(cr.inner);
        }
        todo!(); // Return command recorders, semaphores to pool
    }
    fn record(&self) -> SupaSimResult<B, ()> {
        let mut s = self.inner_mut()?;
        s.recorded = true;
        s.cleared = false;
        let _instance = s.instance.clone();
        let instance = _instance.inner_mut()?;
        match instance.inner_properties.sync_mode {
            SyncMode::Dag => todo!(), // We will work on this when cuda lands with cuda graphs
            SyncMode::Automatic | SyncMode::VulkanStyle => {
                let dag = sync::assemble_dag(&mut s)?;
                let streams = sync::dag_to_command_streams(
                    &dag,
                    instance.inner_properties.sync_mode == SyncMode::VulkanStyle,
                )?;
                sync::record_command_streams(&streams, &mut s)?;
            }
        }
        Ok(())
    }
}
impl<B: hal::Backend> Drop for CommandRecorderInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.command_recorders.remove(self.id, Some(None));
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
        self.start < other.start + other.len
            && other.start < self.start + self.len
            && (self.needs_mut || other.needs_mut)
    }
}

api_type!(Buffer, {
    instance: Instance<B>,
    inner: B::Buffer,
    id: Id<Buffer<B>>,
    _semaphores: Vec<(Id<WaitHandle<B>>, BufferRange)>,
    host_using: Vec<BufferRange>,
    create_info: BufferDescriptor,
},);
impl<B: hal::Backend> Drop for BufferInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.buffers.remove(self.id, Some(None));
            let _ = unsafe { instance.inner.destroy_buffer(std::ptr::read(&self.inner)) };
        }
    }
}

pub struct MappedBuffer<B: hal::Backend> {
    instance: Instance<B>,
    inner: *mut u8,
    len: u64,
    buffer: Id<Buffer<B>>,
    has_mut: bool,
    was_used_mut: bool,
}
impl<B: hal::Backend> MappedBuffer<B> {
    pub fn readable<T: bytemuck::Pod>(&self) -> SupaSimResult<B, &[T]> {
        // This code lol... maybe I need to do some major refactor this is gross
        let buffer_align = self
            .instance
            .inner()?
            .buffers
            .get(self.buffer)
            .as_ref()
            .ok_or(SupaSimError::AlreadyDestroyed)?
            .as_ref()
            .ok_or(SupaSimError::AlreadyDestroyed)?
            .upgrade()?
            .inner()?
            .create_info
            .contents_align;
        let s = unsafe { std::slice::from_raw_parts(self.inner, self.len as usize) };
        if (s.len() % size_of::<T>()) == 0 && (s.len() as u64 % buffer_align) == 0 {
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
        let buffer_align = self
            .instance
            .inner()?
            .buffers
            .get(self.buffer)
            .as_ref()
            .ok_or(SupaSimError::AlreadyDestroyed)?
            .as_ref()
            .ok_or(SupaSimError::AlreadyDestroyed)?
            .upgrade()?
            .inner()?
            .create_info
            .contents_align;
        let s = unsafe { std::slice::from_raw_parts_mut(self.inner, self.len as usize) };
        if (s.len() % size_of::<T>()) == 0 && (s.len() as u64 % buffer_align) == 0 {
            Ok(bytemuck::cast_slice_mut(s))
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }
}

api_type!(WaitHandle, {
    instance: Instance<B>,
    inner: B::Semaphore,
    id: Id<WaitHandle<B>>,
},);
impl<B: hal::Backend> Drop for WaitHandleInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.wait_handles.remove(self.id, None);
            let _ = unsafe {
                instance
                    .inner
                    .destroy_semaphore(std::ptr::read(&self.inner))
            };
        }
    }
}
impl<B: hal::Backend> WaitHandle<B> {
    pub fn wait(&self) -> SupaSimResult<B, ()> {
        let mut s = self.inner_mut()?;
        let _i = s.instance.clone();
        let mut instance = _i.inner_mut()?;
        unsafe { s.inner.wait(&mut instance.inner).map_supasim()? }
        Ok(())
    }
}
struct Event<B: hal::Backend> {
    id: Id<Self>,
    inner: B::Event,
}
