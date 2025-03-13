//! Issues this must handle:
//!
//! * Sharing references/multithreading
//! * Moving buffers in and out of GPU memory when OOM is hit
//! * Synchronization/creation and synchronization of command buffers
//! * Lazy operations
//! * Combine/optimize allocations and creation of things

mod api;

use hal::BackendInstance as _;
use std::{
    marker::PhantomData,
    ops::{Deref, DerefMut},
    sync::{Arc, RwLock, RwLockReadGuard, RwLockWriteGuard},
};
use thiserror::Error;

pub use hal;

/// Contains the index, and a certain "random" value to check if a destroyed thing has been replaced
#[derive(Clone, Copy, Debug)]
pub struct Id(u32, u32);
impl Default for Id {
    fn default() -> Self {
        Self(u32::MAX, 0)
    }
}
pub struct Tracker<T> {
    list: Vec<(u32, T)>,
    unused: Vec<u32>,
    current_identifier: u32,
}
impl<T> Tracker<T> {
    pub fn get(&self, id: Id) -> Option<&T> {
        let value = &self.list[id.0 as usize];
        if value.0 != id.1 {
            return None;
        }
        Some(&value.1)
    }
    pub fn add(&mut self, value: T) -> Id {
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
        Id(idx, identifier)
    }
    pub fn remove(&mut self, id: Id, replace_with: Option<T>) {
        let value = &mut self.list[id.0 as usize];
        if value.0 == id.1 {
            value.0 = u32::MAX;
            if let Some(v) = replace_with {
                value.1 = v;
            }
        }
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

macro_rules! api_type {
    ($name: ident, { $($field:tt)* }, $($attr: meta),*) => {
        paste::paste! {
            // Inner type
            pub(crate) struct [<$name Inner>] <B: hal::Backend> {
                _phantom: PhantomData<B>, // Ensures B is always used
                $($field)*
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
                    Ok(InnerRef(self.0.read().map_err(|e| SupaSimError::Poison(e.to_string()))?))
                }
                pub(crate) fn inner_mut(&self) -> SupaSimResult<B, InnerRefMut<[<$name Inner>]<B>>> {
                    Ok(InnerRefMut(self.0.write().map_err(|e| SupaSimError::Poison(e.to_string()))?))
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
            }
        }
    };
}
#[derive(Error, Debug)]
pub enum SupaSimError<B: hal::Backend> {
    // Rust thinks that B::Error could be SupaSimError. Nevermind that this would be a recursive definition
    HalError(B::Error),
    Poison(String),
    Other(anyhow::Error),
    AlreadyDestroyed,
    DestroyWhileInUse,
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
pub type SupaSimResult<B, T> = Result<T, SupaSimError<B>>;

struct Fence<B: hal::Backend> {
    inner: B::Fence,
    in_use: bool,
}
pub struct InstanceProperties {}
api_type!(Instance, {
    inner: B::Instance,
    kernels: Tracker<Option<Kernel<B>>>,
    kernel_caches: Tracker<Option<KernelCache<B>>>,
    command_recorders: Tracker<Option<CommandRecorder<B>>>,
    buffers: Tracker<Option<Buffer<B>>>,
    wait_handles: Tracker<WaitHandle<B>>,
    fences: Tracker<Fence<B>>,
},);
impl<B: hal::Backend> Instance<B> {
    pub fn properties(&self) -> InstanceProperties {
        InstanceProperties {}
    }
    pub fn compile_kernel(
        &self,
        binary: &[u8],
        reflection: shaders::ShaderReflectionInfo,
        cache: Option<&KernelCache<B>>,
    ) -> SupaSimResult<B, Kernel<B>> {
        let mut cache_lock = if let Some(cache) = cache {
            Some(cache.inner_mut()?)
        } else {
            None
        };
        let mut s = self.inner_mut()?;

        let kernel = s
            .inner
            .compile_kernel(
                binary,
                &reflection,
                if let Some(lock) = cache_lock.as_mut() {
                    Some(&mut lock.inner)
                } else {
                    None
                },
            )
            .map_supasim()?;
        let k = Kernel::from_inner(KernelInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner: kernel,
            id: Default::default(),
        });
        k.inner_mut()?.id = s.kernels.add(Some(k.clone()));
        Ok(k)
    }
    pub fn create_kernel_cache(&self, data: &[u8]) -> SupaSimResult<B, KernelCache<B>> {
        let mut s = self.inner_mut()?;
        let inner = s.inner.create_pipeline_cache(data).map_supasim()?;
        let k = KernelCache::from_inner(KernelCacheInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner,
            id: Default::default(),
        });
        k.inner_mut()?.id = s.kernel_caches.add(Some(k.clone()));
        Ok(k)
    }
    pub fn create_recorder(&self, reusable: bool) -> SupaSimResult<B, CommandRecorder<B>> {
        let mut s = self.inner_mut()?;
        let inner = s.inner.create_recorder(reusable).map_supasim()?;
        let r = CommandRecorder::from_inner(CommandRecorderInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner,
            id: Default::default(),
            current_fence: None,
            used_buffers: Vec::new(),
        });
        r.inner_mut()?.id = s.command_recorders.add(Some(r.clone()));
        Ok(r)
    }
    pub fn create_buffer(&self, desc: &types::BufferDescriptor) -> SupaSimResult<B, Buffer<B>> {
        let mut s = self.inner_mut()?;
        let inner = s.inner.create_buffer(desc).map_supasim()?;
        let b = Buffer::from_inner(BufferInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner,
            id: Default::default(),
            semaphores: Vec::new(),
        });
        b.inner_mut()?.id = s.buffers.add(Some(b.clone()));
        Ok(b)
    }
    pub fn submit_commands(
        &self,
        _recorders: &[CommandRecorder<B>],
    ) -> SupaSimResult<B, CommandRecorder<B>> {
        //let mut s = self.inner_mut()?;
        todo!()
    }
    pub fn wait(
        &self,
        wait_handles: &[WaitHandle<B>],
        wait_for_all: bool,
        timeout: f32,
    ) -> SupaSimResult<B, ()> {
        let mut s = self.inner_mut()?;
        let mut locks = Vec::new();

        for handle in wait_handles {
            locks.push(handle.inner()?);
        }
        let handles: Vec<_> = locks.iter().map(|a| &a.inner).collect();
        s.inner
            .wait_for_semaphores(&handles, wait_for_all, timeout)
            .map_supasim()?;
        Ok(())
    }
    pub fn wait_for_idle(&self, timeout: f32) -> SupaSimResult<B, ()> {
        let mut _s = self.inner_mut()?;
        let s = &mut *_s;
        let fences: Vec<_> = s
            .fences
            .list
            .iter()
            .filter_map(|(_, a)| if a.in_use { Some(&a.inner) } else { None })
            .collect();
        s.inner
            .wait_for_fences(&fences, true, timeout)
            .map_supasim()?;
        Ok(())
    }
    pub fn do_busywork(&self) -> SupaSimResult<B, ()> {
        todo!()
    }
    pub fn clear_cached_resources(&self) -> SupaSimResult<B, ()> {
        let mut s = self.inner_mut()?;
        s.inner.cleanup_cached_resources().map_supasim()?;
        todo!();
        //Ok(())
    }
}
impl<B: hal::Backend> Drop for InstanceInner<B> {
    fn drop(&mut self) {
        let _ = self.inner.wait_for_idle();
        self.command_recorders.list.clear(); // These will call their destructors, politely taking care of themselves
        for (_, f) in std::mem::take(&mut self.fences.list) {
            let _ = self.inner.destroy_fence(f.inner);
        }
        self.wait_handles.list.clear();
        self.kernel_caches.list.clear();
        self.buffers.list.clear();
        self.kernels.list.clear();
    }
}
api_type!(Kernel, {
    instance: Instance<B>,
    inner: B::Kernel,
    id: Id,
},);
impl<B: hal::Backend> Kernel<B> {}
impl<B: hal::Backend> Drop for KernelInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.kernels.remove(self.id, Some(None));
            let _ = instance
                .inner
                .destroy_kernel(std::mem::replace(&mut self.inner, unsafe {
                    #[allow(clippy::uninit_assumed_init)]
                    std::mem::MaybeUninit::uninit().assume_init()
                }));
        }
    }
}
api_type!(KernelCache, {
    instance: Instance<B>,
    inner: B::PipelineCache,
    id: Id,
},);
impl<B: hal::Backend> KernelCache<B> {
    pub fn get_data(self) -> SupaSimResult<B, Vec<u8>> {
        let mut inner = self.inner_mut()?;
        let instance = inner.instance.clone();
        let data = instance
            .inner_mut()?
            .inner
            .get_pipeline_cache_data(&mut inner.inner)
            .map_supasim()?;
        Ok(data)
    }
}
impl<B: hal::Backend> Drop for KernelCacheInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.kernel_caches.remove(self.id, Some(None));
            let _ =
                instance
                    .inner
                    .destroy_pipeline_cache(std::mem::replace(&mut self.inner, unsafe {
                        #[allow(clippy::uninit_assumed_init)]
                        std::mem::MaybeUninit::uninit().assume_init()
                    }));
        }
    }
}
api_type!(CommandRecorder, {
    #[allow(dead_code)]
    instance: Instance<B>,
    inner: B::CommandRecorder,
    id: Id,
    #[allow(dead_code)]
    current_fence: Option<u32>,
    #[allow(dead_code)]
    used_buffers: Vec<Id>,
},);
impl<B: hal::Backend> Drop for CommandRecorderInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.command_recorders.remove(self.id, Some(None));
            let _ = instance
                .inner
                .destroy_recorder(std::mem::replace(&mut self.inner, unsafe {
                    #[allow(clippy::uninit_assumed_init)]
                    std::mem::MaybeUninit::uninit().assume_init()
                }));
        }
    }
}

api_type!(Buffer, {
    #[allow(dead_code)]
    instance: Instance<B>,
    #[allow(dead_code)]
    inner: B::Buffer,
    id: Id,
    #[allow(dead_code)]
    semaphores: Vec<Id>,
},);
impl<B: hal::Backend> Drop for BufferInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.buffers.remove(self.id, Some(None));
            let _ = instance
                .inner
                .destroy_buffer(std::mem::replace(&mut self.inner, unsafe {
                    #[allow(clippy::uninit_assumed_init)]
                    std::mem::MaybeUninit::uninit().assume_init()
                }));
        }
    }
}

api_type!(WaitHandle, {
    #[allow(dead_code)]
    instance: Instance<B>,
    inner: B::Semaphore,
    #[allow(dead_code)]
    id: Id,
},);
impl<B: hal::Backend> Drop for WaitHandleInner<B> {
    fn drop(&mut self) {
        if let Ok(mut instance) = self.instance.clone().inner_mut() {
            instance.wait_handles.remove(self.id, None);
            let _ = instance
                .inner
                .destroy_semaphore(std::mem::replace(&mut self.inner, unsafe {
                    #[allow(clippy::uninit_assumed_init)]
                    std::mem::MaybeUninit::uninit().assume_init()
                }));
        }
    }
}
