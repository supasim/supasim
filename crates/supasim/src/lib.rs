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
            pub struct $name <B: hal::Backend> (Arc<RwLock<Option<[<$name Inner>]<B>>>>);
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
                pub fn destroy(self) -> SupaSimResult<B, ()> {
                    *self.0.write().map_err(|e| SupaSimError::Poison(e.to_string()))? = None;
                    Ok(())
                }
            }
        }
    };
}
#[derive(Error)]
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

pub struct InstanceProperties {}
api_type!(Instance, {
    inner: B::Instance,
},);
api_type!(Kernel, {
    instance: Instance<B>,
    inner: B::Kernel,
},);
api_type!(KernelCache, {
    instance: Instance<B>,
    inner: B::PipelineCache,
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
impl<B: hal::Backend> Drop for KernelCache<B> {
    fn drop(&mut self) {
        if let Ok(mut inner) = self.inner_mut() {
            let instance_clone = inner.instance.clone();
            if let Ok(mut instance) = instance_clone.inner_mut() {
                let _ = instance.inner.destroy_pipeline_cache(std::mem::replace(
                    &mut inner.inner,
                    unsafe {
                        #[allow(clippy::uninit_assumed_init)]
                        std::mem::MaybeUninit::uninit().assume_init()
                    },
                ));
            }
            drop(instance_clone);
        }
    }
}
impl<B: hal::Backend> Kernel<B> {}
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

        let kernel = self
            .inner_mut()?
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
        Ok(Kernel::from_inner(KernelInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner: kernel,
        }))
    }
    pub fn create_kernel_cache(&self, data: &[u8]) -> SupaSimResult<B, KernelCache<B>> {
        let inner = self
            .inner_mut()?
            .inner
            .create_pipeline_cache(data)
            .map_supasim()?;
        Ok(KernelCache::from_inner(KernelCacheInner {
            _phantom: Default::default(),
            instance: self.clone(),
            inner,
        }))
    }
    pub fn wait_for_idle(&self) -> SupaSimResult<B, ()> {
        self.inner_mut()?.inner.wait_for_idle().map_supasim()?;
        Ok(())
    }
    pub fn do_busywork(&self) -> SupaSimResult<B, ()> {
        todo!()
    }
}
