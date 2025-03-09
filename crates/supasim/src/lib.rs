//! Issues this must handle:
//!
//! * Sharing references/multithreading
//! * Moving buffers in and out of GPU memory when OOM is hit
//! * Synchronization/creation and synchronization of command buffers
//! * Lazy operations
//! * Combine/optimize allocations and creation of things

mod api;

use api::BackendInstance;
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
                pub(crate) fn from_inner(&self, inner: [<$name Inner>]<B>) -> Self {
                    Self(Arc::new(RwLock::new(Some(inner))))
                }
                pub(crate) fn inner(&self) -> SupaSimResult<B, InnerRef<[<$name Inner>]<B>>> {
                    Ok(InnerRef(self.0.read().map_err(|e| SupaSimError::Poison(e.to_string()))?))
                }
                pub(crate) fn inner_mut(&self) -> SupaSimResult<B, InnerRefMut<[<$name Inner>]<B>>> {
                    Ok(InnerRefMut(self.0.write().map_err(|e| SupaSimError::Poison(e.to_string()))?))
                }
            }
        }
    };
}
#[derive(Error)]
pub enum SupaSimError<B: hal::Backend> {
    HalError(B::Error),
    Poison(String),
    Other(#[from] anyhow::Error),
    AlreadyDestroyed,
    DestroyWhileInUse,
}
pub type SupaSimResult<B, T> = Result<T, SupaSimError<B>>;

pub struct InstanceProperties {}
api_type!(Instance, {
    inner: B::Instance,
},);
api_type!(Kernel, {
    inner: B::Kernel,
},);
api_type!(KernelCache, {
    inner: B::PipelineCache,
},);
impl<B: hal::Backend> Kernel<B> {
    pub fn destroy(self) -> SupaSimResult<B, ()> {
        //let a = Abc::clone();
        //let res = a.0.read().unwrap();
        todo!()
    }
}
impl<B: hal::Backend> Instance<B> {
    pub fn properties(&self) -> InstanceProperties {
        InstanceProperties {}
    }
    pub fn compile_kernel(
        &self,
        binary: &[u8],
        reflection: shaders::ShaderReflectionInfo,
        cache: Option<KernelCache<B>>,
    ) -> SupaSimResult<B, Kernel<B>> {
        let cache = if let Some(cache) = cache {
            Some(cache.inner_mut()?)
        } else {
            None
        };
        let cache_ref = if let Some(mut l) = cache {
            Some(&mut l.inner)
        } else {
            None
        };
        let kernel = self
            .inner_mut()?
            .inner
            .compile_kernel(binary, &reflection, cache_ref);
        todo!()
    }
}
