use std::sync::{Arc, RwLock, RwLockWriteGuard};
use thiserror::Error;

pub use hal;

#[derive(Error)]
pub enum SupaSimError<B: hal::Backend> {
    HalError(B::Error),
    Poison(String),
    Other(#[from] anyhow::Error),
}

pub type SupaSimResult<B, T> = Result<T, SupaSimError<B>>;

pub struct InnerRef<'a, T, I: AsMut<T> + 'a>(RwLockWriteGuard<'a, I>, std::marker::PhantomData<T>);
impl<'a, T, I: AsMut<T> + 'a> AsMut<T> for InnerRef<'a, T, I> {
    fn as_mut(&mut self) -> &mut T {
        self.0.as_mut()
    }
}
pub struct InstanceInner<B: hal::Backend> {
    inner: B::Instance,
}
impl<B: hal::Backend> AsMut<B::Instance> for InstanceInner<B> {
    fn as_mut(&mut self) -> &mut B::Instance {
        &mut self.inner
    }
}
pub struct Instance<B: hal::Backend> {
    inner: Arc<RwLock<InstanceInner<B>>>,
}
impl<B: hal::Backend> Instance<B> {
    pub fn as_inner(&self) -> SupaSimResult<B, InnerRef<B::Instance, InstanceInner<B>>> {
        Ok(InnerRef(
            self.inner
                .write()
                .map_err(|e| SupaSimError::Poison(e.to_string()))?,
            Default::default(),
        ))
    }
}
