//! Issues this must handle:
//!
//! * Sharing references/multithreading
//! * Moving buffers in and out of GPU memory when OOM is hit
//! * Synchronization/creation and synchronization of command buffers
//! * Lazy operations
//! * Combine/optimize allocations and creation of things

mod api;

use api::BackendInstance;
use std::{
    marker::PhantomData,
    sync::{Arc, RwLock},
};
use thiserror::Error;

pub use hal;

pub struct Backend<B: hal::Backend>(PhantomData<B>);
impl<B: hal::Backend> api::Backend for Backend<B> {}

#[derive(Error)]
pub enum SupaSimError<B: hal::Backend> {
    HalError(B::Error),
    Poison(String),
    Other(#[from] anyhow::Error),
}

pub type SupaSimResult<B, T> = Result<T, SupaSimError<B>>;

pub struct InstanceInner<B: hal::Backend> {
    inner: B::Instance,
}
#[derive(Clone)]
pub struct Instance<B: hal::Backend> {
    inner: Arc<RwLock<InstanceInner<B>>>,
}
impl<B: hal::Backend> Instance<B> {
    pub fn from_inner(inner: B::Instance) -> Self {
        Self {
            inner: Arc::new(RwLock::new(InstanceInner { inner })),
        }
    }
}
