/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use hal::{
    BindGroup as _, Buffer as _, CommandRecorder, Device as _, HalBufferSlice, RecorderSubmitInfo,
    Semaphore, Stream as _,
};
use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::marker::PhantomData;
use std::ops::Deref;
use std::panic::UnwindSafe;
use std::sync::Arc;
use std::time::{Duration, Instant};
use thunderdome::Index;
use types::SyncOperations;

use crate::{
    Buffer, BufferCommand, BufferCommandInner, BufferRange, BufferSlice, CommandRecorderInner,
    Instance, Kernel, MapSupasimError, MappedBuffer, SendableUserBufferAccessClosure, SupaSimError,
    SupaSimResult,
};

/// A semaphore with info about its device that returns to a pool on drop
pub struct DeviceSemaphore<B: hal::Backend> {
    pub inner: Option<B::Semaphore>,
    pub device_index: usize,
    instance: Instance<B>,
    /// Other semaphores (on other devices) that should be signalled when this finishes
    others_to_signal: Vec<Arc<DeviceSemaphore<B>>>,
}
impl<B: hal::Backend> Drop for DeviceSemaphore<B> {
    fn drop(&mut self) {
        // TODO: Return the used semaphore to the instance
    }
}

pub struct SubmissionResources<B: hal::Backend> {
    pub kernels: Vec<crate::Kernel<B>>,
    pub buffers: Vec<crate::Buffer<B>>,
    pub temp_copy_buffer: Option<B::Buffer>,
}
impl<B: hal::Backend> Default for SubmissionResources<B> {
    fn default() -> Self {
        Self {
            kernels: Vec::new(),
            buffers: Vec::new(),
            temp_copy_buffer: None,
        }
    }
}
