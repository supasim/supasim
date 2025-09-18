/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use std::sync::Arc;

use hal::Semaphore as _;
use smallvec::SmallVec;

use crate::{DEVICE_SMALLVEC_SIZE, Instance, MapSupasimError, SupaSimResult};

/// A semaphore with info about its device that returns to a pool on drop
pub struct DeviceSemaphore<B: hal::Backend> {
    pub inner: Option<B::Semaphore>,
    /// The device, stream and submission that will signal it. If None, then the host will signal.
    pub device_stream_submission: Option<(usize, usize, usize)>,
    pub submission: usize,
    instance: Instance<B>,
    /// Other semaphores (on other devices) that should be signalled when this finishes
    to_signal_per_device: SmallVec<[Option<Arc<DeviceSemaphore<B>>>; DEVICE_SMALLVEC_SIZE]>,
}
impl<B: hal::Backend> DeviceSemaphore<B> {
    pub fn wait(&self) -> SupaSimResult<B, ()> {
        unsafe {
            self.inner
                .as_ref()
                .unwrap()
                .wait(self.instance.inner()?.instance.lock().as_ref().unwrap())
                .map_supasim()
        }
    }
    pub fn is_signalled(&self) -> SupaSimResult<B, bool> {
        unsafe {
            self.inner
                .as_ref()
                .unwrap()
                .is_signalled(self.instance.inner()?.instance.lock().as_ref().unwrap())
                .map_supasim()
        }
    }
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
