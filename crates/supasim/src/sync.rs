/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use std::sync::Arc;

use crate::Instance;

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
