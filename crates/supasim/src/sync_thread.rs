/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use std::sync::{Arc, mpsc::Sender};

use parking_lot::{Condvar, Mutex};

use crate::{
    Buffer, Kernel, SupaSimError, SupaSimResult, residency::OutOfDateWait,
    sync::SubmissionResources,
};

enum StreamThreadMessage<B: hal::Backend> {
    Submission(GpuSubmissionInfo<B>),
    ShutDown,
}

pub struct GpuSubmissionInfo<B: hal::Backend> {
    /// Probably not strictly necessary
    pub index: u64,
    pub command_recorder: B::CommandRecorder,
    pub bind_groups: Vec<(B::BindGroup, Kernel<B>)>,
    pub used_buffer_ranges: Vec<(OutOfDateWait<B>, Buffer<B>)>,
    pub used_buffers: Vec<Buffer<B>>,
    pub used_resources: SubmissionResources<B>,
}

pub struct StreamThreadHandle<B: hal::Backend> {
    pub current_submitted_count: u64,
    pub current_completed_index: Arc<(Mutex<u64>, Condvar)>,
    pub sender: Sender<StreamThreadMessage<B>>,
}
impl<B: hal::Backend> StreamThreadHandle<B> {
    pub fn submit(&mut self, submission: GpuSubmissionInfo<B>) -> SupaSimResult<B, ()> {
        self.sender
            .send(StreamThreadMessage::Submission(submission))
            .map_err(|e| SupaSimError::SyncThreadPanic(e.to_string()))?;
        Ok(())
    }
}
