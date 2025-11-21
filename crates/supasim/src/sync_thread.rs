/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use std::sync::{Arc, mpsc::Sender};

use parking_lot::{Condvar, Mutex};

use crate::{
    Buffer, Instance, Kernel, SupaSimError, SupaSimResult, residency::OutOfDateWait,
    sync::SubmissionResources,
};

pub enum StreamThreadMessage<B: hal::Backend> {
    Submission(GpuSubmissionInfo<B>),
    /// The sync thread should quit as soon as everything is complete.
    ShutDown,
}

pub struct GpuSubmissionInfo<B: hal::Backend> {
    /// Probably not strictly necessary, mainly for sanity purposes
    pub index: u64,
    pub command_recorder: B::CommandRecorder,
    /// The bind groups used that may be destroyed upon submission completion
    pub bind_groups: Vec<(B::BindGroup, Kernel<B>)>,
    /// Ranges that may be freed up for other use upon submission completion
    pub used_buffer_ranges: Vec<(OutOfDateWait<B>, Buffer<B>)>,
    /// Other resources that might be affected by submission completion
    pub used_resources: SubmissionResources<B>,
}

pub struct StreamThreadHandle<B: hal::Backend> {
    /// Begins at 1
    pub current_submitted_count: u64,
    /// Begins at 0
    pub current_completed_index: Arc<(Mutex<u64>, Condvar)>,
    pub sender: Sender<StreamThreadMessage<B>>,
    pub thread: std::thread::JoinHandle<()>,
}
impl<B: hal::Backend> StreamThreadHandle<B> {
    pub fn submit(&mut self, submission: GpuSubmissionInfo<B>) -> SupaSimResult<B, ()> {
        self.sender
            .send(StreamThreadMessage::Submission(submission))
            .map_err(|e| SupaSimError::SyncThreadPanic(e.to_string()))?;
        Ok(())
    }
}

pub fn create_sync_thread<B: hal::Backend>(
    _instance: Instance<B>,
    _device_idx: usize,
    _stream_idx: usize,
) -> StreamThreadHandle<B> {
    todo!()
}
