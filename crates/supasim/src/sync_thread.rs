/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use crate::{
    Buffer, Instance, Kernel, SupaSimError, SupaSimResult, buffer::residency::OutOfDateWait,
    sync::SubmissionResources,
};
use parking_lot::{Condvar, Mutex};
use std::sync::{Arc, mpsc::Sender};

pub enum StreamThreadMessage<B: hal::Backend> {
    Submission(GpuSubmissionInfo<B>),
    /// The sync thread should quit as soon as everything is complete.
    ShutDown,
}

pub struct GpuSubmissionInfo<B: hal::Backend> {
    /// Probably not strictly necessary, mainly for sanity purposes
    pub _index: u64,
    pub _command_recorder: B::CommandRecorder,
    /// The bind groups used that may be destroyed upon submission completion
    pub _bind_groups: Vec<(B::BindGroup, Kernel<B>)>,
    /// Ranges that may be freed up for other use upon submission completion
    pub _used_buffer_ranges: Vec<(OutOfDateWait<B>, Buffer<B>)>,
    /// Other resources that might be affected by submission completion
    pub _used_resources: SubmissionResources<B>,
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

    pub fn wait_for_submission(&self, idx: u64) {
        // Panic here because this shouldn't be publicly accessible
        if idx >= self.current_submitted_count {
            panic!("Submission index too high");
        }
        let mut lock = self.current_completed_index.0.lock();
        while *lock < idx {
            self.current_completed_index.1.wait(&mut lock);
        }
    }
}

pub fn create_sync_thread<B: hal::Backend>(
    _instance: Instance<B>,
    _device_idx: usize,
    _stream_idx: usize,
) -> StreamThreadHandle<B> {
    todo!()
}
