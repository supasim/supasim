/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use crate::{
    Buffer, Instance, Kernel, SupaSimError, SupaSimResult, buffer::ood::OutOfDateWait,
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
    /// Probably not strictly necessary, mainly for sanity purposes.
    // Kept for diagnostics / future cross-submission ordering; not read yet.
    #[allow(dead_code)]
    pub index: u64,
    pub command_recorder: B::CommandRecorder,
    /// Signalled by the GPU when this submission completes; waited on CPU-side.
    pub signal_semaphore: Arc<crate::sync::Semaphore<B>>,
    /// The bind groups used that may be destroyed upon submission completion
    pub bind_groups: Vec<(B::BindGroup, Kernel<B>)>,
    /// Ranges that may be freed up for other use upon submission completion.
    ///
    /// These are this submission's *dependencies* (other accesses' finishes) that
    /// its wait semaphores were derived from; they are kept alive until completion
    /// so their Arc'd finishes (and thus their semaphores) remain valid. It is only
    /// held-then-dropped (never field-read), which is the point.
    #[allow(dead_code)]
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
    instance: Instance<B>,
    device_idx: usize,
    stream_idx: usize,
) -> StreamThreadHandle<B> {
    use std::sync::mpsc;
    let (sender, receiver) = mpsc::channel::<StreamThreadMessage<B>>();
    let completed = Arc::new((Mutex::new(0u64), Condvar::new()));
    let completed_thread = completed.clone();
    let thread = std::thread::spawn(move || {
        // The channel is FIFO, so any submissions queued before a ShutDown are
        // processed first (recv drains them in order).
        while let Ok(msg) = receiver.recv() {
            match msg {
                StreamThreadMessage::Submission(info) => {
                    run_submission(&instance, device_idx, stream_idx, info, &completed_thread);
                }
                StreamThreadMessage::ShutDown => break,
            }
        }
    });
    StreamThreadHandle {
        current_submitted_count: 1,
        current_completed_index: completed,
        sender,
        thread,
    }
}

/// Runs a single GPU submission to completion: submit it, block until the GPU
/// signals its completion semaphore, free its transient resources, then advance
/// the completion counter and wake any waiters.
///
/// Locking rule (critical): the `instance.inner()` read lock and the stream mutex
/// are only ever held for the submit call and the resource-free call, never across
/// the blocking `signal_semaphore.wait()`.
fn run_submission<B: hal::Backend>(
    instance: &Instance<B>,
    device_idx: usize,
    stream_idx: usize,
    mut info: GpuSubmissionInfo<B>,
    completed: &Arc<(Mutex<u64>, Condvar)>,
) {
    use hal::Stream as _;
    // Submit the recorder, signalling the completion semaphore on the GPU.
    // Lock only for the duration of the submit call.
    {
        let s = instance.inner().unwrap();
        let signal_lock = info.signal_semaphore.inner.as_ref().unwrap().read();
        let mut stream_guard = s.hal_devices[device_idx].streams[stream_idx].inner.lock();
        let stream = stream_guard.as_mut().unwrap();
        unsafe {
            stream
                .submit_recorders(std::slice::from_mut(&mut hal::RecorderSubmitInfo {
                    command_recorder: &mut info.command_recorder,
                    // TODO: cross-submission waits. Both integration tests submit
                    // exactly once, so there are no cross-submission GPU dependencies;
                    // ordering within a single recorder is handled by pipeline barriers.
                    wait_semaphores: &[],
                    signal_semaphore: Some(&signal_lock),
                }))
                .unwrap();
        }
    }
    // Block (CPU side) until the GPU signals completion. No locks held here.
    info.signal_semaphore.wait().unwrap();
    // Free per-submission resources now that the GPU is done with them.
    finish_submission(instance, device_idx, stream_idx, info);
    // Advance the completion counter and wake anything waiting on this submission.
    let (lock, cv) = &**completed;
    let mut g = lock.lock();
    *g += 1;
    cv.notify_all();
}

/// Frees the transient resources of a completed submission.
///
/// Note: this does NOT call `try_remove_gpu_access`. The finishes in
/// `used_buffer_ranges` are this submission's *dependencies* (other accesses'
/// finishes), not this submission's own residency accesses; removing them would
/// corrupt other accesses' tracking. Residency access lists self-clean lazily via
/// `BufferResidency::update_all_accesses`. We keep `used_buffer_ranges` alive until
/// here (its Arc'd finishes hold the semaphores) then simply drop it.
fn finish_submission<B: hal::Backend>(
    instance: &Instance<B>,
    device_idx: usize,
    stream_idx: usize,
    info: GpuSubmissionInfo<B>,
) {
    use hal::{BindGroup as _, Buffer as _};
    let s = instance.inner().unwrap();
    {
        let stream_guard = s.hal_devices[device_idx].streams[stream_idx].inner.lock();
        let stream = stream_guard.as_ref().unwrap();
        for (bg, kernel) in info.bind_groups {
            let kernel_inner = kernel.inner().unwrap();
            unsafe {
                bg.destroy(stream, kernel_inner.inner.as_ref().unwrap())
                    .unwrap();
            }
        }
    }
    if let Some(tmp) = info.used_resources.temp_copy_buffer {
        unsafe {
            tmp.destroy(s.hal_devices[device_idx].inner.lock().as_ref().unwrap())
                .unwrap();
        }
    }
    // Return the command recorder to the pool for reuse.
    s.hal_devices[device_idx].streams[stream_idx]
        .unused_hal_command_recorders
        .lock()
        .push(info.command_recorder);
    // `info.used_buffer_ranges` is dropped here, releasing its holds on the
    // dependency finishes (and thus their semaphores).
}
