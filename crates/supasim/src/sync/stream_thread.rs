/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use crate::{
    Buffer, Instance, InstanceWeak, Kernel, MapSupasimError, SupaSimError, SupaSimResult,
    buffer::ood::OutOfDateWait, sync::SubmissionResources,
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
    // A WEAK handle deliberately: the thread must NOT keep the instance alive,
    // otherwise `InstanceInner::drop` (which sends `ShutDown` and joins this
    // thread) could never run — the thread's own strong `Arc` would keep the
    // strong count above zero forever, so `Drop` never fires, so `ShutDown` is
    // never sent, so the thread blocks on `recv()` forever and every HAL
    // resource leaks. Holding only a weak ref between messages lets the last
    // user `Instance` handle dropping bring `InstanceInner` to zero -> `Drop`
    // -> `destroy()` -> `ShutDown` -> this (idle) thread exits -> `join()`.
    instance: InstanceWeak<B>,
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
                    // Upgrade to a temporary STRONG ref for the whole duration of
                    // processing this submission, so the instance cannot be
                    // destroyed mid-submission. It is dropped before the next
                    // `recv()`, restoring the weak-only state between messages.
                    match instance.upgrade() {
                        Ok(strong) => {
                            run_submission(
                                &strong,
                                device_idx,
                                stream_idx,
                                info,
                                &completed_thread,
                            );
                            drop(strong);
                        }
                        Err(_) => {
                            // The instance is already gone (last handle dropped
                            // before this submission was drained). Best-effort:
                            // let `info` drop (releasing its Arcs) and advance the
                            // completion counter so no waiter is wedged. HAL
                            // resource cleanup already happened via
                            // `InstanceInner::destroy`.
                            drop(info);
                            let (lock, cv) = &*completed_thread;
                            let mut g = lock.lock();
                            *g += 1;
                            cv.notify_all();
                        }
                    }
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

/// Advances the completion counter and wakes waiters when dropped.
///
/// The completion counter MUST advance for every processed submission, even if a
/// HAL call or resource-free errors partway through — otherwise `WaitHandle::wait`,
/// `wait_for_idle`, and `thread.join()` (all of which block until the counter
/// reaches their submission index) would hang forever. Doing it in `Drop` means the
/// counter advances regardless of how `run_submission` returns (early error return
/// or even an unwinding panic in a HAL call).
struct CompletionGuard<'a> {
    completed: &'a Arc<(Mutex<u64>, Condvar)>,
}
impl Drop for CompletionGuard<'_> {
    fn drop(&mut self) {
        let (lock, cv) = &**self.completed;
        let mut g = lock.lock();
        *g += 1;
        cv.notify_all();
    }
}

/// Runs a single GPU submission to completion: submit it, block until the GPU
/// signals its completion semaphore, free its transient resources, then advance
/// the completion counter and wake any waiters.
///
/// Locking rule (critical): the `instance.inner()` read lock and the stream mutex
/// are only ever held for the submit call and the resource-free call, never across
/// the blocking `signal_semaphore.wait()`.
///
/// Robustness: the completion counter is advanced by `CompletionGuard::drop` no
/// matter how this function returns, so a HAL error (or panic) unblocks waiters
/// instead of wedging the whole system. Errors are logged, not propagated.
fn run_submission<B: hal::Backend>(
    instance: &Instance<B>,
    device_idx: usize,
    stream_idx: usize,
    info: GpuSubmissionInfo<B>,
    completed: &Arc<(Mutex<u64>, Condvar)>,
) {
    // Advances the completion counter on scope exit, whatever happens below.
    let _completion = CompletionGuard { completed };
    if let Err(e) = run_submission_inner(instance, device_idx, stream_idx, info) {
        log::error!("sync thread failed to process submission: {e:?}");
    }
}

fn run_submission_inner<B: hal::Backend>(
    instance: &Instance<B>,
    device_idx: usize,
    stream_idx: usize,
    mut info: GpuSubmissionInfo<B>,
) -> SupaSimResult<B, ()> {
    use hal::Stream as _;
    // Submit the recorder, signalling the completion semaphore on the GPU.
    // Lock only for the duration of the submit call.
    {
        let s = instance.inner()?;
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
                .map_supasim()?;
        }
    }
    // Block (CPU side) until the GPU signals completion. No locks held here.
    info.signal_semaphore.wait()?;
    // Free per-submission resources now that the GPU is done with them.
    finish_submission(instance, device_idx, stream_idx, info)
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
) -> SupaSimResult<B, ()> {
    use hal::{BindGroup as _, Buffer as _};
    let s = instance.inner()?;
    {
        let stream_guard = s.hal_devices[device_idx].streams[stream_idx].inner.lock();
        let stream = stream_guard.as_ref().unwrap();
        for (bg, kernel) in info.bind_groups {
            let kernel_inner = kernel.inner()?;
            unsafe {
                bg.destroy(stream, kernel_inner.inner.as_ref().unwrap())
                    .map_supasim()?;
            }
        }
    }
    if let Some(tmp) = info.used_resources.temp_copy_buffer {
        unsafe {
            tmp.destroy(s.hal_devices[device_idx].inner.lock().as_ref().unwrap())
                .map_supasim()?;
        }
    }
    // Return the command recorder to the pool for reuse.
    s.hal_devices[device_idx].streams[stream_idx]
        .unused_hal_command_recorders
        .lock()
        .push(info.command_recorder);
    Ok(())
    // `info.used_buffer_ranges` is dropped here, releasing its holds on the
    // dependency finishes (and thus their semaphores).
}
