/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! # Canonical lock ordering (issue #120)
//!
//! SupaSim uses `parking_lot` `RwLock`/`Mutex` liberally and none of them is
//! reentrant. To avoid deadlock, any code path that holds more than one lock at a
//! time MUST acquire them in the following top-to-bottom order, and release them in
//! the reverse order. This is the order the code actually takes after the Task 7
//! audit; keep it in sync if the locking changes.
//!
//! 1. `Instance::inner()` — the outer `Instance<B>` RwLock (`InstanceInner`).
//!    Almost always taken as a **read** guard and held for the whole operation.
//!    The only **write** lockers are `Instance::destroy` and
//!    `clear_cached_resources`; because those need exclusive access, no other
//!    operation may be in flight when they run.
//! 2. Instance-owned arenas: `instance.{buffers,kernels,wait_handles,
//!    command_recorders}.{read,write}()`. Short-lived; used to look up / register
//!    weak handles.
//! 3. Per-object outer guards, taken in a fixed order by object type
//!    (alphabetical): `Buffer::inner()`, `CommandRecorder::inner_mut()`,
//!    `Kernel::inner()`, `WaitHandle::inner()`. When several buffers are locked at
//!    once (`access_buffers`, `record::*`) they are collected first, then locked.
//! 4. `buffer.residency.0.{read,write}()` — the per-buffer `BufferResidency`.
//! 5. `stream.stream_handle.{read,write}()` — the per-stream `StreamThreadHandle`.
//! 6. Innermost, always brief and never held across anything blocking:
//!    `hal_instance.{read,write}()`, `device.inner.lock()`,
//!    `stream.inner.lock()`, `unused_hal_command_recorders.lock()`,
//!    `unused_semaphores.lock()`, and the per-access `BufferAccessFinish`
//!    `is_complete` / `device_semaphore` mutexes.
//!
//! ## The one hard rule: no lock across a blocking semaphore wait
//!
//! A CPU-side wait on a GPU completion semaphore (`Semaphore::wait`,
//! `BufferAccessFinish::wait_host`, `StreamThreadHandle::wait_for_submission`) may
//! block for an unbounded time until the GPU signals. **No residency write lock, no
//! instance write lock, and no stream/device mutex may be held across such a wait.**
//! The blocking waiter itself may need `instance.inner()` (a *read* lock) — e.g. the
//! sync thread's `run_submission` and the residency `ensure_*_current` copies re-take
//! a read lock only for the short submit call, drop it, then block. Holding a write
//! lock across the wait would starve that reader and deadlock.
//!
//! ### Known tolerated residuals (tracked for #120)
//!
//! These hold an instance/buffer **read** guard (never a write guard) across a
//! blocking wait. They are not deadlocks today (no concurrent writer of that lock
//! runs while a buffer access / submission is live), but they widen the lock scope
//! more than strictly necessary:
//!  * `Buffer::{read,write,access}` (`buffer/access.rs`) hold a hoisted `instance`
//!    read guard across `wait_for_cpu_access`.
//!  * `submit_command_recorders`' `ensure_device_current` pass holds a buffer read
//!    guard (and the instance read guard `s`) across the blocking host->device copy.
//!
//! Fully removing them requires threading the borrowed resources out of the guards,
//! a larger refactor left to #120.

pub mod stream_thread;

use crate::{
    Instance, MapSupasimError, SupaSimError, SupaSimResult, WaitHandle, WaitHandleInner, record,
    sync::stream_thread::GpuSubmissionInfo,
};
use hal::{CommandRecorder as _, Semaphore as _, Stream};
use parking_lot::RwLock;
use std::{collections::HashSet, sync::Arc};
use thunderdome::Index;
use types::SyncMode;

/// A semaphore with info about its device that returns to a pool on drop
pub struct Semaphore<B: hal::Backend> {
    /// This will be Some until the semaphore is destroyed.
    pub inner: Option<RwLock<B::Semaphore>>,
    // TODO: should this just be bool for whether its GPU?
    /// The device, stream and submission that will signal it. If None, then the host will signal.
    pub device_stream_submission: Option<(u16, u16, u64)>,
    pub instance: Instance<B>,
}

impl<B: hal::Backend> Semaphore<B> {
    pub fn wait(&self) -> SupaSimResult<B, ()> {
        assert!(self.device_stream_submission.is_some());
        unsafe {
            self.inner
                .as_ref()
                .unwrap()
                .read()
                .wait(self.instance.inner()?.hal_instance.read().as_ref().unwrap())
                .map_supasim()
        }
    }

    pub fn is_signalled(&self) -> SupaSimResult<B, bool> {
        unsafe {
            self.inner
                .as_ref()
                .unwrap()
                .read()
                .is_signalled(self.instance.inner()?.hal_instance.read().as_ref().unwrap())
                .map_supasim()
        }
    }

    pub fn signal(&self) -> SupaSimResult<B, ()> {
        assert!(self.device_stream_submission.is_none());
        unsafe {
            self.inner
                .as_ref()
                .unwrap()
                .write()
                .signal(self.instance.inner()?.hal_instance.read().as_ref().unwrap())
                .map_supasim()
        }
    }
}

// Safety: mirrors the `api_type!` wrappers (`Instance`/`Buffer`/`Kernel`), which
// blanket-impl Send + Sync. `B::Semaphore` is `Send` (see the `hal::Semaphore`
// trait bound) and is only ever a plain GPU handle whose wait/signal is routed
// through the instance under a lock; the sync thread deliberately moves and shares
// `Arc<Semaphore<B>>` across threads.
unsafe impl<B: hal::Backend> Send for Semaphore<B> {}
unsafe impl<B: hal::Backend> Sync for Semaphore<B> {}

impl<B: hal::Backend> Drop for Semaphore<B> {
    fn drop(&mut self) {
        if self.inner.is_some() {
            let s = self.instance.inner().unwrap();
            s.unused_semaphores
                .lock()
                .push(self.inner.take().unwrap().into_inner());
        }
    }
}

/// Resources used in a GPU submission that might need to be
/// destroyed or something when the submission completes.
pub struct SubmissionResources<B: hal::Backend> {
    pub kernels: Vec<crate::Kernel<B>>,
    pub buffers: Vec<crate::Buffer<B>>,
    /// A buffer that is purely used for sugary data uploading, and is definitionally unused
    /// anywhere else. Can always be destroyed immediately upon submission completion.
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

pub fn submit_command_recorders<B: hal::Backend>(
    instance: &Instance<B>,
    recorders: &[crate::CommandRecorder<B>],
) -> SupaSimResult<B, WaitHandle<B>> {
    let s = instance.inner()?;
    s.check_destroyed()?;
    let submission_idx;
    let semaphore;
    {
        let mut semaphore_raw = s.get_semaphore()?;
        // Reset the (possibly pooled) timeline semaphore so its CPU-side wait/signal
        // target advances past any value it already reached in a previous life. A
        // semaphore returned to the pool by a completed submission (e.g. the
        // device<->host copies in `BufferResidency`) still has its GPU counter at the
        // last signalled value; reusing it without `reset` would make the next
        // submission's signal a no-op and its wait return immediately or hang.
        unsafe {
            semaphore_raw
                .reset(s.hal_instance.read().as_ref().unwrap())
                .map_supasim()?;
        }
        let mut recorder_locks = Vec::new();
        for r in recorders.iter() {
            r.check_destroyed()?;
            recorder_locks.push(r.inner_mut()?);
        }
        let mut recorder_inners = Vec::new();
        for r in &mut recorder_locks {
            recorder_inners.push(&mut **r);
        }

        let mut recorder = if let Some(mut r) = s.hal_devices[0].streams[0]
            .unused_hal_command_recorders
            .lock()
            .pop()
        {
            unsafe {
                r.clear(s.hal_devices[0].streams[0].inner.lock().as_ref().unwrap())
                    .map_supasim()?;
            }
            r
        } else {
            unsafe {
                s.hal_devices[0].streams[0]
                    .inner
                    .lock()
                    .as_ref()
                    .unwrap()
                    .create_recorder()
            }
            .map_supasim()?
        };
        let mut used_buffers = HashSet::new();
        let mut used_buffer_ranges = Vec::new();
        let streams = record::assemble_streams(
            &mut recorder_inners,
            instance,
            s.hal_instance_properties.sync_mode == SyncMode::VulkanStyle,
            0,
        )?;
        let mut lock = s.hal_devices[0].streams[0]
            .stream_handle
            .as_ref()
            .unwrap()
            .write();
        submission_idx = lock.current_submitted_count;
        lock.current_submitted_count += 1;
        semaphore = Arc::new(Semaphore::<B> {
            inner: Some(RwLock::new(semaphore_raw)),
            device_stream_submission: Some((0, 0, submission_idx)),
            instance: instance.clone(),
        });
        // WRITE-side residency pass: before the GPU submission consumes any buffer
        // range, make device[0] current for it. If the range's current data was last
        // written on the host (e.g. via `Buffer::write`), the device copy is stale;
        // `ensure_device_current` records a host->device copy, submits it, waits with
        // all residency locks dropped, and marks device[0]'s tracker current.
        //
        // This runs in a pass that is NOT inside the `add_gpu_use` residency-write
        // section below: `ensure_device_current` owns its own locking (it takes a
        // brief residency read lock, drops it before the blocking GPU wait, then a
        // brief write lock to mark current). Nesting it inside the `add_gpu_use`
        // write lock would deadlock. After this pass, the `add_gpu_use` call for the
        // same range sees device[0] current and does not re-schedule a copy.
        for (buf_id, ranges) in &streams.used_ranges {
            let b = s
                .buffers
                .read()
                .get(*buf_id)
                .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                .as_ref()
                .unwrap()
                .upgrade()?;
            // NOTE (residual, tracked for #120): `b_inner` is the buffer's *read* guard
            // and is held across `ensure_device_current`, which blocks on a GPU copy
            // internally (it drops all *residency* locks before that blocking wait, but
            // this outer buffer read guard stays alive). This is tolerated, not a
            // deadlock: `ensure_device_current` never re-locks this buffer's `inner`
            // and never takes a buffer *write* guard, and buffer write guards only come
            // from create/destroy paths. Tightening (borrowing the `BufferResidencyRef`
            // out of the guard so the guard can drop before the wait) needs a larger
            // refactor; left as-is per the Task 7 lock-audit scope.
            let b_inner = b.inner()?;
            for &range in ranges {
                b_inner.residency.ensure_device_current(0, range.range, &s);
            }
        }
        for (buf_id, ranges) in &streams.used_ranges {
            let b = s
                .buffers
                .read()
                .get(*buf_id)
                .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                .as_ref()
                .unwrap()
                .upgrade()?;
            let _b = b.clone();
            let b_mut = _b.inner()?;
            for &range in ranges {
                let ood_wait = b_mut.residency.0.write().add_gpu_use(
                    range.range,
                    range.needs_mut,
                    semaphore.clone(),
                    0,
                    &s,
                );
                used_buffer_ranges.push((ood_wait, b.clone()))
            }
            used_buffers.insert(buf_id);
        }
        let bind_groups = record::record_command_streams(
            &streams,
            instance.clone(),
            &mut recorder,
            &streams.resources.temp_copy_buffer,
            0,
            0,
        )?;
        let kernels = streams.resources.kernels.clone();
        lock.submit(GpuSubmissionInfo {
            index: submission_idx,
            command_recorder: recorder,
            signal_semaphore: semaphore.clone(),
            bind_groups,
            used_buffer_ranges,
            used_resources: streams.resources,
        })?;

        for _kernel in &kernels {
            // Update the kernel's last usage
        }
        for (buf_id, _) in streams.used_ranges {
            let _b = s
                .buffers
                .read()
                .get(buf_id)
                .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                .as_ref()
                .unwrap()
                .upgrade()?;
            // Update the buffer's last usage
        }
    }
    for recorder in recorders {
        recorder.inner_mut()?.destroy(&s);
    }
    Ok(WaitHandle::from_inner(WaitHandleInner {
        _phantom: Default::default(),
        instance: instance.clone(),
        id: Index::DANGLING,
        is_alive: true,
        semaphore: semaphore.clone(),
    }))
}
