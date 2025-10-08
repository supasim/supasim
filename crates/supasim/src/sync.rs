/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use hal::Semaphore as _;

use crate::{Instance, MapSupasimError, SupaSimResult, WaitHandle, record};

/// A semaphore with info about its device that returns to a pool on drop
pub struct Semaphore<B: hal::Backend> {
    pub inner: Option<B::Semaphore>,
    /// The device, stream and submission that will signal it. If None, then the host will signal.
    pub device_stream_submission: Option<(usize, usize, usize)>,
    instance: Instance<B>,
}
impl<B: hal::Backend> Semaphore<B> {
    pub fn wait(&self) -> SupaSimResult<B, ()> {
        unsafe {
            self.inner
                .as_ref()
                .unwrap()
                .wait(self.instance.inner()?.hal_instance.read().as_ref().unwrap())
                .map_supasim()
        }
    }
    pub fn is_signalled(&self) -> SupaSimResult<B, bool> {
        unsafe {
            self.inner
                .as_ref()
                .unwrap()
                .is_signalled(self.instance.inner()?.hal_instance.read().as_ref().unwrap())
                .map_supasim()
        }
    }
}
impl<B: hal::Backend> Drop for Semaphore<B> {
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

pub fn submit_command_recorders<B: hal::Backend>(
    instance: &Instance<B>,
    recorders: &[crate::CommandRecorder<B>],
) -> SupaSimResult<B, WaitHandle<B>> {
    instance.check_destroyed()?;
    let submission_idx;
    {
        let s = instance.inner()?;

        let mut recorder_locks = Vec::new();
        for r in recorders.iter_mut() {
            r.check_destroyed()?;
            recorder_locks.push(r.inner_mut()?);
        }
        let mut recorder_inners = Vec::new();
        for r in &mut recorder_locks {
            recorder_inners.push(&mut **r);
        }

        let mut recorder = if let Some(mut r) = s.hal_command_recorders.write().pop() {
            unsafe {
                r.clear(s.stream.lock().as_mut().unwrap()).map_supasim()?;
            }
            r
        } else {
            unsafe { s.stream.lock().as_mut().unwrap().create_recorder() }.map_supasim()?
        };
        let mut used_buffers = HashSet::new();
        let mut used_buffer_ranges = Vec::new();
        let streams = record::assemble_streams(
            &mut recorder_inners,
            &s,
            s.hal_instance_properties.sync_mode == SyncMode::VulkanStyle,
        )?;
        for (buf_id, ranges) in &streams.used_ranges {
            let b = s
                .buffers
                .lock()
                .get(*buf_id)
                .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                .as_ref()
                .unwrap()
                .upgrade()?;
            let _b = b.clone();
            let b_mut = _b.inner()?;
            for range in ranges {
                let id = b_mut.slice_tracker.acquire(
                    &s,
                    *range,
                    if b_mut.slice_tracker.mutex.lock().gpu_available {
                        BufferUser::Gpu(u64::MAX)
                    } else {
                        BufferUser::Cross(u64::MAX)
                    },
                    false,
                )?;
                used_buffer_ranges.push((id, b.clone()))
            }
            used_buffers.insert(buf_id);
        }
        drop(s);
        let bind_groups = sync::record_command_streams(
            &streams,
            self.clone(),
            &mut recorder,
            &streams.resources.temp_copy_buffer,
        )?;
        let s = self.inner()?;
        let used_buffers: Vec<_> = used_buffers
            .iter()
            .map(|a| {
                s.buffers
                    .lock()
                    .get(**a)
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .upgrade()
                    .unwrap()
            })
            .collect();
        let kernels = streams.resources.kernels.clone();
        submission_idx = s.sync_thread().submit_gpu(GpuSubmissionInfo {
            command_recorders: vec![recorder],
            bind_groups,
            used_buffer_ranges: used_buffer_ranges.clone(),
            used_buffers,
            used_resources: streams.resources,
        })?;

        for kernel in &kernels {
            kernel.inner_mut()?.last_used = submission_idx;
        }
        for (buf_id, _) in streams.used_ranges {
            let b = s
                .buffers
                .lock()
                .get(buf_id)
                .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                .as_ref()
                .unwrap()
                .upgrade()?;
            let mut b_mut = b.inner_mut()?;
            b_mut.last_used = submission_idx;
        }
        for (id, b) in used_buffer_ranges {
            b.inner()?
                .slice_tracker
                .update_user_submission(id, submission_idx, &s);
        }
    }

    for recorder in recorders {
        recorder._destroy()?;
    }
    Ok(WaitHandle::from_inner(WaitHandleInner {
        _phantom: Default::default(),
        _is_destroyed: false,
        instance: self.clone(),
        index: submission_idx,
        id: Index::DANGLING,
        is_alive: true,
    }))
}
