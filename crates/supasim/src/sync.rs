/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use std::{collections::HashSet, sync::Arc};

use hal::{BackendInstance, CommandRecorder as _, Semaphore as _, Stream};
use thunderdome::Index;
use types::SyncMode;

use crate::{
    Instance, MapSupasimError, SupaSimError, SupaSimResult, WaitHandle, WaitHandleInner, record,
    sync_thread::GpuSubmissionInfo,
};

/// A semaphore with info about its device that returns to a pool on drop
pub struct Semaphore<B: hal::Backend> {
    pub inner: Option<B::Semaphore>,
    /// The device, stream and submission that will signal it. If None, then the host will signal.
    pub device_stream_submission: Option<(u16, u16, u64)>,
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
        if self.inner.is_some() {
            let s = self.instance.inner().unwrap();
            s.unused_semaphores.lock().push(self.inner.take().unwrap());
        }
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
    let s = instance.inner()?;
    s.check_destroyed()?;
    let submission_idx;
    let semaphore;
    {
        let semaphore_raw = if let Some(s) = s.unused_semaphores.lock().pop() {
            s
        } else {
            unsafe {
                s.hal_instance
                    .read()
                    .as_ref()
                    .unwrap()
                    .create_semaphore()
                    .map_supasim()?
            }
        };
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
            &instance,
            s.hal_instance_properties.sync_mode == SyncMode::VulkanStyle,
            0,
        )?;
        let mut lock = s.hal_devices[0].streams[0].stream_handle.write();
        submission_idx = lock.current_submitted_count;
        lock.current_submitted_count += 1;
        semaphore = Arc::new(Semaphore::<B> {
            inner: Some(semaphore_raw),
            device_stream_submission: Some((0, 0, submission_idx)),
            instance: instance.clone(),
        });
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
                let ood_wait =
                    b_mut
                        .residency
                        .add_gpu_use(range.into(), range.needs_mut, semaphore.clone());
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
        let used_buffers: Vec<_> = used_buffers
            .iter()
            .map(|a| {
                s.buffers
                    .read()
                    .get(**a)
                    .unwrap()
                    .as_ref()
                    .unwrap()
                    .upgrade()
                    .unwrap()
            })
            .collect();
        let kernels = streams.resources.kernels.clone();
        lock.submit(GpuSubmissionInfo {
            index: submission_idx,
            command_recorder: recorder,
            bind_groups,
            used_buffer_ranges,
            used_buffers,
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
