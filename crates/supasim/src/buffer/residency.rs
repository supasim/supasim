/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use crate::{
    DEVICE_SMALLVEC_SIZE, InstanceInner, MapSupasimError, SupaSimResult, buffer::BufferAccessRange,
    sync::Semaphore,
};
use hal::Buffer;
use memmap2::{MmapMut, MmapOptions};
use parking_lot::{Condvar, Mutex, RwLock};
use smallvec::SmallVec;
use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    sync::Arc,
};

pub struct OutOfDateWait<B: hal::Backend> {
    semaphores: Vec<Arc<Semaphore<B>>>,
    other_copy_range: Option<BufferAccessRange>,
}

#[derive(Default)]
struct OutOfDateTracker<B: hal::Backend> {
    /// Sorted by range start. Ranges that will be out of date until their copy completes or is started.
    ///
    /// This represents ranges that are out of date after copies and operations finish.
    out_of_date_ranges: Vec<BufferAccessRange>,
    /// Copies that will make ranges valid when complete
    current_copies: Vec<Arc<BufferAccessFinish<B>>>,
}

impl<B: hal::Backend> OutOfDateTracker<B> {
    pub fn uninit(length: u64) -> Self {
        Self {
            out_of_date_ranges: vec![BufferAccessRange { start: 0, length }],
            current_copies: Vec::new(),
        }
    }

    /// Mark range as up to date
    pub fn update_range_immediate(&mut self, _range: BufferAccessRange) {
        todo!()
    }

    /// Mark range as not up to date
    pub fn invalidate_range(&mut self, _range: BufferAccessRange) {
        todo!()
    }

    /// Mark range as up to date when something finishes
    pub fn update_range_delayed(&mut self, _finish: BufferAccessFinish<B>) {
        todo!()
    }

    /// Returns what needs to be waited for and what needs to be copied
    pub fn get_needed_waits(&mut self, _range: BufferAccessRange) -> OutOfDateWait<B> {
        todo!()
    }

    /// Applies updates from copies that have completed
    pub fn check_all_current_copies(&mut self) {
        for i in (0..self.current_copies.len()).rev() {
            if self.current_copies[i].is_complete_host() {
                self.current_copies.remove(i);
            }
        }
    }
}

/// Handles all residency and synchronizatino for a buffer
pub struct DeviceResidencyState<B: hal::Backend> {
    /// The backing memory
    pub buffer: Option<B::Buffer>,
    ood_tracker: OutOfDateTracker<B>,
}

impl<B: hal::Backend> Default for DeviceResidencyState<B> {
    fn default() -> Self {
        Self {
            buffer: None,
            ood_tracker: OutOfDateTracker {
                out_of_date_ranges: vec![],
                current_copies: vec![],
            },
        }
    }
}
impl<B: hal::Backend> DeviceResidencyState<B> {
    fn destroy(self, instance: &InstanceInner<B>, device_idx: u32) -> SupaSimResult<B, ()> {
        unsafe {
            if let Some(b) = self.buffer {
                b.destroy(
                    instance.hal_devices[device_idx as usize]
                        .inner
                        .lock()
                        .as_ref()
                        .unwrap(),
                )
                .map_supasim()
            } else {
                Ok(())
            }
        }
    }
}

/// Stored in the file system
pub struct StorageResidencyState<B: hal::Backend> {
    /// The temporary file, it should be dropped last
    file: File,
    /// The file used to store the buffer's contents
    mapped_file: MmapMut,
    /// The tracker for out of date ranges for this copy of the data
    ood_tracker: OutOfDateTracker<B>,
}
impl<B: hal::Backend> StorageResidencyState<B> {
    pub fn new(len: u64) -> Self {
        let file = tempfile::tempfile().unwrap();
        file.set_len(len).unwrap();

        let mapped_file = unsafe { MmapOptions::new().map_mut(&file).unwrap() };
        Self {
            file,
            mapped_file,
            ood_tracker: OutOfDateTracker::uninit(len),
        }
    }

    /// Technically unnecessary and redundant
    pub fn destroy(self) {
        drop(self.mapped_file);
        drop(self.file);
    }
}

/// Contains data for waiting on a buffer access
pub struct BufferAccessFinish<B: hal::Backend> {
    /// The conditional variable, only used in CPU->CPU synchronization
    condvar: Option<Condvar>,
    /// This is set by the CPU before ringing the condvar. This is set when
    /// the work is first observed to be done, including for GPU work, to
    /// avoid unnecessary underlying semaphore operations.
    is_complete: Mutex<bool>,
    /// Always some for GPU work. CPU work will not set this itself, but
    /// GPU work may come later and set this such that the CPU will signal
    /// it when finished.
    device_semaphore: Option<Arc<Semaphore<B>>>,
    /// The range that is being accessed
    range: BufferAccessRange,
    /// The ID used to look this up in a more efficient hashmap
    id: u64,
}

impl<B: hal::Backend> BufferAccessFinish<B> {
    pub fn is_complete_host(&self) -> bool {
        if self.condvar.is_some() {
            let lock = self.is_complete.lock();
            *lock
        } else if let Some(s) = self.device_semaphore.as_ref() {
            let lock = self.is_complete.lock();
            if *lock {
                return true;
            }
            assert!(s.device_stream_submission.is_some());
            let res = s.is_signalled().unwrap();
            if res {
                *self.is_complete.lock() = true;
            }
            res
        } else {
            unreachable!()
        }
    }
    pub fn wait_host(&self) {
        if let Some(cv) = self.condvar.as_ref() {
            let mut lock = self.is_complete.lock();
            loop {
                if *lock {
                    return;
                }
                cv.wait(&mut lock);
            }
        } else if let Some(s) = self.device_semaphore.as_ref() {
            let lock = self.is_complete.lock();
            if *lock {
                return;
            }
            assert!(s.device_stream_submission.is_some());
            s.wait().unwrap();
            *self.is_complete.lock() = true;
        } else {
            unreachable!()
        }
    }
}

pub struct BufferResidency<B: hal::Backend> {
    /// Residency state for each device
    pub devices: SmallVec<[DeviceResidencyState<B>; DEVICE_SMALLVEC_SIZE]>,
    /// Residency state for the host. Optional for destruction purposes
    pub host: Option<DeviceResidencyState<B>>,
    /// Alternative to residencystate buffer for host memory
    pub storage: Option<StorageResidencyState<B>>,
    /// Used to create indices for buffer access finishes
    pub current_index: u64,
    /// Sorted by range start
    pub read_accesses: HashMap<u64, Arc<BufferAccessFinish<B>>>,
    /// Sorted by order of submission. New accesses should start from the back to find the last
    /// conflicting accesses and wait for those.
    pub write_accesses: VecDeque<Arc<BufferAccessFinish<B>>>,
    /// Used to determine when to unmap
    pub num_mappings: u64,
}

impl<B: hal::Backend> BufferResidency<B> {
    pub fn new(num_devices: u32) -> Self {
        let mut devices = SmallVec::with_capacity(num_devices as usize);
        for _ in 0..num_devices {
            devices.push(DeviceResidencyState::default());
        }
        Self {
            devices,
            host: Some(DeviceResidencyState::default()),
            storage: None,
            current_index: 0,
            read_accesses: HashMap::new(),
            write_accesses: VecDeque::new(),
            num_mappings: 0,
        }
    }

    pub fn update_all_accesses(&mut self) -> SupaSimResult<B, ()> {
        let mut ids = Vec::with_capacity(64);
        for (id, access) in &self.read_accesses {
            if access.is_complete_host() {
                ids.push(*id);
            }
        }
        for id in ids {
            self.read_accesses.remove(&id);
        }

        let mut i = 0;
        while i < self.write_accesses.len() {
            if self.write_accesses[i].is_complete_host() {
                self.write_accesses.remove(i);
            } else {
                i += 1;
            }
        }

        for dev in &mut self.devices {
            if dev.buffer.is_some() {
                dev.ood_tracker.check_all_current_copies();
            }
        }
        Ok(())
    }

    pub unsafe fn destroy(&mut self, instance: &InstanceInner<B>) -> SupaSimResult<B, ()> {
        for (dev_id, dev) in std::mem::take(&mut self.devices)
            .into_iter()
            .chain([self.host.take().unwrap()])
            .enumerate()
        {
            let dev_id = if dev_id < instance.hal_devices.len() {
                dev_id
            } else {
                0
            };
            dev.destroy(instance, dev_id as u32)?;
        }
        Ok(())
    }
}
impl<B: hal::Backend> BufferResidency<B> {
    pub fn add_gpu_use(
        &mut self,
        range: BufferAccessRange,
        is_mut: bool,
        semaphore: Arc<Semaphore<B>>,
        device_index: u32,
    ) -> OutOfDateWait<B> {
        // TODO: make sure buffer is created
        // Push buffer access
        // Make each dependency signal a semaphore and hook into those
        // Update out of date ranges
        let finish = Arc::new(BufferAccessFinish::<B> {
            condvar: None,
            is_complete: Mutex::new(false),
            device_semaphore: Some(semaphore),
            range,
            id: self.current_index,
        });
        self.current_index += 1;
        if is_mut {
            self.read_accesses.insert(finish.id, finish.clone());
            for (i, d) in self
                .devices
                .iter_mut()
                .chain(std::iter::once(self.host.as_mut().unwrap()))
                .enumerate()
            {
                if (i as u32) != device_index && d.buffer.is_some() {
                    d.ood_tracker.invalidate_range(range);
                }
            }
            if let Some(d) = &mut self.storage {
                d.ood_tracker.invalidate_range(range);
            }
        } else {
            self.write_accesses.push_back(finish.clone());
        }

        self.update_all_accesses().unwrap();

        // TODO: push other dependencies to this wait, these are only for copies
        self.devices[device_index as usize]
            .ood_tracker
            .get_needed_waits(range)
    }

    pub fn release_cpu_access(&mut self, finish: Arc<BufferAccessFinish<B>>, is_mut: bool) {
        self.num_mappings -= 1;
        // Update the finish, condvar, and possibly semaphore
        // Remove from access list
        {
            let mut lock = finish.is_complete.lock();
            *lock = true;
            finish.condvar.as_ref().unwrap().notify_all();
        }
        if let Some(sem) = &finish.device_semaphore {
            sem.signal().unwrap();
        }
        if is_mut {
            let idx = self.write_accesses.iter().position(|a| {
                let a: &BufferAccessFinish<B> = a.as_ref();
                let b: &BufferAccessFinish<B> = finish.as_ref();
                (a as *const BufferAccessFinish<B>).addr()
                    == (b as *const BufferAccessFinish<B>).addr()
            });
            self.write_accesses.remove(idx.unwrap());
        } else {
            self.read_accesses.remove(&finish.id);
        }
    }

    pub fn try_remove_gpu_access(&mut self, finish: Arc<BufferAccessFinish<B>>, is_mut: bool) {
        if is_mut {
            let idx = self.write_accesses.iter().position(|a| {
                let a: &BufferAccessFinish<B> = a.as_ref();
                let b: &BufferAccessFinish<B> = finish.as_ref();
                (a as *const BufferAccessFinish<B>).addr()
                    == (b as *const BufferAccessFinish<B>).addr()
            });
            if let Some(idx) = idx {
                self.write_accesses.remove(idx);
            }
        } else {
            self.read_accesses.remove(&finish.id);
        }
    }
}

pub struct BufferResidencyRef<B: hal::Backend>(pub RwLock<BufferResidency<B>>);
impl<B: hal::Backend> BufferResidencyRef<B> {
    /// This doesn't need access for a long time, in fact that would cause deadlocks
    pub fn get_cpu_access(
        &self,
        range: BufferAccessRange,
        is_mut: bool,
    ) -> Arc<BufferAccessFinish<B>> {
        // TODO: make sure buffer is created

        // TODO: wgpu doesn't support buffer mapping while its in a command on the GPU.
        // We should in these cases request access to the entire buffer.
        let mut s = self.0.write();
        s.num_mappings += 1;
        // Push this buffer access, update out of date ranges
        // Wait for dependencies to finish
        let finish = Arc::new(BufferAccessFinish::<B> {
            condvar: Some(Condvar::new()),
            is_complete: Mutex::new(false),
            device_semaphore: None,
            range,
            id: s.current_index,
        });
        s.current_index += 1;
        let my_id = finish.id;

        if is_mut {
            s.read_accesses.insert(finish.id, finish.clone());
            for d in &mut s.devices {
                d.ood_tracker.invalidate_range(range);
            }
            if let Some(d) = &mut s.storage {
                d.ood_tracker.invalidate_range(range);
            }
        } else {
            s.write_accesses.push_back(finish.clone());
        }
        let mut residency = Some(s);
        let mut any_waits = false;
        loop {
            let mut wait_is_mut = false;
            let mut wait = None;
            if is_mut {
                for read in residency.as_ref().unwrap().read_accesses.values() {
                    if read.range.intersects(&range) && read.id < my_id {
                        wait = Some(read.clone());
                        break;
                    }
                }
            }
            if wait.is_none() {
                for write in residency.as_ref().unwrap().write_accesses.iter().rev() {
                    if write.range.intersects(&range) && write.id < my_id {
                        wait_is_mut = true;
                        wait = Some(write.clone());
                        break;
                    }
                }
            }
            if let Some(f) = wait {
                drop(residency);
                f.wait_host();
                residency = Some(self.0.write());
                if f.condvar.is_none() {
                    residency
                        .as_mut()
                        .unwrap()
                        .try_remove_gpu_access(f, wait_is_mut);
                }
                any_waits = true;
            } else {
                break;
            }
        }
        let mut residency = residency.unwrap();
        if any_waits {
            residency.update_all_accesses().unwrap();
        }
        let _needed_waits = residency
            .host
            .as_mut()
            .unwrap()
            .ood_tracker
            .get_needed_waits(range);

        // TODO: finalize copies for out of date stuff
        finish
    }
}
