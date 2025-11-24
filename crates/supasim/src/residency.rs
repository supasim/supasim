/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use std::{
    collections::{HashMap, VecDeque},
    fs::File,
    sync::Arc,
};

use hal::{Buffer, Semaphore as _};
use parking_lot::{Condvar, Mutex, RwLock};
use smallvec::SmallVec;

use crate::{
    BufferRange, DEVICE_SMALLVEC_SIZE, Instance, InstanceInner, MapSupasimError, SupaSimResult,
    sync::Semaphore,
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
    pub fn check_all_current_copies(&mut self, instance: &InstanceInner<B>) {
        for i in (0..self.current_copies.len()).rev() {
            if self.current_copies[i].is_complete_host(instance) {
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
/// Stored in the file system
pub struct StorageResidencyState<B: hal::Backend> {
    /// The file used to store the buffer's contents
    file: File,
    /// The tracker for out of date ranges for this copy of the data
    ood_tracker: OutOfDateTracker<B>,
}
/// Buffer range without read/write information
pub struct BufferAccessRange {
    pub start: u64,
    pub length: u64,
}
impl From<BufferRange> for BufferAccessRange {
    fn from(value: BufferRange) -> Self {
        Self {
            start: value.start,
            length: value.len,
        }
    }
}
impl BufferAccessRange {
    pub fn join(&self, other: &Self) -> Option<Self> {
        if self.start < (other.start + other.length) && other.start < (self.start + self.length) {
            let start = self.start.min(other.start);
            Some(Self {
                start: self.start.min(other.start),
                length: (self.start + self.length).max(other.start + other.length) - start,
            })
        } else {
            None
        }
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
    pub fn is_complete_host(&self, instance: &InstanceInner<B>) -> bool {
        if let Some(cv) = self.condvar.as_ref() {
            let mut lock = self.is_complete.lock();
            loop {
                if *lock {
                    return true;
                }
                cv.wait(&mut lock);
            }
        } else if let Some(s) = self.device_semaphore.as_ref() {
            let lock = self.is_complete.lock();
            if *lock {
                return true;
            }
            assert!(s.device_stream_submission.is_some());
            unsafe {
                s.inner
                    .as_ref()
                    .unwrap()
                    .is_signalled(instance.hal_instance.read().as_ref().unwrap())
                    .unwrap();
            }
            todo!()
        } else {
            unreachable!()
        }
    }
}

pub struct BufferResidency<B: hal::Backend> {
    pub instance: Instance<B>,
    /// Residency state for each device
    pub devices: SmallVec<[DeviceResidencyState<B>; DEVICE_SMALLVEC_SIZE]>,
    /// Residency state for the host
    pub host: DeviceResidencyState<B>,
    /// Alternative to residencystate buffer for host memory
    pub storage: Option<StorageResidencyState<B>>,
    /// Used to create indices for buffer access finishes
    pub current_index: u64,
    /// Sorted by range start
    pub read_accesses: HashMap<u64, Arc<BufferAccessFinish<B>>>,
    /// Sorted by order of submission. New accesses should start from the back to find the last
    /// conflicting accesses and wait for those.
    pub write_accesses: VecDeque<Arc<BufferAccessFinish<B>>>,
}
impl<B: hal::Backend> BufferResidency<B> {
    pub fn new(instance: Instance<B>, num_devices: u32) -> Self {
        let mut devices = SmallVec::with_capacity(num_devices as usize);
        for _ in 0..num_devices {
            devices.push(DeviceResidencyState::default());
        }
        Self {
            instance,
            devices,
            host: DeviceResidencyState::default(),
            storage: None,
            current_index: 0,
            read_accesses: HashMap::new(),
            write_accesses: VecDeque::new(),
        }
    }
    pub unsafe fn destroy(&mut self, instance: &InstanceInner<B>) -> SupaSimResult<B, ()> {
        for (dev_id, dev) in &mut self.devices.iter_mut().chain([&mut self.host]).enumerate() {
            if let Some(b) = dev.buffer.take() {
                unsafe {
                    let dev_id = if dev_id < instance.hal_devices.len() {
                        dev_id
                    } else {
                        0
                    };
                    b.destroy(instance.hal_devices[dev_id].inner.lock().as_ref().unwrap())
                        .map_supasim()?;
                }
            }
        }
        Ok(())
    }
}
pub struct BufferResidencyRef<B: hal::Backend>(pub RwLock<BufferResidency<B>>);
impl<B: hal::Backend> BufferResidencyRef<B> {
    pub fn add_gpu_use(
        &self,
        _range: BufferAccessRange,
        _needs_mut: bool,
        _semaphore: Arc<Semaphore<B>>,
    ) -> OutOfDateWait<B> {
        todo!()
    }
    pub fn get_cpu_access(
        &self,
        _range: BufferAccessRange,
        _is_mut: bool,
    ) -> Arc<BufferAccessFinish<B>> {
        todo!()
    }
    pub fn release_cpu_access(&self, _finish: Arc<BufferAccessFinish<B>>, _is_mut: bool) {
        // Signal the finish in mutex
        // Signal the condvar
        // Update the out of date things
        // Remove the access remove from list
        todo!()
    }
}
