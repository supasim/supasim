/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use std::{collections::VecDeque, sync::Arc};

use hal::Buffer;
use parking_lot::{Condvar, Mutex};
use smallvec::SmallVec;

use crate::{DEVICE_SMALLVEC_SIZE, Instance, sync::Semaphore};

struct OutOfDateWait<B: hal::Backend> {
    semaphores: Vec<Arc<Semaphore<B>>>,
    needs_more_copy: bool,
}
#[derive(Default)]
struct OutOfDateTracker<B: hal::Backend> {
    /// Sorted by range start. Ranges that will be out of date until their copy completes or is started.
    out_of_date_ranges: Vec<BufferAccessRange>,
    current_copies: Vec<Arc<BufferAccessFinish<B>>>,
}
impl<B: hal::Backend> OutOfDateTracker<B> {
    pub fn update_range_immediate(&mut self, _range: BufferAccessRange) {
        todo!()
    }
    pub fn invalidate_range(&mut self, _range: BufferAccessRange) {
        todo!()
    }
    pub fn update_range_delayed(&mut self, _finish: BufferAccessFinish<B>) {
        todo!()
    }
    pub fn get_needed_waits(&mut self, _range: BufferAccessRange) -> OutOfDateWait<B> {
        todo!()
    }
}

/// Handles all residency and synchronizatino for a buffer
struct DeviceResidencyState<B: hal::Backend> {
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
struct StorageResidencyState<B: hal::Backend> {
    file_name: String,
    ood_tracker: OutOfDateTracker<B>,
}
/// Buffer range without read/write information
struct BufferAccessRange {
    pub start: u64,
    pub length: u64,
}
/// Contains data for waiting on a buffer access
struct BufferAccessFinish<B: hal::Backend> {
    /// The conditional variable, only used in CPU->CPU synchronization
    condvar: Option<Condvar>,
    is_complete: Mutex<bool>,
    device_semaphore: Option<Semaphore<B>>,
    range: BufferAccessRange,
}
pub struct BufferResidency<B: hal::Backend> {
    /// Residency state for each device
    pub devices: SmallVec<[DeviceResidencyState<B>; DEVICE_SMALLVEC_SIZE]>,
    /// Residency state for the host
    pub host: DeviceResidencyState<B>,
    /// Alternative to residencystate buffer for host memory
    pub storage: Option<StorageResidencyState<B>>,
    /// Sorted by range start
    pub read_accesses: Vec<Arc<BufferAccessFinish<B>>>,
    pub write_accesses: VecDeque<Arc<BufferAccessFinish<B>>>,
}
impl<B: hal::Backend> BufferResidency<B> {
    pub fn new(num_devices: u32) -> Self {
        let mut devices = SmallVec::with_capacity(num_devices as usize);
        for _ in 0..num_devices {
            devices.push(DeviceResidencyState::default());
        }
        Self {
            devices,
            host: DeviceResidencyState::default(),
            storage: None,
            read_accesses: Vec::new(),
            write_accesses: VecDeque::new(),
        }
    }
    pub unsafe fn destroy(&mut self, _instance: &Instance<B>) {
        let instance = _instance.inner().unwrap();
        for (dev_id, dev) in &mut self.devices.iter_mut().chain([&mut self.host]).enumerate() {
            if let Some(b) = dev.buffer.take() {
                unsafe {
                    let dev_id = if dev_id < instance.devices.len() {
                        dev_id
                    } else {
                        0
                    };
                    b.destroy(instance.devices[dev_id].inner.lock().as_ref().unwrap());
                }
            }
        }
    }
}
