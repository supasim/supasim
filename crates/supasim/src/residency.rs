use std::{collections::VecDeque, sync::Arc};

use parking_lot::{Condvar, Mutex};
use smallvec::SmallVec;

use crate::InstanceState;

const DEVICE_SMALLVEC_SIZE: usize = 4;

/// A semaphore with info about its device that returns to a pool on drop
struct DeviceSemaphore<B: hal::Backend> {
    pub inner: Option<B::Semaphore>,
    pub device_index: usize,
    instance: InstanceState<B>,
}
impl<B: hal::Backend> Drop for DeviceSemaphore<B> {
    fn drop(&mut self) {
        // TODO: Return the used semaphore to the instance
    }
}
struct OutOfDateWait<B: hal::Backend> {
    semaphores: Vec<Arc<DeviceSemaphore<B>>>,
    needs_more_copy: bool,
}
struct OutOfDateTracker<B: hal::Backend> {
    /// Sorted by range start. Ranges that will be out of date until their copy completes or is started.
    out_of_date_ranges: Vec<BufferAccessRange>,
    current_copies: Vec<Arc<BufferAccessFinish<B>>>,
}
impl<B: hal::Backend> OutOfDateTracker<B> {
    pub fn update_range_immediate(&mut self, range: BufferAccessRange) {
        todo!()
    }
    pub fn invalidate_range(&mut self, range: BufferAccessRange) {
        todo!()
    }
    pub fn update_range_delayed(&mut self, finish: BufferAccessFinish<B>) {
        todo!()
    }
    pub fn get_needed_waits(&mut self, range: BufferAccessRange) -> OutOfDateWait<B> {
        todo!()
    }
}
/// Handles all residency and synchronizatino for a buffer
struct DeviceResidencyState<B: hal::Backend> {
    /// The backing memory
    buffer: Option<B::Buffer>,
    ood_tracker: OutOfDateTracker<B>,
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
    /// If none, it is a host access, and condvar is Some. If some, device_semaphores has length 1 and
    /// the only element is a host-waitable semaphore, and condvar is None.
    device_index: Option<u32>,
    /// The conditional variable, only used in CPU->CPU synchronization
    condvar: Option<Condvar>,
    is_complete: Mutex<bool>,
    device_semaphores: SmallVec<[Option<DeviceSemaphore<B>>; DEVICE_SMALLVEC_SIZE]>,
    range: BufferAccessRange,
}
pub struct BufferResidency<B: hal::Backend> {
    /// Residency state for each device
    devices: SmallVec<[DeviceResidencyState<B>; DEVICE_SMALLVEC_SIZE]>,
    /// Residency state for the host
    host: DeviceResidencyState<B>,
    /// Alternative to residencystate buffer for host memory
    storage: Option<StorageResidencyState<B>>,
    /// Sorted by range start
    read_accesses: Vec<Arc<BufferAccessFinish<B>>>,
    write_accesses: VecDeque<Arc<BufferAccessFinish<B>>>,
}
