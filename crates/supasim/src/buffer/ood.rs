/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use crate::{
    InstanceInner,
    buffer::{BufferRange, residency::BufferAccessFinish},
    sync::Semaphore,
};
use parking_lot::{Mutex, RwLock};
use std::sync::Arc;

// TODO: optimize and validate all OOD logic

pub struct OutOfDateWait<B: hal::Backend> {
    pub semaphores: Vec<Arc<BufferAccessFinish<B>>>,
    pub other_copy_range: Option<Arc<BufferAccessFinish<B>>>,
}

#[derive(Default)]
pub struct OutOfDateTracker<B: hal::Backend> {
    /// Sorted by range start. Ranges that will be out of date until their copy completes or is started.
    ///
    /// This represents ranges that are out of date after copies and operations finish.
    pub out_of_date_ranges: Vec<BufferRange>,
    /// Copies that will make ranges valid when complete. Second element is the actual range that will be updated.
    /// This may be shortened as immediate updates occur.
    pub current_copies: Vec<(Arc<BufferAccessFinish<B>>, BufferRange)>,
}

impl<B: hal::Backend> OutOfDateTracker<B> {
    pub fn uninit(length: u64) -> Self {
        Self {
            out_of_date_ranges: vec![BufferRange { start: 0, length }],
            current_copies: Vec::new(),
        }
    }

    /// Mark range as up to date
    pub fn update_range_immediate(&mut self, range: BufferRange) {
        let mut i = 0;
        while i < self.out_of_date_ranges.len() {
            let other = self.out_of_date_ranges[i];
            let subtract = other.subtract(&range);
            if subtract.0.length == 0 {
                self.out_of_date_ranges.remove(i);
                continue;
            }
            self.out_of_date_ranges[i] = subtract.0;
            if let Some(second) = subtract.1 {
                self.out_of_date_ranges.insert(i + 1, second);
                i += 1;
            }
            i += 1;
        }
    }

    /// Mark range as not up to date
    pub fn invalidate_range(&mut self, range: BufferRange) {
        if range.length == 0 {
            return;
        }
        let mut index = usize::MAX;
        let mut combined_range = range; // This will always be overridden
        for i in 0..self.out_of_date_ranges.len() {
            let other_range = self.out_of_date_ranges[i];
            if let Some(joined) = other_range.join(&range) {
                self.out_of_date_ranges[i] = joined;
                index = i;
                combined_range = joined;
                break;
            } else if range.start < other_range.start {
                self.out_of_date_ranges.insert(i, range);
                return;
            }
        }

        if index == usize::MAX {
            self.out_of_date_ranges.push(range);
            return;
        }

        let mut last_index_to_join = index;
        while last_index_to_join < self.out_of_date_ranges.len() - 1 {
            last_index_to_join += 1;
            if let Some(combined) =
                combined_range.join(&self.out_of_date_ranges[last_index_to_join])
            {
                combined_range = combined;
            } else {
                last_index_to_join -= 1;
                break;
            }
        }
        self.out_of_date_ranges[index] = combined_range;
        self.out_of_date_ranges
            .drain(index + 1..=last_index_to_join);
    }

    /// Mark range as up to date when something finishes
    pub fn update_range_delayed(&mut self, finish: Arc<BufferAccessFinish<B>>) {
        let range = finish.range;
        self.update_range_immediate(range);
        self.current_copies.push((finish, range));
    }

    /// Returns what needs to be waited for and what needs to be copied
    ///
    /// It is the responsibility of the caller to setup the extra copy's
    /// id and such.
    pub fn get_needed_waits(
        &mut self,
        range: BufferRange,
        instance: &InstanceInner<B>,
    ) -> OutOfDateWait<B> {
        if range.length == 0 {
            return OutOfDateWait {
                semaphores: Vec::new(),
                other_copy_range: None,
            };
        }
        self.check_all_current_copies();
        let mut waits = Vec::new();
        let mut needing_copy = vec![range];
        for &(ref copy, other_range) in &self.current_copies {
            if range.intersects(&other_range) {
                {
                    let mut lock = copy.device_semaphore.lock();
                    if lock.is_none() {
                        *lock = Some(Arc::new(Semaphore {
                            inner: Some(RwLock::new(instance.get_semaphore().unwrap())),
                            device_stream_submission: Some((u16::MAX, u16::MAX, u64::MAX)),
                            instance: instance.self_weak.as_ref().unwrap().upgrade().unwrap(),
                        }))
                    }
                }
                waits.push(copy.clone());

                let mut i = 0;
                while i < needing_copy.len() && !needing_copy.is_empty() {
                    let diff = needing_copy[i].subtract(&other_range);
                    if diff.0.length == 0 {
                        needing_copy.remove(i);
                        continue;
                    }
                    needing_copy[i] = diff.0;
                    i += 1;
                    if let Some(o) = diff.1 {
                        needing_copy.insert(i, o);
                        i += 1;
                    }
                }
            }
        }
        let extra_copy = if needing_copy.is_empty() {
            None
        } else {
            let start = needing_copy.iter().min_by_key(|a| a.start).unwrap().start;
            let end = needing_copy
                .iter()
                .max_by_key(|a| a.start + a.length)
                .unwrap();
            let end = end.start + end.length;
            let range = BufferRange {
                start,
                length: end - start,
            };
            let sem_raw = instance.get_semaphore().unwrap();
            let sem = Arc::new(Semaphore {
                inner: Some(RwLock::new(sem_raw)),
                device_stream_submission: Some((u16::MAX, u16::MAX, u64::MAX)),
                instance: instance.self_weak.as_ref().unwrap().upgrade().unwrap(),
            });
            Some(BufferAccessFinish {
                condvar: None,
                is_complete: Mutex::new(false),
                device_semaphore: Mutex::new(Some(sem)),
                range,
                id: u64::MAX,
            })
        };
        OutOfDateWait {
            semaphores: waits,
            other_copy_range: extra_copy.map(Arc::new),
        }
    }

    /// Applies updates from copies that have completed
    pub fn check_all_current_copies(&mut self) {
        for i in (0..self.current_copies.len()).rev() {
            if self.current_copies[i].1.length == 0 || self.current_copies[i].0.is_complete_host() {
                self.current_copies.remove(i);
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use crate::buffer::BufferRange;

    fn r(start: u64, length: u64) -> BufferRange {
        BufferRange { start, length }
    }

    #[test]
    fn range_intersects() {
        assert!(r(0, 10).intersects(&r(5, 10)));
        assert!(!r(0, 10).intersects(&r(10, 5)));
        assert!(!r(0, 10).intersects(&r(20, 5)));
    }

    #[test]
    fn range_subtract_middle_splits() {
        // subtracting [4,6) from [0,10) leaves [0,4) and [6,10)
        let (a, b) = r(0, 10).subtract(&r(4, 2));
        assert_eq!(a, r(0, 4));
        assert_eq!(b, Some(r(6, 4)));
    }

    #[cfg(feature = "vulkan")]
    mod tracker {
        use crate::buffer::{BufferRange, ood::OutOfDateTracker};

        type T = OutOfDateTracker<hal::Vulkan>;

        #[test]
        fn tracker_starts_all_out_of_date() {
            let t = T::uninit(100);
            assert_eq!(
                t.out_of_date_ranges,
                vec![BufferRange {
                    start: 0,
                    length: 100
                }]
            );
        }

        #[test]
        fn immediate_update_clears_range() {
            let mut t = T::uninit(100);
            t.update_range_immediate(BufferRange {
                start: 0,
                length: 100,
            });
            assert!(t.out_of_date_ranges.is_empty());
        }

        #[test]
        fn invalidate_merges_adjacent() {
            let mut t = T::uninit(0); // starts empty (length 0 -> [0,0))
            t.out_of_date_ranges.clear();
            t.invalidate_range(BufferRange {
                start: 0,
                length: 10,
            });
            t.invalidate_range(BufferRange {
                start: 10,
                length: 10,
            });
            assert_eq!(
                t.out_of_date_ranges,
                vec![BufferRange {
                    start: 0,
                    length: 20
                }]
            );
        }
    }
}
