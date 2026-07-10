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
use parking_lot::RwLock;
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
        // Wait on every in-flight copy in `current_copies` that intersects `range`
        // (attaching a real completion semaphore to each if one isn't already set).
        //
        // We NEVER manufacture an `other_copy_range` placeholder here. Previously,
        // any part of `range` still in `out_of_date_ranges` with no in-flight copy
        // produced an `extra_copy` carrying a fresh timeline semaphore that NOTHING
        // ever signalled — leaking a HAL semaphore per submission and, if that stale
        // placeholder was ever waited, deadlocking (the B2 bug). Such still-out-of-date
        // ranges are benign and need no in-tracker copy:
        //   * A WRITE access produces the data itself (normal write-first case, e.g. a
        //     copy/dispatch destination that never held data anywhere). `add_gpu_use`
        //     then marks this location current via `update_range_immediate`.
        //   * A READ access has already been made current up front by
        //     `ensure_device_current` / `ensure_host_current`, so the range is no
        //     longer out of date by the time we get here.
        let mut waits = Vec::new();
        for (copy, other_range) in &self.current_copies {
            if range.intersects(other_range) {
                let mut lock = copy.device_semaphore.lock();
                if lock.is_none() {
                    *lock = Some(Arc::new(Semaphore {
                        inner: Some(RwLock::new(instance.get_semaphore().unwrap())),
                        device_stream_submission: Some((u16::MAX, u16::MAX, u64::MAX)),
                        instance: instance.self_weak.as_ref().unwrap().upgrade().unwrap(),
                    }))
                }
                drop(lock);
                waits.push(copy.clone());
            }
        }
        OutOfDateWait {
            semaphores: waits,
            other_copy_range: None,
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

    #[test]
    fn range_intersection() {
        // Overlapping -> the shared middle.
        assert_eq!(r(0, 10).intersection(&r(5, 10)), Some(r(5, 5)));
        assert_eq!(r(5, 10).intersection(&r(0, 10)), Some(r(5, 5)));
        // Fully contained -> the smaller.
        assert_eq!(r(0, 100).intersection(&r(10, 20)), Some(r(10, 20)));
        // Disjoint / adjacent -> None.
        assert_eq!(r(0, 10).intersection(&r(10, 5)), None);
        assert_eq!(r(0, 10).intersection(&r(20, 5)), None);
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

        // Mirrors the `needing_copy` computation at the top of `get_needed_waits`
        // (before the `current_copies` subtraction): the ranges that genuinely need
        // a copy are `out_of_date_ranges ∩ range`. This is the crux of the B2 fix —
        // once `ensure_*_current` marks a range current via `update_range_immediate`,
        // that range is gone from `out_of_date_ranges`, so no placeholder is produced.
        fn needing_copy(t: &T, range: BufferRange) -> Vec<BufferRange> {
            t.out_of_date_ranges
                .iter()
                .filter_map(|ood| ood.intersection(&range))
                .collect()
        }

        #[test]
        fn current_range_needs_no_copy() {
            let mut t = T::uninit(100);
            let range = BufferRange {
                start: 0,
                length: 100,
            };
            // Fresh buffer: everything is out of date, so the whole range needs a copy.
            assert_eq!(needing_copy(&t, range), vec![range]);
            // After `ensure_*_current` marks it current, NO range needs a copy — hence
            // no unsignalled placeholder semaphore is manufactured on the next access.
            t.update_range_immediate(range);
            assert!(needing_copy(&t, range).is_empty());
        }

        #[test]
        fn partially_current_range_needs_partial_copy() {
            let mut t = T::uninit(100);
            // Mark the middle [40,60) current; the ends stay out of date.
            t.update_range_immediate(BufferRange {
                start: 40,
                length: 20,
            });
            let asked = BufferRange {
                start: 0,
                length: 100,
            };
            assert_eq!(
                needing_copy(&t, asked),
                vec![
                    BufferRange {
                        start: 0,
                        length: 40
                    },
                    BufferRange {
                        start: 60,
                        length: 40
                    },
                ]
            );
        }

        #[test]
        fn invalidate_merges_adjacent() {
            // `uninit(0)` produces a single zero-length range [0,0); clear it so the
            // test starts from a genuinely empty tracker.
            let mut t = T::uninit(0);
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
