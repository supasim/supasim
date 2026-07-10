/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

pub mod access;
pub mod ood;
pub mod residency;

use crate::{Instance, InstanceInner, SupaSimError, SupaSimResult};
use residency::BufferResidencyRef;
use std::ops::Bound;
use thunderdome::Index;

/// Reference to a buffer as well as the range to be used & mutability state
#[derive(Clone, Debug)]
pub struct BufferSlice<B: hal::Backend> {
    pub buffer: Buffer<B>,
    pub access: BufferAccess,
}

impl<B: hal::Backend> BufferSlice<B> {
    pub fn validate(&self) -> SupaSimResult<B, ()> {
        let b = self.buffer.inner()?;
        if self
            .access
            .range
            .start
            .is_multiple_of(b.create_info.contents_align)
            && self
                .access
                .range
                .length
                .is_multiple_of(b.create_info.contents_align)
        {
            Ok(())
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }

    pub fn validate_with_align(&self, align: u64) -> SupaSimResult<B, ()> {
        let b = self.buffer.inner()?;
        // This is explained in MappedBuffer associated methods
        if align.is_multiple_of(b.create_info.contents_align)
            && self.access.range.start.is_multiple_of(align)
            && self.access.range.length.is_multiple_of(align)
        {
            Ok(())
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }

    pub fn entire_buffer(buffer: &Buffer<B>, needs_mut: bool) -> SupaSimResult<B, Self> {
        Ok(Self {
            buffer: buffer.clone(),
            access: BufferAccess {
                range: BufferRange {
                    start: 0,
                    length: buffer.inner()?.create_info.size,
                },
                needs_mut,
            },
        })
    }
}

/// A range and mutability without reference to a specific buffer
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferAccess {
    pub range: BufferRange,
    pub needs_mut: bool,
}

/// Just an interval
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct BufferRange {
    pub start: u64,
    pub length: u64,
}

impl BufferRange {
    pub fn join(&self, other: &Self) -> Option<Self> {
        // Use <= to treat adjacent (touching) intervals as joinable, e.g. [0,10) + [10,10) -> [0,20)
        if self.start <= (other.start + other.length) && other.start <= (self.start + self.length) {
            let start = self.start.min(other.start);
            Some(Self {
                start,
                length: (self.start + self.length).max(other.start + other.length) - start,
            })
        } else {
            None
        }
    }

    pub fn intersects(&self, other: &Self) -> bool {
        self.start < other.start + other.length && other.start < self.start + self.length
    }

    pub fn subtract(&self, other: &Self) -> (Self, Option<Self>) {
        let self_end = self.start + self.length;
        let other_end = other.start + other.length;
        if !self.intersects(other) {
            return (*self, None);
        }

        let mut r1 = None;
        let mut r2 = None;

        if self.start < other.start {
            r1 = Some(BufferRange {
                start: self.start,
                length: other.start - self.start,
            });
        }

        if self_end > other_end {
            let r = BufferRange {
                start: other_end,
                length: self_end - other_end,
            };
            if r1.is_none() {
                r1 = Some(r);
            } else {
                r2 = Some(r);
            }
        }

        (
            r1.unwrap_or(BufferRange {
                start: 0,
                length: 0,
            }),
            r2,
        )
    }
}

#[derive(Clone, Copy, Debug)]
pub struct BufferDescriptor {
    /// The size needed in bytes
    pub size: u64,
    /// The value that the contents of the buffer must be aligned to. This is important for when supasim must detect
    pub contents_align: u64,
    /// Currently unused. In the future this may be used to prefer keeping some buffers in memory when device runs out of memory and swapping becomes necessary
    pub priority: f32,
    /// If `Some`, the device given will be preferred for operations using this buffer. This is useful for example when exporting memory.
    pub preferred_device_index: Option<usize>,
}

impl Default for BufferDescriptor {
    fn default() -> Self {
        Self {
            size: 0,
            contents_align: 0,
            priority: 1.0,
            preferred_device_index: None,
        }
    }
}

impl BufferAccess {
    pub fn overlaps(&self, other: &Self) -> bool {
        // Starts before the end of the other
        // and the other starts before the end of this
        // and at least one of them is mutable
        self.range.start < other.range.start + other.range.length
            && other.range.start < self.range.start + self.range.length
            && (self.needs_mut || other.needs_mut)
    }

    pub fn overlaps_ignore_mut(&self, other: &Self) -> bool {
        self.range.start < other.range.start + other.range.length
            && other.range.start < self.range.start + self.range.length
    }

    pub fn contains(&self, other: &Self) -> bool {
        self.range.start <= other.range.start
            && self.range.start + self.range.length >= other.range.start + other.range.length
            && (!other.needs_mut || self.needs_mut)
    }

    pub fn try_join(&self, other: &Self) -> Option<Self> {
        if self.overlaps_ignore_mut(other) {
            if self.needs_mut == other.needs_mut {
                let start = self.range.start.min(other.range.start);
                let end = (self.range.start + self.range.length)
                    .max(other.range.start + other.range.length);
                let length = end - start;
                Some(Self {
                    needs_mut: self.needs_mut,
                    range: BufferRange { start, length },
                })
            } else if self.contains(other) {
                Some(*self)
            } else if other.contains(self) {
                Some(*other)
            } else {
                None
            }
        } else {
            None
        }
    }

    pub fn intersection(&self, other: &Self) -> Option<Self> {
        if self.overlaps(other) {
            let start = self.range.start.min(other.range.start);
            let end =
                (self.range.start + self.range.length).max(other.range.start + other.range.length);
            Some(Self {
                range: BufferRange {
                    start,
                    length: end - start,
                },
                needs_mut: true,
            })
        } else {
            None
        }
    }
}

api_type!(Buffer, {
    pub instance: Instance<B>,
    pub id: Index,
    pub create_info: BufferDescriptor,
    pub residency: BufferResidencyRef<B>,
    pub is_alive: bool,
},);

impl<B: hal::Backend> Buffer<B> {
    pub fn slice(&self, range: impl std::ops::RangeBounds<u64>, needs_mut: bool) -> BufferSlice<B> {
        let start = match range.start_bound() {
            Bound::Included(v) => *v,
            Bound::Excluded(v) => *v + 1,
            Bound::Unbounded => 0,
        };
        let end = match range.end_bound() {
            Bound::Included(v) => *v + 1,
            Bound::Excluded(v) => *v,
            Bound::Unbounded => self.inner().unwrap().create_info.size,
        };
        BufferSlice {
            buffer: self.clone(),
            access: BufferAccess {
                range: BufferRange {
                    start,
                    length: end - start,
                },
                needs_mut,
            },
        }
    }
}

impl<B: hal::Backend> std::fmt::Debug for Buffer<B> {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("Buffer(undecorated)")
    }
}

impl<B: hal::Backend> BufferInner<B> {
    pub(crate) fn destroy(&mut self, instance: &InstanceInner<B>) {
        instance.buffers.write().remove(self.id);
        unsafe {
            self.residency.0.write().destroy(instance).unwrap();
        }
        self.is_alive = false;
    }
}

impl<B: hal::Backend> Drop for BufferInner<B> {
    fn drop(&mut self) {
        if self.is_alive
            && let Ok(instance) = self.instance.clone().inner()
        {
            self.destroy(&instance);
        }
    }
}
