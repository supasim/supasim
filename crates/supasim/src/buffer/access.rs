/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use crate::{Buffer, BufferSlice, Instance, SupaSimError, SupaSimResult};
use std::marker::PhantomData;

pub type UserBufferAccessClosure<'a, B> =
    Box<dyn FnOnce(&mut [MappedBuffer<'a, B>]) -> anyhow::Result<()>>;

pub type SendableUserBufferAccessClosure<B> =
    Box<dyn Send + FnOnce(&mut [MappedBuffer<'_, B>]) -> anyhow::Result<()>>;

/// If the entire buffer isn't the same type you are trying to read, read as bytes first then cast yourself.
/// SupaSim does checks for alignments and validates offsets with the size of types
pub struct MappedBuffer<'a, B: hal::Backend> {
    _instance: Instance<B>,
    inner: *mut u8,
    len: u64,
    buffer_align: u64,
    in_buffer_offset: u64,
    has_mut: bool,
    was_used_mut: bool,
    _buffer: Buffer<B>,
    _user_id: u64,
    /// Size of the vector in which the data is allocated for non memory mapping scenarios
    _vec_capacity: Option<usize>,
    _p: PhantomData<&'a ()>,
}

impl<B: hal::Backend> Instance<B> {
    /// If the closure panics, memory issues may occur.
    /// Also, calling any supasim related functions in
    /// the closure may cause deadlocks.
    #[allow(clippy::type_complexity)]
    pub fn access_buffers(
        &self,
        _closure: UserBufferAccessClosure<B>,
        _buffers: &[&BufferSlice<B>],
    ) -> SupaSimResult<B, ()> {
        todo!()
    }
}

impl<B: hal::Backend> Buffer<B> {
    pub fn write<T: bytemuck::Pod>(&self, _offset: u64, _data: &[T]) -> SupaSimResult<B, ()> {
        todo!()
    }
    pub fn read<T: bytemuck::Pod>(&self, _offset: u64, _out: &mut [T]) -> SupaSimResult<B, ()> {
        todo!()
    }
    pub fn access(
        &'_ self,
        _offset: u64,
        _len: u64,
        _needs_mut: bool,
    ) -> SupaSimResult<B, MappedBuffer<'_, B>> {
        todo!()
    }
}

impl<B: hal::Backend> MappedBuffer<'_, B> {
    pub fn readable<T: bytemuck::Pod>(&self) -> SupaSimResult<B, &[T]> {
        let s = unsafe { std::slice::from_raw_parts(self.inner, self.len as usize) };
        // Length of the slice is a multiple of the length of the type
        if self.len.is_multiple_of(size_of::<T>() as u64)
        // Length of the type is a multiple of the buffer alignment
        && ((size_of::<T>() as u64).is_multiple_of(self.buffer_align) || self.buffer_align.is_multiple_of(size_of::<T>() as u64))
            // The offset is reasonable given the size of T
            && self.in_buffer_offset.is_multiple_of(size_of::<T>() as u64)
        {
            Ok(bytemuck::cast_slice(s))
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }

    pub fn writeable<T: bytemuck::Pod>(&mut self) -> SupaSimResult<B, &mut [T]> {
        if !self.has_mut {
            return Err(SupaSimError::BufferRegionNotValid);
        }
        self.was_used_mut = true;
        let s = unsafe { std::slice::from_raw_parts_mut(self.inner, self.len as usize) };
        // Length of the slice is a multiple of the length of the type
        if self.len.is_multiple_of(size_of::<T>() as u64)
        // Length of the type is a multiple of the buffer alignment
        && ((size_of::<T>() as u64).is_multiple_of(self.buffer_align) || self.buffer_align.is_multiple_of(size_of::<T>() as u64))
            // The offset is reasonable given the size of T
            && self.in_buffer_offset.is_multiple_of(size_of::<T>() as u64)
        {
            Ok(bytemuck::cast_slice_mut(s))
        } else {
            Err(SupaSimError::BufferRegionNotValid)
        }
    }
}

impl<B: hal::Backend> Drop for MappedBuffer<'_, B> {
    fn drop(&mut self) {
        todo!()
    }
}
