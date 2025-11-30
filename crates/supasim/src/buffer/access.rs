/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use hal::Buffer as _;

use crate::{
    Buffer, BufferSlice, Instance, MapSupasimError, SupaSimError, SupaSimResult,
    buffer::{BufferAccessRange, residency::BufferAccessFinish},
};
use std::{marker::PhantomData, sync::Arc};

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
    buffer: Buffer<B>,
    access: Arc<BufferAccessFinish<B>>,
    /// Size of the vector in which the data is allocated for non memory mapping scenarios
    vec_capacity: Option<usize>,
    _p: PhantomData<&'a ()>,
}

impl<B: hal::Backend> Instance<B> {
    /// If the closure panics, memory issues may occur.
    /// Also, calling any supasim related functions in
    /// the closure may cause deadlocks.
    ///
    /// This should only used to do quick operations on memory.
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
    pub fn write<T: bytemuck::Pod>(&self, start: u64, data: &[T]) -> SupaSimResult<B, ()> {
        let s = self.inner()?;
        let data = bytemuck::cast_slice::<T, u8>(data);
        let access = s.residency.get_cpu_access(
            BufferAccessRange {
                start,
                length: data.len() as u64,
            },
            true,
            &*s.instance.inner()?,
        );
        let mut lock = s.residency.0.write();
        unsafe {
            lock.host
                .buffer
                .as_mut()
                .unwrap()
                .write(
                    s.instance.inner()?.hal_devices[0]
                        .inner
                        .lock()
                        .as_ref()
                        .unwrap(),
                    start,
                    data,
                )
                .map_supasim()?;
        }
        s.residency.0.write().release_cpu_access(access, true);
        Ok(())
    }
    pub fn read<T: bytemuck::Pod>(&self, start: u64, out: &mut [T]) -> SupaSimResult<B, ()> {
        let s = self.inner()?;
        let data = bytemuck::cast_slice_mut::<T, u8>(out);
        let access = s.residency.get_cpu_access(
            BufferAccessRange {
                start,
                length: data.len() as u64,
            },
            false,
            &*s.instance.inner()?,
        );
        let mut lock = s.residency.0.write();
        unsafe {
            lock.host
                .buffer
                .as_mut()
                .unwrap()
                .read(
                    s.instance.inner()?.hal_devices[0]
                        .inner
                        .lock()
                        .as_ref()
                        .unwrap(),
                    start,
                    data,
                )
                .map_supasim()?;
        }
        s.residency.0.write().release_cpu_access(access, false);
        Ok(())
    }
    pub fn access(
        &'_ self,
        start: u64,
        length: u64,
        needs_mut: bool,
    ) -> SupaSimResult<B, MappedBuffer<'_, B>> {
        let s = self.inner()?;
        let instance_props = s.instance.inner()?.hal_instance_properties;

        let access = s.residency.get_cpu_access(
            BufferAccessRange { start, length },
            needs_mut,
            &*s.instance.inner()?,
        );
        let mut residency = s.residency.0.write();
        let (mapping, vec_capacity) = if instance_props.map_buffers {
            let mapping = unsafe {
                residency
                    .host
                    .buffer
                    .as_mut()
                    .unwrap()
                    .map(
                        s.instance.inner()?.hal_devices[0]
                            .inner
                            .lock()
                            .as_ref()
                            .unwrap(),
                    )
                    .map_supasim()?
                    .add(start as usize)
            };
            (mapping, None)
        } else {
            let mut vec = vec![0u8; length as usize];
            self.read(start, &mut vec)?;
            let ptr = vec.as_mut_ptr();
            let cap = vec.capacity();
            std::mem::forget(vec);
            (ptr, Some(cap))
        };
        drop(residency);
        Ok(MappedBuffer {
            _instance: s.instance.clone(),
            inner: mapping,
            len: length,
            buffer_align: s.create_info.contents_align,
            in_buffer_offset: start,
            has_mut: needs_mut,
            was_used_mut: false,
            buffer: self.clone(),
            access,
            vec_capacity,
            _p: Default::default(),
        })
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
        let s = self.buffer.inner().unwrap();
        let mut residency = s.residency.0.write();
        if let Some(cap) = self.vec_capacity {
            unsafe {
                let vec = Vec::from_raw_parts(self.inner, self.len as usize, cap);
                residency.devices[0]
                    .buffer
                    .as_mut()
                    .unwrap()
                    .write(
                        s.instance.inner().unwrap().hal_devices[0]
                            .inner
                            .lock()
                            .as_ref()
                            .unwrap(),
                        self.in_buffer_offset,
                        &vec,
                    )
                    .unwrap();
            }
        } else if !self
            .buffer
            .inner()
            .unwrap()
            .instance
            .inner()
            .unwrap()
            .hal_instance_properties
            .map_buffer_while_gpu_use
            && residency.num_mappings == 1
        {
            unsafe {
                residency
                    .host
                    .buffer
                    .as_mut()
                    .unwrap()
                    .unmap(
                        s.instance.inner().unwrap().hal_devices[0]
                            .inner
                            .lock()
                            .as_ref()
                            .unwrap(),
                    )
                    .unwrap();
            }
        }
        residency.release_cpu_access(self.access.clone(), self.has_mut);
    }
}
