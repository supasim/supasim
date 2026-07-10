/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use hal::Buffer as _;

use crate::{
    Buffer, BufferSlice, Instance, MapSupasimError, SupaSimError, SupaSimResult,
    buffer::{
        BufferRange,
        residency::{BufferAccessFinish, BufferResidency},
    },
};
use std::{collections::HashMap, sync::Arc};

/// If the entire buffer isn't the same type you are trying to read, read as bytes first then cast yourself.
/// SupaSim does checks for alignments and validates offsets with the size of types
pub struct MappedBuffer<B: hal::Backend> {
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
}

impl<B: hal::Backend> Instance<B> {
    #[allow(clippy::type_complexity)]
    pub fn access_buffers(
        &'_ self,
        buffers: &[&BufferSlice<B>],
    ) -> SupaSimResult<B, Vec<MappedBuffer<B>>> {
        let instance = self.inner()?;
        let mut buffer_locks = HashMap::new();
        let mut residency_locks = HashMap::new();
        let mut first_id = HashMap::new();
        let mut accesses = Vec::with_capacity(buffers.len());
        for buffer in buffers {
            let b_inner = buffer.buffer.inner()?;
            buffer_locks.entry(b_inner.id).or_insert(b_inner);
        }
        for (id, lock) in &mut buffer_locks {
            residency_locks.insert(id, lock.residency.0.write());
        }
        for b in buffers {
            let id = b.buffer.inner()?.id;

            let residency = residency_locks.get_mut(&id).unwrap();
            let access = residency.get_cpu_access(b.access.range, b.access.needs_mut, &instance);
            first_id.entry(id).or_insert(access.id);
            accesses.push((access, id, b.access.needs_mut, b.buffer.clone()));
        }
        drop(residency_locks);
        let mut mapped_buffers = Vec::with_capacity(buffers.len());
        for &(ref access, buffer_id, is_mut, ref buffer) in &accesses {
            let my_id = first_id[&buffer_id];
            let residency = &buffer_locks[&buffer_id].residency;

            residency.wait_for_cpu_access(access.range, is_mut, my_id, &instance);
            let mapped = MappedBuffer::new(
                &mut residency.0.write(),
                buffer.clone(),
                access.range.start,
                access.range.length,
                is_mut,
                access.clone(),
            );
            mapped_buffers.push(mapped);
        }
        Ok(mapped_buffers)
    }
}

impl<B: hal::Backend> Buffer<B> {
    pub fn write<T: bytemuck::Pod>(&self, start: u64, data: &[T]) -> SupaSimResult<B, ()> {
        let s = self.inner()?;
        let data = bytemuck::cast_slice::<T, u8>(data);
        let range = BufferRange {
            start,
            length: data.len() as u64,
        };
        // Single hoisted instance read guard (canonical lock order: Instance first).
        // It is a *read* guard, so it does not block other readers (the sync thread,
        // other accesses); the only instance *write* lockers are `destroy` /
        // `clear_cached_resources`. Holding it across the blocking `wait_for_cpu_access`
        // is a tolerated residual (see the lock-order note in `sync/mod.rs`); dropping
        // it earlier would require threading the instance guard out of that call.
        let instance = s.instance.inner()?;
        let access = s.residency.0.write().get_cpu_access(range, true, &instance);
        s.residency
            .wait_for_cpu_access(range, true, access.id, &instance);
        let mut lock = s.residency.0.write();
        unsafe {
            lock.host
                .buffer
                .as_mut()
                .unwrap()
                .write(
                    instance.hal_devices[0].inner.lock().as_ref().unwrap(),
                    start,
                    data,
                )
                .map_supasim()?;
        }
        // Drop the instance read guard before `release_cpu_access`: if a concurrent
        // GPU use attached a semaphore to this access, `release_cpu_access` signals
        // it via `Semaphore::signal`, which re-locks `instance.inner()`. Releasing
        // our guard first keeps that a fresh (non-nested) read lock.
        drop(instance);
        // Reuse the already-held write guard: re-acquiring `residency.0.write()`
        // here would self-deadlock (parking_lot RwLock is not reentrant).
        lock.release_cpu_access(access, true);
        Ok(())
    }
    pub fn read<T: bytemuck::Pod>(&self, start: u64, out: &mut [T]) -> SupaSimResult<B, ()> {
        let s = self.inner()?;
        let data = bytemuck::cast_slice_mut::<T, u8>(out);
        let range = BufferRange {
            start,
            length: data.len() as u64,
        };
        // See `write` above: hoisted instance *read* guard, held across the blocking
        // `wait_for_cpu_access` as a tolerated residual (no instance writer runs
        // concurrently with a live buffer access).
        let instance = s.instance.inner()?;
        let access = s
            .residency
            .0
            .write()
            .get_cpu_access(range, false, &instance);
        s.residency
            .wait_for_cpu_access(range, false, access.id, &instance);
        let mut lock = s.residency.0.write();
        unsafe {
            lock.host
                .buffer
                .as_mut()
                .unwrap()
                .read(
                    instance.hal_devices[0].inner.lock().as_ref().unwrap(),
                    start,
                    data,
                )
                .map_supasim()?;
        }
        // Drop the instance read guard before `release_cpu_access` (see `write`): it may
        // signal a semaphore that re-locks `instance.inner()`.
        drop(instance);
        // Reuse the already-held write guard: re-acquiring `residency.0.write()`
        // here would self-deadlock (parking_lot RwLock is not reentrant).
        lock.release_cpu_access(access, false);
        Ok(())
    }
    pub fn access(
        &'_ self,
        start: u64,
        length: u64,
        needs_mut: bool,
    ) -> SupaSimResult<B, MappedBuffer<B>> {
        let s = self.inner()?;
        let range = BufferRange { start, length };
        // See `write` above: hoisted instance *read* guard, held across the blocking
        // `wait_for_cpu_access` as a tolerated residual.
        let instance = s.instance.inner()?;
        let access = s
            .residency
            .0
            .write()
            .get_cpu_access(range, needs_mut, &instance);
        s.residency
            .wait_for_cpu_access(range, needs_mut, access.id, &instance);
        // Drop the instance guard before `MappedBuffer::new` re-locks the instance
        // internally (it takes its own `instance.inner()`), avoiding a nested guard.
        drop(instance);

        let mut residency = s.residency.0.write();
        Ok(MappedBuffer::new(
            &mut residency,
            self.clone(),
            start,
            length,
            needs_mut,
            access,
        ))
    }
}

impl<B: hal::Backend> MappedBuffer<B> {
    pub fn readable<T: bytemuck::Pod>(&self) -> SupaSimResult<B, &[T]> {
        // Length is guaranteed for this ptr
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
        // Length is guaranteed for this ptr
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

    fn new(
        residency: &mut BufferResidency<B>,
        buffer: Buffer<B>,
        start: u64,
        length: u64,
        needs_mut: bool,
        access: Arc<BufferAccessFinish<B>>,
    ) -> Self {
        let b = buffer.inner().unwrap();
        let instance_props = b.instance.inner().unwrap().hal_instance_properties;
        let (mapping, vec_capacity) = if instance_props.map_buffers {
            let mapping = unsafe {
                residency
                    .host
                    .buffer
                    .as_mut()
                    .unwrap()
                    .map(
                        b.instance.inner().unwrap().hal_devices[0]
                            .inner
                            .lock()
                            .as_ref()
                            .unwrap(),
                    )
                    .unwrap()
                    .add(start as usize)
            };
            (mapping, None)
        } else {
            let mut vec = vec![0u8; length as usize];
            buffer.read(start, &mut vec).unwrap();
            let ptr = vec.as_mut_ptr();
            let cap = vec.capacity();
            std::mem::forget(vec);
            (ptr, Some(cap))
        };
        let buffer_align = b.create_info.contents_align;
        let _instance = b.instance.clone();
        drop(b);

        MappedBuffer {
            _instance,
            inner: mapping,
            len: length,
            buffer_align,
            in_buffer_offset: start,
            has_mut: needs_mut,
            was_used_mut: false,
            buffer,
            access,
            vec_capacity,
        }
    }
}

impl<B: hal::Backend> Drop for MappedBuffer<B> {
    fn drop(&mut self) {
        let s = self.buffer.inner().unwrap();
        let mut residency = s.residency.0.write();
        if let Some(cap) = self.vec_capacity {
            // If the vec_capacity is Some() then this ptr was obtained from a vector,
            // which we must copy back into the buffer and then destroy
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
