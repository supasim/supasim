/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use hal::{
    BindGroup as _, Buffer as _, CommandRecorder, Device as _, HalBufferSlice, RecorderSubmitInfo,
    Semaphore, Stream as _,
};
use parking_lot::{Condvar, Mutex};
use std::collections::HashMap;
use std::collections::hash_map::Entry;
use std::marker::PhantomData;
use std::ops::Deref;
use std::panic::UnwindSafe;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};
use thunderdome::Index;
use types::SyncOperations;

use crate::{
    Buffer, BufferCommand, BufferCommandInner, BufferRange, BufferSlice, BufferUserId,
    CommandRecorderInner, InstanceState, Kernel, MapSupasimError, MappedBuffer,
    SendableUserBufferAccessClosure, SupaSimError, SupaSimInstance, SupaSimResult,
};

pub struct SubmissionResources<B: hal::Backend> {
    pub kernels: Vec<crate::Kernel<B>>,
    pub buffers: Vec<crate::Buffer<B>>,
    pub temp_copy_buffer: Option<B::Buffer>,
}
impl<B: hal::Backend> Default for SubmissionResources<B> {
    fn default() -> Self {
        Self {
            kernels: Vec::new(),
            buffers: Vec::new(),
            temp_copy_buffer: None,
        }
    }
}

pub enum HalCommandBuilder {
    CopyBuffer {
        src_buffer: Index,
        dst_buffer: Index,
        src_offset: u64,
        dst_offset: u64,
        len: u64,
    },
    CopyFromTemp {
        src_offset: u64,
        dst_buffer: Index,
        dst_offset: u64,
        len: u64,
    },
    ZeroBuffer {
        buffer: Index,
        offset: u64,
        size: u64,
    },
    DispatchKernel {
        kernel: Index,
        bg: u32,
        push_constants: Vec<u8>,
        workgroup_dims: [u32; 3],
    },
    /// Only for vulkan like synchronization
    PipelineBarrier {
        before: SyncOperations,
        after: SyncOperations,
    },
    /// Only for vulkan like synchronization. Will hitch a ride with the previous PipelineBarrier or WaitEvent
    MemoryBarrier {
        resource: Index,
        offset: u64,
        len: u64,
    },
    MemoryTransfer {
        resource: Index,
        offset: u64,
        len: u64,
        import: bool,
    },
    UpdateBindGroup {
        bg: Index,
        kernel: Index,
        resources: Vec<Index>,
    },
    Dummy,
}

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct BindGroupDesc {
    kernel_idx: Index,
    items: Vec<(Index, BufferRange)>,
}
impl BindGroupDesc {
    fn uninit() -> Self {
        Self {
            kernel_idx: Index::DANGLING,
            items: vec![],
        }
    }
}
pub struct CommandStream {
    pub commands: Vec<HalCommandBuilder>,
}
/// These are split into multiple streams so that certain operations can be waited without waiting for all
pub struct StreamingCommands<B: hal::Backend> {
    /// Contains the index of the kernel, the index of the buffer, and the range of the buffer
    pub bind_groups: Vec<BindGroupDesc>,
    pub streams: Vec<CommandStream>,
    pub used_ranges: HashMap<Index, Vec<BufferRange>>,
    pub resources: SubmissionResources<B>,
}

struct StreamBlock<'a, B: hal::Backend> {
    commands: Vec<&'a BufferCommand<B>>,
    /// Indices of commands needing sync
    needing_sync: BufferUsageTracker,
}
impl<'a, B: hal::Backend> StreamBlock<'a, B> {
    pub fn new() -> Self {
        Self {
            commands: Vec::new(),
            needing_sync: BufferUsageTracker::default(),
        }
    }
    pub fn overlapping_buffers(&mut self, cmd: &BufferCommand<B>) -> Vec<BufferSlice<B>> {
        let mut slices = Vec::new();
        for &other_cmd in &self.commands {
            for b1 in &cmd.buffers {
                for b2 in &other_cmd.buffers {
                    if b1.buffer == b2.buffer
                        && let Some(intersection) = b1.range().intersection(&b2.range())
                    {
                        slices.push(BufferSlice {
                            buffer: b1.buffer.clone(),
                            start: intersection.start,
                            len: intersection.len,
                            needs_mut: true,
                        });
                    }
                }
            }
        }
        slices
    }
    pub fn append_command(&mut self, cmd: &'a BufferCommand<B>) {
        self.commands.push(cmd);
    }
    pub fn remove_command(&mut self, index: usize) {
        self.commands.remove(index);
    }
}

#[derive(Default)]
struct BufferUsageTracker {
    inner: HashMap<Index, Vec<BufferRange>>,
}
impl BufferUsageTracker {
    pub fn add<B: hal::Backend>(&mut self, b: &BufferSlice<B>) -> SupaSimResult<B, ()> {
        let id = b.buffer.inner()?.id;
        match self.inner.entry(id) {
            Entry::Occupied(mut v) => {
                v.get_mut().push(b.range());
            }
            Entry::Vacant(v) => {
                v.insert(vec![b.range()]);
            }
        }
        Ok(())
    }
    pub fn compact(&mut self) {
        for v in self.inner.values_mut() {
            // The idea here is to minimize the number of individual usage elements
            let mut i = 0;
            while i < v.len() {
                let mut j = i + 1;
                while j < v.len() {
                    if let Some(joined) = v[i].try_join(&v[j]) {
                        v[i] = joined;
                        v.remove(j);
                        if j < i {
                            i -= 1;
                        }
                    } else {
                        j += 1;
                    }
                }
                i += 1;
            }
        }
    }
}

pub fn assemble_streams<B: hal::Backend>(
    crs: &mut [&mut CommandRecorderInner<B>],
    instance: &InstanceState<B>,
    vulkan_style: bool,
) -> SupaSimResult<B, StreamingCommands<B>> {
    // TODO: clean up stuff like .inner()? by possibly acquiring locks over all of the used buffers and stuff
    if crs.iter().all(|a| a.commands.is_empty()) {
        return Ok(StreamingCommands {
            bind_groups: Vec::new(),
            streams: vec![CommandStream {
                commands: Vec::new(),
            }],
            used_ranges: HashMap::new(),
            resources: SubmissionResources::default(),
        });
    }

    let mut resources = SubmissionResources::default();
    let mut buffers_tracker = BufferUsageTracker::default();
    let mut bind_groups: HashMap<BindGroupDesc, u32> = HashMap::new();

    let mut commands = Vec::new();
    let mut src_buffer_len = 0;
    for cr in crs.iter_mut() {
        let mut cmds = std::mem::take(&mut cr.commands);
        for cmd in &mut cmds {
            for b in &cmd.buffers {
                buffers_tracker.add(b)?;
            }
            if let BufferCommandInner::CopyFromTemp { src_offset } = &mut cmd.inner {
                *src_offset += src_buffer_len;
            }
            for b in &cmd.buffers {
                resources.buffers.push(b.buffer.clone());
            }
            if let BufferCommandInner::KernelDispatch { kernel, .. } = &cmd.inner
                && !resources.kernels.contains(kernel)
            {
                resources.kernels.push(kernel.clone());
            }
        }
        commands.extend(cmds);
        commands.push(BufferCommand {
            inner: BufferCommandInner::CommandRecorderEnd,
            buffers: vec![],
        });
        src_buffer_len += cr.writes_slice.len() as u64;
    }

    resources.temp_copy_buffer = if src_buffer_len > 0 {
        let mut buf = unsafe {
            instance
                .device
                .lock()
                .as_mut()
                .unwrap()
                .create_buffer(&types::HalBufferDescriptor {
                    size: src_buffer_len,
                    memory_type: types::HalBufferType::Upload,
                    min_alignment: 16,
                })
                .map_supasim()?
        };
        let mut current_offset = 0;
        unsafe {
            // Map it exactly once so we don't map/unmap for every CR
            let device = instance.device.lock();
            buf.map(device.as_ref().unwrap()).map_supasim()?;
            for cr in crs.iter_mut() {
                buf.write(device.as_ref().unwrap(), current_offset, &cr.writes_slice)
                    .map_supasim()?;
                current_offset += cr.writes_slice.len() as u64;
            }
            buf.unmap(device.as_ref().unwrap()).map_supasim()?;
        }
        Some(buf)
    } else {
        None
    };

    // Inspired by the greatest algorithm ever, bubble sort!
    // The idea is that commands will "bubble up" to the earliest block possible

    let mut blocks = vec![StreamBlock::new()];
    for cmd in &commands {
        let mut overlapping = blocks.last_mut().unwrap().overlapping_buffers(cmd);
        if overlapping.is_empty() {
            let mut last_supported = blocks.len() - 1;
            while last_supported > 1 {
                overlapping = blocks[last_supported - 1].overlapping_buffers(cmd);
                if overlapping.is_empty() {
                    last_supported -= 1;
                } else {
                    break;
                }
            }
            blocks[last_supported].append_command(cmd);
            for a in &overlapping {
                blocks[last_supported].needing_sync.add(a)?;
            }
        } else {
            let mut block = StreamBlock::new();
            block.append_command(cmd);
            for s in &overlapping {
                block.needing_sync.add(s)?;
            }
            blocks.push(block);
        }
    }
    for block in &mut blocks {
        // Metal backend may eventually allow parallelizing copies with other things
        // so we want to start them first and let them run in the background
        block.commands.sort_unstable_by_key(|a| match a.inner {
            BufferCommandInner::CopyBufferToBuffer
            | BufferCommandInner::CopyFromTemp { .. }
            | BufferCommandInner::ZeroBuffer => 0,
            _ => 1,
        });
    }

    let mut out_commands = Vec::new();

    for (block_idx, mut block) in blocks.into_iter().enumerate() {
        if vulkan_style && block_idx != 0 {
            out_commands.push(HalCommandBuilder::PipelineBarrier {
                before: SyncOperations::Both,
                after: SyncOperations::Both,
            });
            block.needing_sync.compact();
            for (b, ranges) in block.needing_sync.inner {
                for range in ranges {
                    out_commands.push(HalCommandBuilder::MemoryBarrier {
                        resource: b,
                        offset: range.start,
                        len: range.len,
                    });
                }
            }
        }
        for cmd in block.commands {
            let mut items = Vec::new();
            for b in &cmd.buffers {
                let id = b.buffer.inner()?.id;
                items.push((id, b.range()));
            }
            let new_cmd = match &cmd.inner {
                &BufferCommandInner::KernelDispatch {
                    ref kernel,
                    workgroup_dims,
                    ..
                } => {
                    let kernel_idx = kernel.inner()?.id;
                    let desc = BindGroupDesc {
                        kernel_idx: kernel.inner()?.id,
                        items,
                    };
                    let current_count = bind_groups.len() as u32;
                    let bg_idx = match bind_groups.entry(desc) {
                        Entry::Vacant(v) => {
                            v.insert(current_count);
                            current_count
                        }
                        Entry::Occupied(v) => *v.get(),
                    };
                    HalCommandBuilder::DispatchKernel {
                        kernel: kernel_idx,
                        bg: bg_idx,
                        push_constants: vec![],
                        workgroup_dims,
                    }
                }
                BufferCommandInner::Dummy => continue,
                BufferCommandInner::CopyBufferToBuffer => HalCommandBuilder::CopyBuffer {
                    src_buffer: cmd.buffers[0].buffer.inner()?.id,
                    dst_buffer: cmd.buffers[1].buffer.inner()?.id,
                    src_offset: cmd.buffers[0].start,
                    dst_offset: cmd.buffers[1].start,
                    len: cmd.buffers[0].len,
                },
                &BufferCommandInner::CopyFromTemp { src_offset } => {
                    HalCommandBuilder::CopyFromTemp {
                        src_offset,
                        dst_buffer: cmd.buffers[0].buffer.inner()?.id,
                        dst_offset: cmd.buffers[0].start,
                        len: cmd.buffers[0].len,
                    }
                }
                BufferCommandInner::ZeroBuffer => HalCommandBuilder::ZeroBuffer {
                    buffer: cmd.buffers[0].buffer.inner()?.id,
                    offset: cmd.buffers[0].start,
                    size: cmd.buffers[0].len,
                },
                BufferCommandInner::MemoryTransfer { import } => {
                    HalCommandBuilder::MemoryTransfer {
                        resource: cmd.buffers[0].buffer.inner()?.id,
                        offset: cmd.buffers[0].start,
                        len: cmd.buffers[0].len,
                        import: *import,
                    }
                }

                BufferCommandInner::CommandRecorderEnd => HalCommandBuilder::Dummy,
            };
            out_commands.push(new_cmd);
        }
    }

    let stream = CommandStream {
        commands: out_commands,
    };
    let mut bind_groups_out = vec![BindGroupDesc::uninit(); bind_groups.len()];
    for (bg, idx) in bind_groups {
        bind_groups_out[idx as usize] = bg;
    }
    buffers_tracker.compact();
    Ok(StreamingCommands {
        bind_groups: bind_groups_out,
        streams: vec![stream],
        used_ranges: buffers_tracker.inner,
        resources,
    })
}
#[allow(clippy::type_complexity)]
pub fn record_command_streams<B: hal::Backend>(
    streams: &StreamingCommands<B>,
    instance: SupaSimInstance<B>,
    recorder: &mut B::CommandRecorder,
    write_buffer: &Option<B::Buffer>,
) -> SupaSimResult<B, Vec<(B::BindGroup, Kernel<B>)>> {
    let instance = instance.inner()?;
    let mut bindgroups = Vec::new();
    for bg in &streams.bind_groups {
        let _k = instance
            .kernels
            .lock()
            .get(bg.kernel_idx)
            .ok_or(SupaSimError::AlreadyDestroyed("Kernel".to_owned()))?
            .upgrade()?;
        let mut kernel = _k.inner_mut()?;
        let mut resources_a = Vec::new();
        for res in &bg.items {
            resources_a.push(
                instance
                    .buffers
                    .lock()
                    .get(res.0)
                    .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                    .as_ref()
                    .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                    .upgrade()?,
            );
        }
        let mut resource_locks = Vec::new();
        for res in &resources_a {
            resource_locks.push(res.inner()?);
        }
        let mut resources = Vec::new();
        for (i, res) in resource_locks.iter().enumerate() {
            let range = bg.items[i].1;
            resources.push(hal::HalBufferSlice {
                buffer: res.inner.as_ref().unwrap(),
                offset: range.start,
                len: range.len,
            });
        }
        let bg = unsafe {
            instance
                .stream
                .lock()
                .as_mut()
                .unwrap()
                .create_bind_group(
                    instance.device.lock().as_ref().unwrap(),
                    kernel.inner.as_mut().unwrap(),
                    &resources,
                )
                .map_supasim()?
        };
        bindgroups.push((bg, _k.clone()));
    }
    for stream in &streams.streams {
        let mut buffer_refs = Vec::new();
        let mut kernel_refs = Vec::new();
        for cmd in &stream.commands {
            match cmd {
                HalCommandBuilder::CopyBuffer {
                    src_buffer,
                    dst_buffer,
                    ..
                } => {
                    buffer_refs.push(
                        instance
                            .buffers
                            .lock()
                            .get(*src_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                    buffer_refs.push(
                        instance
                            .buffers
                            .lock()
                            .get(*dst_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                }
                HalCommandBuilder::DispatchKernel { kernel, .. } => {
                    kernel_refs.push(
                        instance
                            .kernels
                            .lock()
                            .get(*kernel)
                            .ok_or(SupaSimError::AlreadyDestroyed("Kernel".to_owned()))?
                            .upgrade()?,
                    );
                }
                HalCommandBuilder::MemoryBarrier { resource, .. } => {
                    buffer_refs.push(
                        instance
                            .buffers
                            .lock()
                            .get(*resource)
                            .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                }
                HalCommandBuilder::CopyFromTemp { dst_buffer, .. } => buffer_refs.push(
                    instance
                        .buffers
                        .lock()
                        .get(*dst_buffer)
                        .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                        .as_ref()
                        .unwrap()
                        .upgrade()?,
                ),
                HalCommandBuilder::ZeroBuffer { buffer, .. } => buffer_refs.push(
                    instance
                        .buffers
                        .lock()
                        .get(*buffer)
                        .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                        .as_ref()
                        .unwrap()
                        .upgrade()?,
                ),
                HalCommandBuilder::MemoryTransfer { resource, .. } => buffer_refs.push(
                    instance
                        .buffers
                        .lock()
                        .get(*resource)
                        .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                        .as_ref()
                        .unwrap()
                        .upgrade()?,
                ),
                _ => (),
            }
        }
        let mut buffer_locks = Vec::new();
        for buffer_ref in &buffer_refs {
            buffer_locks.push(buffer_ref.inner()?);
        }
        let mut kernel_locks = Vec::new();
        for kernel_ref in &kernel_refs {
            kernel_locks.push(kernel_ref.inner()?);
        }
        let mut hal_commands = Vec::new();
        {
            let mut current_buffer_index = 0;
            let mut current_kernel_index = 0;
            let mut get_buffer = || {
                let buffer = buffer_locks[current_buffer_index]
                    .deref()
                    .inner
                    .as_ref()
                    .unwrap();
                current_buffer_index += 1;
                buffer
            };
            let mut get_kernel = || {
                let kernel = kernel_locks[current_kernel_index].inner.as_ref().unwrap();
                current_kernel_index += 1;
                kernel
            };
            for cmd in &stream.commands {
                let cmd = match cmd {
                    HalCommandBuilder::CopyBuffer {
                        src_offset,
                        dst_offset,
                        len,
                        ..
                    } => hal::BufferCommand::CopyBuffer {
                        src_buffer: get_buffer(),
                        dst_buffer: get_buffer(),
                        src_offset: *src_offset,
                        dst_offset: *dst_offset,
                        len: *len,
                    },
                    HalCommandBuilder::ZeroBuffer { offset, size, .. } => {
                        hal::BufferCommand::ZeroMemory {
                            buffer: HalBufferSlice {
                                buffer: get_buffer(),
                                offset: *offset,
                                len: *size,
                            },
                        }
                    }
                    HalCommandBuilder::DispatchKernel {
                        bg,
                        push_constants,
                        workgroup_dims,
                        ..
                    } => hal::BufferCommand::DispatchKernel {
                        kernel: get_kernel(),
                        bind_group: &bindgroups[*bg as usize].0,
                        push_constants,
                        workgroup_dims: *workgroup_dims,
                    },
                    HalCommandBuilder::MemoryBarrier { offset, len, .. } => {
                        hal::BufferCommand::MemoryBarrier {
                            buffer: HalBufferSlice {
                                buffer: get_buffer(),
                                offset: *offset,
                                len: *len,
                            },
                        }
                    }
                    HalCommandBuilder::PipelineBarrier { before, after } => {
                        hal::BufferCommand::PipelineBarrier {
                            before: *before,
                            after: *after,
                        }
                    }
                    HalCommandBuilder::CopyFromTemp {
                        src_offset,
                        dst_offset,
                        len,
                        ..
                    } => hal::BufferCommand::CopyBuffer {
                        src_buffer: write_buffer.as_ref().unwrap(),
                        dst_buffer: get_buffer(),
                        src_offset: *src_offset,
                        dst_offset: *dst_offset,
                        len: *len,
                    },
                    HalCommandBuilder::MemoryTransfer {
                        offset,
                        len,
                        import,
                        ..
                    } => {
                        hal_commands.push(hal::BufferCommand::PipelineBarrier {
                            before: if *import {
                                SyncOperations::None
                            } else {
                                SyncOperations::Both
                            },
                            after: if *import {
                                SyncOperations::Both
                            } else {
                                SyncOperations::None
                            },
                        });
                        hal::BufferCommand::MemoryTransfer {
                            buffer: HalBufferSlice {
                                buffer: get_buffer(),
                                offset: *offset,
                                len: *len,
                            },
                            import: *import,
                        }
                    }
                    HalCommandBuilder::UpdateBindGroup { .. } => todo!(),
                    HalCommandBuilder::Dummy => hal::BufferCommand::Dummy,
                };
                hal_commands.push(cmd);
                // Add the commands n shit
            }
        }
        unsafe {
            recorder
                .record_commands(instance.stream.lock().as_ref().unwrap(), &hal_commands)
                .map_supasim()?
        };
    }
    Ok(bindgroups)
}
pub struct GpuSubmissionInfo<B: hal::Backend> {
    /// There are no ordering guarantees between these recorders, but in practice they will execute in orders
    pub command_recorders: Vec<B::CommandRecorder>,
    pub bind_groups: Vec<(B::BindGroup, Kernel<B>)>,
    pub used_buffer_ranges: Vec<(BufferUserId, Buffer<B>)>,
    pub used_buffers: Vec<Buffer<B>>,
    pub used_resources: SubmissionResources<B>,
}
/// A job for the CPU to run when some GPU work has completed or immediately, without ideally blocking for long. This won't necessarily run before other submissions
pub enum SemaphoreFinishedJob<B: hal::Backend> {
    Dummy(PhantomData<B>),
}
impl<B: hal::Backend> SemaphoreFinishedJob<B> {
    pub fn run(self, _instance: &InstanceState<B>) -> SupaSimResult<B, ()> {
        match self {
            Self::Dummy(_) => unreachable!(),
        }
    }
}
/// A job for the CPU to run in between GPU submissions
pub enum CpuSubmission<B: hal::Backend> {
    CreateGpuBuffer {
        buffer_id: Buffer<B>,
    },
    DestroyGpuBuffer {
        buffer_id: Buffer<B>,
    },
    UserClosure {
        closure: SendableUserBufferAccessClosure<B>,
        buffers: Vec<BufferSlice<B>>,
    },
    Dummy,
}
impl<B: hal::Backend> CpuSubmission<B> {
    pub fn run(self, instance: &InstanceState<B>) -> SupaSimResult<B, ()> {
        match self {
            Self::CreateGpuBuffer { .. } => todo!(),
            Self::DestroyGpuBuffer { .. } => todo!(),
            Self::UserClosure { closure, buffers } => {
                let instance_self = unsafe { &*instance.myself.get() }
                    .as_ref()
                    .unwrap()
                    .upgrade()?;
                let properties = instance.instance_properties;
                let mut mapped_buffers = Vec::with_capacity(buffers.len());
                let mut ids = Vec::new();
                for b in &buffers {
                    b.validate()?;
                    ids.push(b.acquire(crate::BufferUser::Cpu, false)?.id);
                }
                if properties.map_buffers {
                    #[allow(clippy::never_loop)]
                    for (i, b) in buffers.iter().enumerate() {
                        let mut buffer_inner = b.buffer.inner_mut()?;
                        let mapping = unsafe {
                            buffer_inner
                                .inner
                                .as_mut()
                                .unwrap()
                                .map(instance.device.lock().as_ref().unwrap())
                                .map_supasim()?
                        };
                        mapped_buffers.push(MappedBuffer {
                            instance: instance_self.clone(),
                            inner: unsafe { mapping.add(b.start as usize) },
                            len: b.len,
                            buffer_align: buffer_inner.create_info.contents_align,
                            has_mut: b.needs_mut,
                            was_used_mut: false,
                            in_buffer_offset: b.start,
                            buffer: b.buffer.clone(),
                            vec_capacity: None,
                            user_id: ids[i],
                            _p: Default::default(),
                        });
                    }
                    // Memory issues if we don't unmap I guess
                    let error =
                        closure(&mut mapped_buffers).map_err(|e| SupaSimError::UserClosure(e));
                    drop(mapped_buffers);
                    error?;
                } else {
                    let mut buffer_contents = Vec::new();
                    for b in &buffers {
                        let mut buffer = b.buffer.inner_mut()?;
                        let _instance = buffer.instance.clone();
                        let mut data;
                        #[allow(clippy::uninit_vec)]
                        {
                            data = Vec::with_capacity(b.len as usize);
                            unsafe {
                                data.set_len(b.len as usize);
                                buffer
                                    .inner
                                    .as_mut()
                                    .unwrap()
                                    .map(_instance.inner()?.device.lock().as_ref().unwrap())
                                    .map_supasim()?;
                            };
                        };
                        buffer_contents.push(data);
                    }
                    for (i, a) in buffer_contents.iter_mut().enumerate() {
                        let b = &buffers[i];
                        let buffer_inner = b.buffer.inner()?;
                        let mapped = MappedBuffer {
                            instance: instance_self.clone(),
                            inner: a.as_mut_ptr(),
                            len: b.len,
                            buffer_align: buffer_inner.create_info.contents_align,
                            has_mut: b.needs_mut,
                            was_used_mut: false,
                            in_buffer_offset: b.start,
                            buffer: b.buffer.clone(),
                            vec_capacity: Some(a.capacity()),
                            user_id: ids[i],
                            _p: Default::default(),
                        };
                        mapped_buffers.push(mapped);
                    }
                    // Memory issues if we don't unmap I guess
                    let error =
                        closure(&mut mapped_buffers).map_err(|e| SupaSimError::UserClosure(e));
                    for b in buffer_contents {
                        std::mem::forget(b);
                    }
                    drop(mapped_buffers);
                    error?;
                }
            }
            Self::Dummy => (),
        }
        Ok(())
    }
}
/// An event sent to the sync thread
pub enum SendSyncThreadEvent<B: hal::Backend> {
    /// GPU work to be done when all prior work is completed
    AddSubmission(GpuSubmissionInfo<B>),
    /// CPU work to be completed when a submission is done or immediately if it is already complete
    AddFinishedJob(u64, SemaphoreFinishedJob<B>),
    /// CPU work to be completed between submissions
    CpuWork(CpuSubmission<B>),
    /// Any currently queued work will begin immediately instead of waiting for more
    SubmitBatchNow,
    WaitFinishAndShutdown,
}
pub struct SyncThreadSharedData<B: hal::Backend> {
    pub next_job: u64,
    pub error: Option<SupaSimError<B>>,
    pub next_submission_idx: u64,
}
pub type SyncThreadShared<B> = Arc<(Mutex<SyncThreadSharedData<B>>, Condvar)>;
struct SyncThreadData<B: hal::Backend> {
    shared: SyncThreadShared<B>,
    receiver: Receiver<SendSyncThreadEvent<B>>,
    instance: Arc<InstanceState<B>>,
}
impl<B: hal::Backend> UnwindSafe for SyncThreadData<B> {}
pub struct SyncThreadHandle<B: hal::Backend> {
    pub sender: Mutex<Sender<SendSyncThreadEvent<B>>>,
    pub shared_thread: SyncThreadShared<B>,
    pub thread: std::thread::JoinHandle<()>,
}
impl<B: hal::Backend> SyncThreadHandle<B> {
    pub fn submit_gpu(&self, submission: GpuSubmissionInfo<B>) -> SupaSimResult<B, u64> {
        self.sender
            .lock()
            .send(SendSyncThreadEvent::AddSubmission(submission))
            .unwrap();
        let mut lock = self.shared_thread.0.lock();
        if let Some(SupaSimError::SyncThreadPanic(e)) = &lock.error {
            return Err(SupaSimError::SyncThreadPanic(e.clone()));
        }
        let id = lock.next_submission_idx;
        lock.next_submission_idx += 1;
        drop(lock);
        Ok(id)
    }
    pub fn append_finished_job(
        &self,
        idx: u64,
        job: SemaphoreFinishedJob<B>,
    ) -> SupaSimResult<B, ()> {
        self.sender
            .lock()
            .send(SendSyncThreadEvent::AddFinishedJob(idx, job))
            .unwrap();
        Ok(())
    }
    pub fn wait_for(&self, idx: u64, force_wait: bool) -> SupaSimResult<B, bool> {
        if force_wait {
            let mut lock = self.shared_thread.0.lock();
            while lock.next_job <= idx {
                if let Some(SupaSimError::SyncThreadPanic(e)) = &lock.error {
                    return Err(SupaSimError::SyncThreadPanic(e.clone()));
                }
                self.shared_thread.1.wait(&mut lock);
            }
            Ok(true)
        } else {
            Ok(self.shared_thread.0.lock().next_job > idx)
        }
    }
    pub fn wait_for_idle(&self) -> SupaSimResult<B, ()> {
        let mut lock = self.shared_thread.0.lock();
        while lock.next_job < lock.next_submission_idx {
            if let Some(SupaSimError::SyncThreadPanic(e)) = &lock.error {
                return Err(SupaSimError::SyncThreadPanic(e.clone()));
            }
            self.shared_thread.1.wait(&mut lock);
        }
        Ok(())
    }
}
pub fn create_sync_thread<B: hal::Backend>(
    instance: SupaSimInstance<B>,
) -> SupaSimResult<B, SyncThreadHandle<B>> {
    let shared_thread = SyncThreadSharedData::<B> {
        next_job: 1,
        error: None,
        next_submission_idx: 1,
    };
    let shared = Arc::new((Mutex::new(shared_thread), Condvar::new()));
    let shared_copy = shared.clone();
    let (sender, receiver) = channel::<SendSyncThreadEvent<B>>();
    let thread = std::thread::spawn(move || {
        let shared = shared_copy;
        let data = SyncThreadData {
            shared: shared.clone(),
            receiver,
            instance: instance.inner().unwrap()._inner.clone(),
        };
        drop(instance);

        if let Err(e) = std::panic::catch_unwind(|| {
            let mut data = data;
            sync_thread_main(&mut data)
        }) {
            let mut lock = shared.0.lock();
            let mut error = String::from("Unknown panic");
            if e.is::<String>() {
                error = *e.downcast::<String>().unwrap();
            }
            lock.error = Some(SupaSimError::SyncThreadPanic(error.clone()));
            shared.1.notify_all();
            drop(lock);
            panic!("Sync thread encountered error: {error}");
        }
    });
    Ok(SyncThreadHandle {
        sender: Mutex::new(sender),
        shared_thread: shared,
        thread,
    })
}
enum Work<B: hal::Backend> {
    GpuSubmission(GpuSubmissionInfo<B>),
    CpuWork(CpuSubmission<B>),
}
fn sync_thread_main<B: hal::Backend>(logic: &mut SyncThreadData<B>) {
    const SUBMISSION_WAIT_PERIOD: Duration = Duration::from_millis(10);
    const MAX_SUBMISSION_WINDOW: Duration = Duration::from_millis(50);

    // Loop logic:
    // First, wait for a submission. Record that submission and wait the rest of ~5ms.
    // If there are more submissions during or by the end of this time, also record those. Then submit altogether.
    // A CPU submission before a GPU submission must break up the submission if the device doesn't support CPU semaphore signalling.
    // Otherwise, the following submission must wait on a CPu signalled semaphore
    //
    // Downsides of this are that if recording takes a long time there will be significant downtime. This can be prevented in the future using other methods, such as an intermediate recorder thread.
    let mut jobs = Vec::new();
    let mut next_submission_idx = 1;
    let mut _num_submitted_so_far = 1;
    let (semaphore_signal, map_buffer_while_gpu_use) = {
        (
            logic.instance.instance_properties.semaphore_signal,
            logic.instance.instance_properties.map_buffer_while_gpu_use,
        )
    };
    let mut semaphores: Vec<B::Semaphore> = Vec::new();
    let acquire_semaphore = |sems: &mut Vec<B::Semaphore>| -> SupaSimResult<B, B::Semaphore> {
        Ok(if let Some(mut s) = sems.pop() {
            unsafe {
                s.reset(logic.instance.device.lock().as_ref().unwrap())
                    .map_supasim()?;
            }
            s
        } else {
            unsafe {
                logic
                    .instance
                    .device
                    .lock()
                    .as_mut()
                    .unwrap()
                    .create_semaphore()
                    .map_supasim()?
            }
        })
    };
    loop {
        let mut temp_submission_vec = Vec::new();
        let mut submits = Vec::new();
        let mut used_semaphores = Vec::new();
        // Initial stuff - any non GPU work can be completed immediately
        loop {
            match logic.receiver.recv().unwrap() {
                SendSyncThreadEvent::AddFinishedJob(_, job) => job.run(&logic.instance).unwrap(),
                SendSyncThreadEvent::CpuWork(job) => job.run(&logic.instance).unwrap(),
                SendSyncThreadEvent::WaitFinishAndShutdown => {
                    for semaphore in semaphores {
                        unsafe {
                            semaphore
                                .destroy(logic.instance.device.lock().as_ref().unwrap())
                                .unwrap();
                        }
                    }
                    return;
                }
                SendSyncThreadEvent::AddSubmission(submission) => {
                    temp_submission_vec.push(Work::GpuSubmission(submission));
                    break;
                }
                SendSyncThreadEvent::SubmitBatchNow => (),
            }
        }
        let first_submission_time = Instant::now();
        let mut last_submission_time = first_submission_time;
        let mut last_was_submission = true;
        let mut final_cpu = None;
        loop {
            let now = Instant::now();
            if (now - first_submission_time) > MAX_SUBMISSION_WINDOW
                || (!last_was_submission && (now - last_submission_time) > SUBMISSION_WAIT_PERIOD)
            {
                break;
            }
            let max_wait = if last_was_submission {
                SUBMISSION_WAIT_PERIOD
            } else {
                SUBMISSION_WAIT_PERIOD - (now - last_submission_time)
            }
            .min(MAX_SUBMISSION_WINDOW - (now - first_submission_time));
            match logic.receiver.recv_timeout(max_wait) {
                Ok(SendSyncThreadEvent::WaitFinishAndShutdown) => {
                    for semaphore in semaphores {
                        unsafe {
                            semaphore
                                .destroy(logic.instance.device.lock().as_ref().unwrap())
                                .unwrap();
                        }
                    }
                    return;
                }
                Ok(SendSyncThreadEvent::CpuWork(job)) => {
                    if !semaphore_signal {
                        final_cpu = Some(job);
                        break;
                    }
                    temp_submission_vec.push(Work::CpuWork(job));
                    last_submission_time = Instant::now();
                    last_was_submission = true;
                }
                Ok(SendSyncThreadEvent::AddFinishedJob(idx, job)) => {
                    last_was_submission = false;
                    if idx < next_submission_idx {
                        job.run(&logic.instance).unwrap();
                    } else {
                        jobs.push((idx, job));
                    }
                }
                Ok(SendSyncThreadEvent::AddSubmission(submission)) => {
                    temp_submission_vec.push(Work::GpuSubmission(submission));
                    last_submission_time = Instant::now();
                    last_was_submission = true;
                }
                Ok(SendSyncThreadEvent::SubmitBatchNow) => break,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    panic!("Main thread disconnected from sender");
                }
            }
        }
        // Reverse sort so we can pop off the end
        jobs.sort_unstable_by_key(|a| u64::MAX - a.0);
        while let Some(job) = jobs.last() {
            if job.0 < next_submission_idx {
                let job = jobs.pop().unwrap();
                job.1.run(&logic.instance).unwrap();
            } else {
                break;
            }
        }
        let mut recorders = Vec::new();
        // Setup the submits and collect all needed semaphores
        {
            let mut prev_was_cpu = false;
            // First submit is always guaranteed to be a GPU submission
            for item in temp_submission_vec.iter_mut() {
                match item {
                    Work::CpuWork(_) => {
                        prev_was_cpu = true;
                    }
                    Work::GpuSubmission(g) => {
                        used_semaphores.push(acquire_semaphore(&mut semaphores).unwrap());
                        if prev_was_cpu {
                            used_semaphores.push(acquire_semaphore(&mut semaphores).unwrap());
                        }
                        prev_was_cpu = false;
                        recorders.push(std::mem::take(&mut g.command_recorders));
                    }
                }
            }
        }
        _num_submitted_so_far += temp_submission_vec.len();
        // Assemble the ranges for the wait semaphores
        let mut waits = Vec::new();
        {
            let mut prev_was_cpu = false;
            let mut semaphore_idx = 0;
            for item in temp_submission_vec.iter_mut() {
                match item {
                    Work::CpuWork(_) => prev_was_cpu = true,
                    Work::GpuSubmission(g) => {
                        if g.command_recorders.is_empty() {
                            continue;
                        }
                        // CPU wait semaphore at the start
                        if prev_was_cpu {
                            waits.push(&used_semaphores[semaphore_idx]);
                            semaphore_idx += 1;
                        }
                        // Signal semaphore at the end
                        semaphore_idx += 1;
                        prev_was_cpu = false;
                    }
                }
            }
        }
        // Give the wait/signal semaphores to the submits
        {
            let mut prev_was_cpu = false;
            let mut recorders_iter = recorders.iter_mut();
            let mut semaphore_idx = 0;
            let mut wait_idx = 0;
            for item in temp_submission_vec.iter_mut() {
                match item {
                    Work::CpuWork(_) => prev_was_cpu = true,
                    Work::GpuSubmission(_) => {
                        let next = recorders_iter.next().unwrap();
                        let count = next.len();
                        for (i, r) in next.iter_mut().enumerate() {
                            submits.push(RecorderSubmitInfo {
                                command_recorder: r,
                                wait_semaphores: if prev_was_cpu && i == 0 {
                                    semaphore_idx += 1;
                                    wait_idx += 1;
                                    std::slice::from_ref(&waits[wait_idx - 1])
                                } else {
                                    &[]
                                },
                                signal_semaphore: if i == count - 1 {
                                    Some(&used_semaphores[semaphore_idx])
                                } else {
                                    None
                                },
                            });
                        }
                        semaphore_idx += 1;
                        prev_was_cpu = false;
                    }
                }
            }
        }
        // Submit
        unsafe {
            logic
                .instance
                .stream
                .lock()
                .as_mut()
                .unwrap()
                .submit_recorders(&mut submits)
                .unwrap();
        }
        // Do the incremental waiting
        {
            let mut submit_idx = 0;
            let mut semaphore_idx = 0;
            for s in temp_submission_vec {
                match s {
                    Work::CpuWork(w) => {
                        w.run(&logic.instance).unwrap();
                        next_submission_idx += 1;
                        logic.shared.0.lock().next_job = next_submission_idx;
                        logic.shared.1.notify_all();
                    }
                    Work::GpuSubmission(mut item) => {
                        // TODO: verify that this makes any sense
                        /*for s in submits[submit_idx].wait_semaphores {
                            unsafe {
                                s.signal(logic.instance.device.lock().as_ref().unwrap())
                                    .unwrap();
                            }
                        }*/
                        semaphore_idx += submits[submit_idx].wait_semaphores.len();
                        unsafe {
                            used_semaphores[semaphore_idx]
                                .wait(logic.instance.device.lock().as_ref().unwrap())
                                .unwrap();
                        }
                        semaphore_idx += 1;
                        submit_idx += 1;

                        next_submission_idx += 1;
                        let mut lock = logic.shared.0.lock();
                        lock.next_job = next_submission_idx;
                        logic.shared.1.notify_all();
                        drop(lock);

                        for b in item.used_buffer_ranges {
                            if let Ok(b_inner) = b.1.inner() {
                                b_inner.slice_tracker.release(b.0);
                            }
                        }
                        if !map_buffer_while_gpu_use {
                            for b in item.used_buffers {
                                if let Ok(b_inner) = b.inner()
                                    && b_inner.last_used == next_submission_idx
                                {
                                    b_inner.slice_tracker.release_cpu();
                                }
                            }
                        }
                        item.used_resources.buffers.clear();
                        item.used_resources.kernels.clear();
                        if let Some(b) = std::mem::take(&mut item.used_resources.temp_copy_buffer) {
                            unsafe {
                                b.destroy(logic.instance.device.lock().as_ref().unwrap())
                                    .unwrap();
                            }
                        }
                        for (bg, kernel) in item.bind_groups {
                            let mut _k = kernel.inner_mut().unwrap();
                            let k = _k.inner.as_mut().unwrap();
                            unsafe {
                                bg.destroy(logic.instance.stream.lock().as_ref().unwrap(), k)
                                    .unwrap();
                            }
                        }
                    }
                }
                while let Some(last) = jobs.last() {
                    if last.0 < next_submission_idx {
                        jobs.pop().unwrap().1.run(&logic.instance).unwrap();
                    } else {
                        break;
                    }
                }
            }
        }
        if let Some(final_cpu) = final_cpu {
            final_cpu.run(&logic.instance).unwrap();
        }
        semaphores.append(&mut used_semaphores);
        let mut hal_recorders_lock = logic.instance.hal_command_recorders.lock();
        for r in &mut recorders {
            hal_recorders_lock.append(r);
        }
    }
}
