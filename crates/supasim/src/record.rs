/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use std::{
    collections::{HashMap, hash_map::Entry},
    ops::Deref,
};

use hal::{Buffer as _, CommandRecorder as _, Device as _, HalBufferSlice, Stream as _};
use thunderdome::Index;
use types::SyncOperations;

use crate::{
    BufferCommand, BufferCommandInner, BufferRange, BufferSlice, CommandRecorderInner, Instance,
    Kernel, MapSupasimError as _, SupaSimError, SupaSimResult, sync::SubmissionResources,
};

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
    instance: &Instance<B>,
    vulkan_style: bool,
    device_idx: usize,
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
            instance.inner()?.devices[device_idx]
                .inner
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
            let _instance = instance.inner()?;
            let device = _instance.devices[device_idx].inner.lock();
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
    instance: Instance<B>,
    recorder: &mut B::CommandRecorder,
    write_buffer: &Option<B::Buffer>,
    device_idx: usize,
    stream_idx: usize,
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
                buffer: res.residency.devices[device_idx].buffer.as_ref().unwrap(),
                offset: range.start,
                len: range.len,
            });
        }
        let bg = unsafe {
            instance.devices[device_idx].streams[stream_idx]
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .create_bind_group(
                    instance.devices[device_idx].inner.lock().as_ref().unwrap(),
                    kernel.per_device[device_idx].as_mut().unwrap(),
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
                let buffer = buffer_locks[current_buffer_index].deref().residency.devices
                    [device_idx]
                    .buffer
                    .as_ref()
                    .unwrap();
                current_buffer_index += 1;
                buffer
            };
            let mut get_kernel = || {
                let kernel = kernel_locks[current_kernel_index].per_device[device_idx]
                    .as_ref()
                    .unwrap();
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
                .record_commands(
                    instance.devices[device_idx].streams[stream_idx]
                        .inner
                        .lock()
                        .as_ref()
                        .unwrap(),
                    &hal_commands,
                )
                .map_supasim()?
        };
    }
    Ok(bindgroups)
}
