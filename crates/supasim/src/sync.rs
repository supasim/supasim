/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 SupaMaggie70 (Magnus Larsson)


  SupaSim is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 3
  of the License, or (at your option) any later version.

  SupaSim is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
END LICENSE */
use hal::{BackendInstance, CommandRecorder, GpuResource};
use std::collections::{HashMap, hash_map::Entry};
use std::ops::Deref;
use thunderdome::Index;
use types::{Dag, NodeIndex, SyncOperations, Walker};

use crate::{
    BufferCommand, BufferCommandInner, BufferRange, BufferSlice, CommandRecorderInner, Kernel,
    MapSupasimError, SupaSimError, SupaSimInstance, SupaSimResult,
};

pub type CommandDag<B> = Dag<BufferCommand<B>>;

pub enum HalCommandBuilder {
    CopyBuffer {
        src_buffer: Index,
        dst_buffer: Index,
        src_offset: u64,
        dst_offset: u64,
        len: u64,
    },
    DispatchKernel {
        kernel: Index,
        bg: u32,
        push_constants: Vec<u8>,
        workgroup_dims: [u32; 3],
    },
    DispatchKernelIndirect {
        kernel: Index,
        bg: u32,
        push_constants: Vec<u8>,
        indirect_buffer: Index,
        buffer_offset: u64,
        validate: bool,
    },
    /// Only for vulkan like synchronization
    SetEvent { event: Index, wait: SyncOperations },
    /// Only for vulkan like synchronization
    WaitEvent {
        event: Index,
        signal: SyncOperations,
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
    UpdateBindGroup {
        bg: Index,
        kernel: Index,
        resources: Vec<Index>,
    },
}
pub struct BindGroupDesc {
    kernel_idx: Index,
    items: Vec<(Index, BufferRange)>,
}
pub struct CommandStream {
    pub commands: Vec<HalCommandBuilder>,
}
/// These are split into multiple streams so that certain operations can be waited without waiting for all
pub struct StreamingCommands {
    /// Contains the index of the kernel, the index of the buffer, and the range of the buffer
    pub bind_groups: Vec<BindGroupDesc>,
    pub streams: Vec<CommandStream>,
}

#[allow(clippy::type_complexity)]
pub fn assemble_dag<B: hal::Backend>(
    cr: &mut [&mut CommandRecorderInner<B>],
) -> SupaSimResult<B, (CommandDag<B>, HashMap<Index, Vec<BufferRange>>)> {
    let mut buffers_tracker: HashMap<Index, Vec<(BufferRange, usize)>> = HashMap::new();

    let mut commands = Vec::new();
    for cr in cr {
        let cmds = std::mem::take(&mut cr.commands);
        commands.extend(cmds);
    }

    let mut dag = Dag::new();
    for cmd in commands {
        dag.add_node(cmd);
    }

    for i in 0..dag.node_count() {
        let mut work_on_buffer =
            |buffer: &BufferSlice<B>, dag: &mut Dag<BufferCommand<B>>| -> SupaSimResult<B, ()> {
                let range: BufferRange = buffer.into();
                let id = buffer.buffer.inner()?.id;
                match buffers_tracker.entry(id) {
                    Entry::Occupied(mut entry) => {
                        for (range2, j) in entry.get().iter() {
                            if range.overlaps(range2) {
                                dag.add_edge(NodeIndex::new(*j), NodeIndex::new(i), ())
                                    .unwrap();
                            }
                        }
                        entry.get_mut().push((range, i));
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(vec![(range, i)]);
                    }
                }
                Ok(())
            };
        if let BufferCommandInner::CopyBufferToBuffer {
            src_buffer,
            dst_buffer,
            src_offset,
            dst_offset,
            len,
        } = &dag[NodeIndex::new(i)].inner
        {
            let src_slice = BufferSlice {
                buffer: src_buffer.clone(),
                start: *src_offset,
                len: *len,
                needs_mut: false,
            };
            let dst_slice = BufferSlice {
                buffer: dst_buffer.clone(),
                start: *dst_offset,
                len: *len,
                needs_mut: true,
            };
            work_on_buffer(&src_slice, &mut dag)?;
            work_on_buffer(&dst_slice, &mut dag)?;
        } else {
            for bf_idx in 0..dag[NodeIndex::new(i)].buffers.len() {
                let buffer = dag[NodeIndex::new(i)].buffers[bf_idx].clone();
                work_on_buffer(&buffer, &mut dag)?;
            }
        }
    }
    dag.add_node(BufferCommand {
        inner: BufferCommandInner::Dummy,
        buffers: vec![],
    });
    for i in 0..dag.node_count() - 1 {
        dag.add_edge(NodeIndex::new(dag.node_count() - 1), NodeIndex::new(i), ())
            .unwrap();
    }
    dag.transitive_reduce(vec![NodeIndex::new(dag.node_count() - 1)]);
    let out_map = buffers_tracker
        .into_iter()
        .map(|(key, value)| (key, value.iter().map(|a| a.0).collect()))
        .collect();
    Ok((dag, out_map))
}
#[allow(clippy::type_complexity)]
pub fn record_dag<B: hal::Backend>(
    _dag: &CommandDag<B>,
    _cr: &mut B::CommandRecorder,
) -> SupaSimResult<B, Vec<(B::BindGroup, Kernel<B>)>> {
    // TODO: work on this when cuda support lands
    todo!()
}
pub fn dag_to_command_streams<B: hal::Backend>(
    dag: &CommandDag<B>,
    vulkan_style: bool,
) -> SupaSimResult<B, StreamingCommands> {
    let mut bind_groups = Vec::new();
    let mut stream = CommandStream {
        commands: Vec::new(),
    };
    {
        // This algorithm fucking sucks. Its like topological sort but the layers are distinct, so that synchronization can be applied only at specific points
        let mut already_in = Vec::new();
        already_in.resize(dag.node_count(), false);
        let mut layers = Vec::new();
        layers.push(Vec::new());
        // I think it looks nicer
        #[allow(clippy::needless_range_loop)]
        for i in 0..dag.node_count() {
            if dag.parents(NodeIndex::new(i)).walk_next(dag).is_none() {
                layers[0].push(i);
                already_in[i] = true;
            }
        }
        while !layers.last().unwrap().is_empty() {
            let mut next_layer = Vec::new();
            let last_layer = layers.last().unwrap();
            for &node in last_layer {
                let mut walker = dag.children(NodeIndex::new(node));
                while let Some((_, child)) = walker.walk_next(dag) {
                    if !already_in[child.index()] {
                        next_layer.push(child.index());
                        already_in[child.index()] = true;
                    }
                }
            }
            layers.push(next_layer);
        }
        layers.pop();
        let nodes = dag.raw_nodes();
        for (i, layer) in layers.into_iter().enumerate() {
            // No synchronization needed for the first layer
            if vulkan_style && i != 0 {
                stream.commands.push(HalCommandBuilder::PipelineBarrier {
                    before: SyncOperations::Both,
                    after: SyncOperations::Both,
                });
                for &idx in &layer {
                    let cmd = &nodes[idx].weight;
                    for buffer in &cmd.buffers {
                        let id = buffer.buffer.inner()?.id;
                        stream.commands.push(HalCommandBuilder::MemoryBarrier {
                            resource: id,
                            offset: buffer.start,
                            len: buffer.len,
                        });
                    }
                    if let BufferCommandInner::CopyBufferToBuffer {
                        src_buffer,
                        dst_buffer,
                        src_offset,
                        dst_offset,
                        len,
                    } = &cmd.inner
                    {
                        let src_id = src_buffer.inner()?.id;
                        let dst_id = dst_buffer.inner()?.id;
                        stream.commands.push(HalCommandBuilder::MemoryBarrier {
                            resource: src_id,
                            offset: *src_offset,
                            len: *len,
                        });
                        stream.commands.push(HalCommandBuilder::MemoryBarrier {
                            resource: dst_id,
                            offset: *dst_offset,
                            len: *len,
                        });
                    }
                }
            }
            for idx in layer {
                let cmd = &nodes[idx].weight;
                let hal = match &cmd.inner {
                    BufferCommandInner::Dummy => continue,
                    BufferCommandInner::CopyBufferToBuffer {
                        src_buffer,
                        dst_buffer,
                        src_offset,
                        dst_offset,
                        len,
                    } => HalCommandBuilder::CopyBuffer {
                        src_buffer: src_buffer.inner()?.id,
                        dst_buffer: dst_buffer.inner()?.id,
                        src_offset: *src_offset,
                        dst_offset: *dst_offset,
                        len: *len,
                    },
                    BufferCommandInner::KernelDispatch {
                        kernel,
                        workgroup_dims,
                    } => {
                        let bg_index = bind_groups.len() as u32;
                        let bg = BindGroupDesc {
                            kernel_idx: kernel.inner()?.id,
                            items: cmd
                                .buffers
                                .iter()
                                .map(|a| {
                                    (
                                        a.buffer.inner().unwrap().id,
                                        BufferRange {
                                            start: a.start,
                                            len: a.len,
                                            needs_mut: a.needs_mut,
                                        },
                                    )
                                })
                                .collect(),
                        };
                        bind_groups.push(bg);
                        HalCommandBuilder::DispatchKernel {
                            kernel: kernel.inner()?.id,
                            bg: bg_index,
                            push_constants: Vec::new(),
                            workgroup_dims: *workgroup_dims,
                        }
                    }
                    BufferCommandInner::KernelDispatchIndirect {
                        kernel,
                        indirect_buffer,
                        needs_validation,
                    } => {
                        let bg_index = bind_groups.len() as u32;
                        let bg = BindGroupDesc {
                            kernel_idx: kernel.inner()?.id,
                            items: cmd
                                .buffers
                                .iter()
                                .map(|a| {
                                    (
                                        a.buffer.inner().unwrap().id,
                                        BufferRange {
                                            start: a.start,
                                            len: a.len,
                                            needs_mut: a.needs_mut,
                                        },
                                    )
                                })
                                .collect(),
                        };
                        bind_groups.push(bg);
                        HalCommandBuilder::DispatchKernelIndirect {
                            kernel: kernel.inner()?.id,
                            bg: bg_index,
                            push_constants: Vec::new(),
                            indirect_buffer: indirect_buffer.buffer.inner()?.id,
                            buffer_offset: indirect_buffer.start,
                            validate: *needs_validation,
                        }
                    }
                };
                stream.commands.push(hal);
            }
        }
    }
    // TODO: priority
    Ok(StreamingCommands {
        bind_groups,
        streams: vec![stream],
    })
}
#[allow(clippy::type_complexity)]
pub fn record_command_streams<B: hal::Backend>(
    streams: &StreamingCommands,
    instance: SupaSimInstance<B>,
    _recorder: &mut B::CommandRecorder,
) -> SupaSimResult<B, Vec<(B::BindGroup, Kernel<B>)>> {
    let mut instance = instance.inner_mut()?;
    let mut bindgroups = Vec::new();
    for bg in &streams.bind_groups {
        let _k = instance
            .kernels
            .get(bg.kernel_idx)
            .ok_or(SupaSimError::AlreadyDestroyed)?
            .as_ref()
            .ok_or(SupaSimError::AlreadyDestroyed)?
            .upgrade()?;
        let mut kernel = _k.inner_mut()?;
        let mut resources_a = Vec::new();
        for res in &bg.items {
            resources_a.push(
                instance
                    .buffers
                    .get(res.0)
                    .ok_or(SupaSimError::AlreadyDestroyed)?
                    .as_ref()
                    .ok_or(SupaSimError::AlreadyDestroyed)?
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
            resources.push(hal::GpuResource::Buffer {
                buffer: res.inner.as_ref().unwrap(),
                offset: range.start,
                len: range.len,
            });
        }
        let bg = unsafe {
            instance
                .inner
                .as_mut()
                .unwrap()
                .create_bind_group(kernel.inner.as_mut().unwrap(), &resources)
                .map_supasim()?
        };
        bindgroups.push((bg, _k.clone()));
    }
    for stream in &streams.streams {
        let mut cr = match instance.hal_command_recorders.pop() {
            Some(a) => a,
            None => unsafe {
                instance
                    .inner
                    .as_mut()
                    .unwrap()
                    .create_recorder()
                    .map_supasim()?
            },
        };
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
                            .get(*src_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed)?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                    buffer_refs.push(
                        instance
                            .buffers
                            .get(*dst_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed)?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                }
                HalCommandBuilder::DispatchKernel { kernel, .. } => {
                    kernel_refs.push(
                        instance
                            .kernels
                            .get(*kernel)
                            .ok_or(SupaSimError::AlreadyDestroyed)?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                }
                HalCommandBuilder::DispatchKernelIndirect {
                    kernel,
                    indirect_buffer,
                    ..
                } => {
                    kernel_refs.push(
                        instance
                            .kernels
                            .get(*kernel)
                            .ok_or(SupaSimError::AlreadyDestroyed)?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                    buffer_refs.push(
                        instance
                            .buffers
                            .get(*indirect_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed)?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                }
                HalCommandBuilder::MemoryBarrier { resource, .. } => {
                    buffer_refs.push(
                        instance
                            .buffers
                            .get(*resource)
                            .ok_or(SupaSimError::AlreadyDestroyed)?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                }
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
                let kernel = kernel_locks[current_kernel_index]
                    .deref()
                    .inner
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
                    HalCommandBuilder::DispatchKernelIndirect {
                        bg,
                        push_constants,
                        buffer_offset,
                        validate,
                        ..
                    } => hal::BufferCommand::DispatchKernelIndirect {
                        kernel: get_kernel(),
                        bind_group: &bindgroups[*bg as usize].0,
                        push_constants,
                        indirect_buffer: get_buffer(),
                        buffer_offset: *buffer_offset,
                        validate: *validate,
                    },
                    HalCommandBuilder::MemoryBarrier { offset, len, .. } => {
                        hal::BufferCommand::MemoryBarrier {
                            resource: GpuResource::Buffer {
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
                    // TODO: implement stuff for events and bind group updates
                    _ => unreachable!(),
                };
                hal_commands.push(cmd);
                // Add the commands n shit
            }
        }
        unsafe {
            cr.record_commands(instance.inner.as_mut().unwrap(), &mut hal_commands)
                .map_supasim()?
        };
    }
    // TODO: priority
    Ok(bindgroups)
}
