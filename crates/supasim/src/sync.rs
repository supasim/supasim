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
use std::collections::{HashMap, hash_map::Entry};

use hal::{BackendInstance, CommandRecorder};
use thunderdome::Index;
use types::{Dag, NodeIndex, SyncOperations};

use crate::{
    BufferCommand, BufferCommandInner, BufferRange, CommandRecorderInner, MapSupasimError,
    SupaSimError, SupaSimResult, UserBufferAccessClosure,
};

pub type CommandDag<B> = Dag<BufferCommand<B>>;

pub enum HalCommandBuilder {
    CopyBuffer {
        src_buffer: Index,
        dst_buffer: Index,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
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
    MemoryBarrier { resource: Index },
    UpdateBindGroup {
        bg: Index,
        kernel: Index,
        resources: Vec<Index>,
    },
}

pub struct CommandStream {
    pub commands: Vec<HalCommandBuilder>,
    pub wait_semaphores: Vec<u32>,
    pub signal_semaphore: Option<u32>,
}
pub struct CpuOperation<B: hal::Backend> {
    pub closure: UserBufferAccessClosure<B>,
    pub wait_semaphores: Vec<u32>,
    pub signal_semaphore: Option<u32>,
}
pub struct StreamingCommands<B: hal::Backend> {
    pub num_semaphores: u32,
    #[allow(clippy::type_complexity)] // Clippy's right but I'm lazy
    pub bindgroups: Vec<(Index, Vec<(Index, BufferRange)>)>,
    pub streams: Vec<CommandStream>,
    pub cpu_ops: Vec<CpuOperation<B>>,
}
pub fn assemble_dag<B: hal::Backend>(
    cr: &mut CommandRecorderInner<B>,
) -> SupaSimResult<B, CommandDag<B>> {
    let mut buffers_tracker: HashMap<Index, Vec<(BufferRange, usize)>> = HashMap::new();

    let commands = std::mem::take(&mut cr.commands);

    let mut dag = Dag::new();
    for cmd in commands {
        dag.add_node(cmd);
    }

    for i in 0..dag.node_count() {
        for bf_idx in 0..dag[NodeIndex::new(i)].buffers.len() {
            let buffer = &dag[NodeIndex::new(i)].buffers[bf_idx];
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
        }
    }
    dag.add_node(BufferCommand {
        inner: BufferCommandInner::Dummy,
        buffers: vec![],
        wait_handle: None,
    });
    for i in 0..dag.node_count() - 1 {
        dag.add_edge(NodeIndex::new(dag.node_count() - 1), NodeIndex::new(i), ())
            .unwrap();
    }
    dag.transitive_reduce(vec![NodeIndex::new(dag.node_count() - 1)]);
    Ok(dag)
}
pub fn record_dag<B: hal::Backend>(
    _dag: &CommandDag<B>,
    _cr: &mut CommandRecorderInner<B>,
) -> SupaSimResult<B, ()> {
    // TODO: work on this when cuda support lands
    todo!()
}
pub fn dag_to_command_streams<B: hal::Backend>(
    _dag: &CommandDag<B>,
    _vulkan_style: bool,
) -> SupaSimResult<B, StreamingCommands<B>> {
    // TODO: priority
    todo!()
}
pub fn record_command_streams<B: hal::Backend>(
    streams: &StreamingCommands<B>,
    cr: &mut CommandRecorderInner<B>,
) -> SupaSimResult<B, ()> {
    let mut instance = cr.instance.inner_mut()?;
    let _supports_signal = instance.inner_properties.semaphore_signal;
    let mut semaphores = Vec::new();
    for _ in 0..streams.num_semaphores {
        semaphores.push(cr.instance.acquire_wait_handle()?);
    }
    let mut bindgroups = Vec::new();
    for bg in &streams.bindgroups {
        let _k = instance
            .kernels
            .get(bg.0)
            .ok_or(SupaSimError::AlreadyDestroyed)?
            .as_ref()
            .ok_or(SupaSimError::AlreadyDestroyed)?
            .upgrade()?;
        let mut kernel = _k.inner_mut()?;
        let mut resources_a = Vec::new();
        for res in &bg.1 {
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
            resource_locks.push(res.as_inner()?);
        }
        let mut resources = Vec::new();
        for (i, res) in resource_locks.iter().enumerate() {
            let range = bg.1[i].1;
            resources.push(hal::GpuResource::Buffer {
                buffer: &res.inner,
                offset: range.start,
                size: range.len,
            });
        }
        let bg = if let Some(mut bg) = instance.hal_bind_groups.pop() {
            unsafe {
                instance
                    .inner
                    .update_bind_group(&mut bg, &mut kernel.inner, &resources)
                    .map_supasim()?
            }
            bg
        } else {
            unsafe {
                instance
                    .inner
                    .create_bind_group(&mut kernel.inner, &resources)
                    .map_supasim()?
            }
        };
        bindgroups.push(bg);
    }
    if !streams.cpu_ops.is_empty() {
        unimplemented!()
    }
    cr.command_recorders.clear();
    for stream in &streams.streams {
        let mut cr = match instance.hal_command_recorders.pop() {
            Some(a) => a,
            None => unsafe { instance.inner.create_recorder().map_supasim()? },
        };
        let mut buffer_locks = Vec::new();
        let mut kernel_locks = Vec::new();
        for cmd in &stream.commands {
            match cmd {
                HalCommandBuilder::CopyBuffer {
                    src_buffer,
                    dst_buffer,
                    ..
                } => {
                    buffer_locks.push(
                        instance
                            .buffers
                            .get(*src_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed)?,
                    );
                    buffer_locks.push(
                        instance
                            .buffers
                            .get(*dst_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed)?,
                    );
                }
                HalCommandBuilder::DispatchKernel { kernel, .. } => {
                    kernel_locks.push(
                        instance
                            .kernels
                            .get(*kernel)
                            .ok_or(SupaSimError::AlreadyDestroyed)?,
                    );
                }
                HalCommandBuilder::DispatchKernelIndirect {
                    kernel,
                    indirect_buffer,
                    ..
                } => {
                    kernel_locks.push(
                        instance
                            .kernels
                            .get(*kernel)
                            .ok_or(SupaSimError::AlreadyDestroyed)?,
                    );
                    buffer_locks.push(
                        instance
                            .buffers
                            .get(*indirect_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed)?,
                    );
                }
                HalCommandBuilder::MemoryBarrier { resource } => {
                    buffer_locks.push(
                        instance
                            .buffers
                            .get(*resource)
                            .ok_or(SupaSimError::AlreadyDestroyed)?,
                    );
                }
                _ => (),
            }
        }
        let mut hal_commands = Vec::new();
        unsafe {
            cr.record_commands(&mut instance.inner, &mut hal_commands)
                .map_supasim()?
        };
    }
    if instance.inner_properties.easily_update_bind_groups {
        instance.hal_bind_groups.extend(bindgroups);
    }
    // TODO: priority
    todo!()
}
