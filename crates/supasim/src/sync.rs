use std::collections::{HashMap, hash_map::Entry};

use types::{Dag, NodeIndex, SyncOperations, toposort};

use crate::{
    Buffer, BufferCommand, BufferCommandInner, BufferRange, CommandRecorderInner, Event, Id,
    Kernel, SupaSimResult, UserBufferAccessClosure,
};

pub type CommandDag<B> = Dag<BufferCommand<B>>;

pub enum HalCommandBuilder {
    CopyBuffer {
        src_buffer: Id<Buffer<hal::Dummy>>,
        dst_buffer: Id<Buffer<hal::Dummy>>,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    },
    DispatchKernel {
        shader: Id<Kernel<hal::Dummy>>,
        bind_group: Vec<Id<Buffer<hal::Dummy>>>,
        push_constants: Vec<u8>,
        workgroup_dims: [u32; 3],
    },
    DispatchKernelIndirect {
        shader: Id<Kernel<hal::Dummy>>,
        bind_group: Vec<Id<Buffer<hal::Dummy>>>,
        push_constants: Vec<u8>,
        indirect_buffer: Id<Buffer<hal::Dummy>>,
        buffer_offset: u64,
        validate: bool,
    },
    /// Only for vulkan like synchronization
    SetEvent {
        event: Id<Event<hal::Dummy>>,
        wait: SyncOperations,
    },
    /// Only for vulkan like synchronization
    WaitEvent {
        event: Id<Event<hal::Dummy>>,
        signal: SyncOperations,
    },
    /// Only for vulkan like synchronization
    PipelineBarrier {
        before: SyncOperations,
        after: SyncOperations,
    },
    /// Only for vulkan like synchronization. Will hitch a ride with the previous PipelineBarrier or WaitEvent
    MemoryBarrier { resource: Id<Buffer<hal::Dummy>> },
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
    pub streams: Vec<CommandStream>,
    pub cpu_ops: Vec<CpuOperation<B>>,
}
pub fn assemble_dag<B: hal::Backend>(
    cr: &mut CommandRecorderInner<B>,
) -> SupaSimResult<B, CommandDag<B>> {
    let mut buffers_tracker: HashMap<Id<Buffer<B>>, Vec<(BufferRange, usize)>> = HashMap::new();

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
    dag: &CommandDag<B>,
    cr: &mut CommandRecorderInner<B>,
) -> SupaSimResult<B, ()> {
    todo!()
}
pub fn dag_to_command_streams<B: hal::Backend>(
    dag: &CommandDag<B>,
    vulkan_style: bool,
) -> SupaSimResult<B, StreamingCommands<B>> {
    let sorted = toposort(dag, None).unwrap();
    todo!()
}
pub fn record_command_streams<B: hal::Backend>(
    streams: &StreamingCommands<B>,
    cr: &mut CommandRecorderInner<B>,
) -> SupaSimResult<B, ()> {
    todo!()
}
