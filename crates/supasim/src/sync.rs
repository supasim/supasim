use types::{Dag, SyncOperations};

use crate::{BufferCommand, CommandRecorderInner, Id, UserBufferAccessClosure};

pub type CommandDag<B> = Dag<BufferCommand<B>>;

pub enum HalCommandBuilder {
    CopyBuffer {
        src_buffer: Id,
        dst_buffer: Id,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    },
    DispatchKernel {
        shader: Id,
        bind_group: Id,
        push_constants: Vec<u8>,
        workgroup_dims: [u32; 3],
    },
    DispatchKernelIndirect {
        shader: Id,
        bind_group: Id,
        push_constants: Vec<u8>,
        indirect_buffer: Id,
        buffer_offset: u64,
        validate: bool,
    },
    /// Only for vulkan like synchronization
    SetEvent { event: Id, wait: SyncOperations },
    /// Only for vulkan like synchronization
    WaitEvent { event: Id, signal: SyncOperations },
    /// Only for vulkan like synchronization
    PipelineBarrier {
        before: SyncOperations,
        after: SyncOperations,
    },
    /// Only for vulkan like synchronization. Will hitch a ride with the previous PipelineBarrier or WaitEvent
    MemoryBarrier { resource: Id },
}

pub struct CommandStream {
    pub commands: Vec<HalCommandBuilder>,
    pub wait_semaphores: Vec<u32>,
}
pub struct CpuOperation<B: hal::Backend> {
    pub closure: UserBufferAccessClosure<B>,
    pub wait_semaphores: Vec<u32>,
    pub signal_semaphore: u32,
}
pub struct StreamingCommands<B: hal::Backend> {
    pub num_semaphores: u32,
    pub streams: Vec<CommandStream>,
    pub cpu_ops: Vec<CpuOperation<B>>,
}
pub fn assemble_dag<B: hal::Backend>(cr: &CommandRecorderInner<B>) -> CommandDag<B> {
    todo!()
}
pub fn dag_to_command_streams<B: hal::Backend>(dag: &CommandDag<B>) -> StreamingCommands<B> {
    todo!()
}
