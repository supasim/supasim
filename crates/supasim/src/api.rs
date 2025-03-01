#![allow(dead_code)]
// This file will probably be deleted eventually

use std::ops::{Deref, DerefMut};

use thiserror::Error;

#[derive(Error, Debug)]
pub enum SupaSimError {
    #[error("Error compiling shaders: {0}")]
    ShaderCompileError(#[from] shaders::ShaderCompileError),
}
pub type SupaSimResult<T> = Result<T, Box<SupaSimError>>;

pub struct ComputeDispatchInfo<B: Backend> {
    pub resources: Vec<B::DynResource>,
}

pub trait Backend: Sized {
    type Instance: BackendInstance<Self>;
    type Kernel: CompiledKernel<Self>;
    type Buffer: Buffer<Self>;
    type MappedBuffer: MappedBuffer<Self>;
    type DynResource: GpuResource<Self>;
    type WaitHandle: WaitHandle<Self>;
    type CommandRecorder: CommandRecorder<Self>;
}
pub trait BackendInstance<B: Backend>: Clone {
    fn shader_format(&self) -> types::ShaderTarget;
    /// Whether or not the system should wait until the last moment to do certain actions. May allow more efficient bundling/memory sharing/certain optimizations
    fn set_lazy(&self, lazy: bool) -> SupaSimResult<()>;
    fn compile_kernel(
        &self,
        binary: &[u8],
        reflection: shaders::ShaderReflectionInfo,
    ) -> SupaSimResult<B::Kernel>;
    fn destroy_kernel(&self, kernel: B::Kernel) -> SupaSimResult<()>;
    /// Wait for all compute work to complete on the GPU.
    fn wait_for_idle(&self) -> SupaSimResult<()>;
    /// Do all work that might take time, such as building yet unused compute pipelines. Useful in benchmarking.
    fn do_busywork(&self) -> SupaSimResult<()>;
    fn create_recorder(&self) -> SupaSimResult<B::CommandRecorder>;
    fn submit_commands(&self, recorder: &B::CommandRecorder) -> SupaSimResult<B::WaitHandle>;
}
pub trait GpuResource<B: Backend>: Clone {
    fn as_buffer(&self) -> Option<&B::Buffer>;
}
pub trait WaitHandle<B: Backend>: Clone + std::ops::Add {
    fn wait(&self);
}
pub trait CommandRecorder<B: Backend>: Clone + CommandRecordContext<B> {
    fn clear(&self) -> SupaSimResult<()>;
    fn dispatch_kernel(
        &self,
        shader: B::Kernel,
        dispatch_info: &ComputeDispatchInfo<B>,
        workgroup_dims: [u32; 3],
    ) -> SupaSimResult<B::WaitHandle>;
    #[allow(clippy::type_complexity)]
    fn cpu_code(
        &self,
        closure: Box<dyn Fn(&[B::MappedBuffer]) -> anyhow::Result<()>>,
        buffers: &[&B::Buffer],
    ) -> SupaSimResult<B::WaitHandle>;
    fn create_buffer(&self, alloc_info: &types::BufferDescriptor) -> SupaSimResult<B::Buffer>;
    fn destroy_buffer(&self, buffer: B::Buffer) -> SupaSimResult<()>;
    fn create_subrecorder(&self, wait: Option<B::WaitHandle>) -> SupaSimResult<B::SubRecorder>;
    fn recorder_wait_handle(&self) -> SupaSimResult<B::WaitHandle>;
    fn map_buffer(&self, buffer: &B::Buffer);
    fn unmap_buffer(&self, buffer: &B::Buffer) -> SupaSimResult<()>;
}
pub trait CompiledKernel<B: Backend>: Clone {}
pub trait MappedBuffer<B: Backend>: Clone + std::io::Read + std::io::Write {
    type ReadLock: Deref<Target = [u8]>;
    type WriteLock: DerefMut<Target = [u8]>;
    fn read(&self) -> SupaSimResult<Self::ReadLock>;
    fn write(&mut self) -> SupaSimResult<Self::WriteLock>;
}
pub trait Buffer<B: Backend>: GpuResource<B> + Clone {
    fn resize(&self, new_size: u64, force_resize: bool) -> SupaSimResult<()>;
    fn get_mapped(&self) -> SupaSimResult<B::MappedBuffer>;
}

pub trait RecorderIndirectExt<
    B: Backend<CommandRecorder: RecorderIndirectExt<B>, SubRecorder: RecorderIndirectExt<B>>,
>: Clone + CommandRecordContext<B>
{
    fn dispatch_kernel_indirect(
        &self,
        shader: B::Kernel,
        dispatch_info: ComputeDispatchInfo<B>,
        indirect_buffer: &B::Buffer,
        buffer_offset: u64,
        num_dispatches: u64,
        validate_dispatches: bool,
    );
    #[allow(clippy::too_many_arguments)]
    fn dispatch_kernel_indirect_count(
        &self,
        shader: B::Kernel,
        dispatch_info: ComputeDispatchInfo<B>,
        indirect_buffer: &B::Buffer,
        buffer_offset: u64,
        count_buffer: &B::Buffer,
        count_offset: u64,
        max_dispatches: u64,
        validate_dispatches: bool,
    );
}
