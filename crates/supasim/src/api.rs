#![allow(dead_code)]
// This file will probably be deleted eventually

use std::ops::{Deref, DerefMut, Range};

use thiserror::Error;
use types::InstanceProperties;

#[derive(Error, Debug)]
pub enum SupaSimError {
    #[error("Error compiling shaders: {0}")]
    ShaderCompileError(#[from] shaders::ShaderCompileError),
}
pub type SupaSimResult<T> = Result<T, Box<SupaSimError>>;

pub struct ComputeDispatchInfo<'a, B: Backend> {
    pub buffers: &'a [B::Buffer],
}
pub trait Backend: Sized {
    type Instance: BackendInstance<Self>;
    type Kernel: CompiledKernel<Self>;
    type Buffer: Buffer<Self>;
    type MappedBuffer: MappedBuffer<Self>;
    type WaitHandle: WaitHandle<Self>;
    type CommandRecorder: CommandRecorder<Self>;
    type PipelineCache: PipelineCache<Self>;
}
pub trait BackendInstance<B: Backend>: Clone {
    fn properties(&self) -> InstanceProperties;
    fn compile_kernel(
        &self,
        binary: &[u8],
        reflection: shaders::ShaderReflectionInfo,
    ) -> SupaSimResult<B::Kernel>;
    fn create_pipeline_cache(&self, data: &[u8]) -> SupaSimResult<B::PipelineCache>;
    fn create_recorder(&self) -> SupaSimResult<B::CommandRecorder>;
    fn create_buffer(&self, alloc_info: &types::BufferDescriptor) -> SupaSimResult<B::Buffer>;
    fn submit_commands(&self, recorders: &[&B::CommandRecorder]) -> SupaSimResult<B::WaitHandle>;
    fn wait(&self, wait_handles: &[B::WaitHandle], wait_for_all: bool) -> SupaSimResult<()>;
    /// Wait for all compute work to complete on the GPU.
    fn wait_for_idle(&self) -> SupaSimResult<()>;
    /// Do all work that might take time, such as building yet unused compute pipelines. Useful in benchmarking.
    fn do_busywork(&self) -> SupaSimResult<()>;
    fn destroy(self) -> SupaSimResult<()>;
}
#[derive(Clone)]
pub struct BufferView<B: Backend> {
    buffer: B::Buffer,
    range: Range<u64>,
}
pub trait WaitHandle<B: Backend>: Clone + std::ops::Add {
    fn destroy(self) -> SupaSimResult<()>;
    fn has_happened(&self) -> SupaSimResult<()>;
}
pub trait CommandRecorder<B: Backend>: Clone {
    fn clear(&self) -> SupaSimResult<()>;
    fn dispatch_kernel(
        &self,
        shader: B::Kernel,
        dispatch_info: &ComputeDispatchInfo<B>,
        workgroup_dims: [u32; 3],
        return_wait: bool,
    ) -> SupaSimResult<Option<B::WaitHandle>>;
    #[allow(clippy::type_complexity)]
    fn cpu_code(
        &self,
        closure: Box<dyn Fn(&[B::MappedBuffer]) -> anyhow::Result<()>>,
        buffers: &[&B::Buffer],
        return_wait: bool,
    ) -> SupaSimResult<Option<B::WaitHandle>>;
    fn recorder_wait_handle(&self) -> SupaSimResult<B::WaitHandle>;
    fn destroy(self) -> SupaSimResult<()>;
}
pub trait CompiledKernel<B: Backend>: Clone {
    fn destroy(self) -> SupaSimResult<()>;
}
pub trait MappedBuffer<B: Backend>: Clone + std::io::Read + std::io::Write {
    type ReadLock: Deref<Target = [u8]>;
    type WriteLock: DerefMut<Target = [u8]>;
    fn read(&self) -> SupaSimResult<Self::ReadLock>;
    fn write(&mut self) -> SupaSimResult<Self::WriteLock>;
}
pub trait Buffer<B: Backend>: Clone {
    fn resize(&self, new_size: u64, force_resize: bool) -> SupaSimResult<()>;
    fn destroy(self) -> SupaSimResult<()>;
}

pub trait RecorderIndirectExt<B: Backend>: Clone + CommandRecorder<B> {
    #[allow(clippy::too_many_arguments)]
    fn dispatch_kernel_indirect(
        &self,
        shader: B::Kernel,
        dispatch_info: ComputeDispatchInfo<B>,
        indirect_buffer: &B::Buffer,
        buffer_offset: u64,
        num_dispatches: u64,
        validate_dispatches: bool,
        return_wait: bool,
    ) -> SupaSimResult<Option<B::WaitHandle>>;
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
        return_wait: bool,
    ) -> SupaSimResult<Option<B::WaitHandle>>;
}
pub trait PipelineCache<B: Backend>: Clone {
    fn get_data(&self) -> Vec<u8>;
    fn destroy(self) -> SupaSimResult<()>;
}
