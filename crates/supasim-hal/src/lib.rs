#[cfg(feature = "vulkan")]
mod vulkan;

use types::*;

pub trait Backend: Sized {
    type Instance: BackendInstance<Self>;
    type Kernel: CompiledKernel<Self>;
    type Buffer: Buffer<Self>;
    type MappedBuffer: MappedBuffer<Self>;
    type CommandRecorder: CommandRecorder<Self>;
    type BindGroup: BindGroup<Self>;
    type PipelineCache: PipelineCache<Self>;
    type Fence: Fence<Self>;
    type Semaphore: Semaphore<Self>;

    type Error: std::error::Error;
}
pub trait BackendInstance<B: Backend<Instance = Self>> {
    fn get_properties(&mut self) -> InstanceProperties;

    fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: shaders::ShaderReflectionInfo,
        cache: Option<&mut B::PipelineCache>,
    ) -> Result<B::Kernel, B::Error>;
    fn create_pipeline_cache(&mut self, initial_data: &[u8]) -> Result<B::PipelineCache, B::Error>;
    fn destroy_pipeline_cache(&mut self, cache: B::PipelineCache) -> Result<(), B::Error>;
    fn get_pipeline_cache_data(&mut self, cache: B::PipelineCache) -> Result<Vec<u8>, B::Error>;
    fn destroy_kernel(&mut self, kernel: B::Kernel) -> Result<(), B::Error>;
    /// Wait for all compute work to complete on the GPU.
    fn wait_for_idle(&mut self) -> Result<(), B::Error>;
    fn create_recorders(&mut self, num: u32) -> Result<Vec<B::CommandRecorder>, B::Error>;
    fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<B>],
        fence: Option<&mut B::Fence>,
    ) -> Result<(), B::Error>;
    fn destroy_recorders(&mut self, recorders: Vec<B::CommandRecorder>) -> Result<(), B::Error>;
    fn clear_recorders(&mut self, buffers: &mut [&mut B::CommandRecorder]) -> Result<(), B::Error>;
    fn create_buffer(&mut self, alloc_info: &BufferDescriptor) -> Result<B::Buffer, B::Error>;
    fn destroy_buffer(&mut self, buffer: B::Buffer) -> Result<(), B::Error>;
    fn write_buffer(
        &mut self,
        buffer: &mut B::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), B::Error>;
    fn map_buffer(
        &mut self,
        buffer: &mut B::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<B::MappedBuffer, B::Error>;
    fn flush_mapped_buffer(
        &self,
        buffer: &mut B::Buffer,
        map: &mut B::MappedBuffer,
    ) -> Result<(), B::Error>;
    fn unmap_buffer(
        &mut self,
        buffer: &mut B::Buffer,
        map: B::MappedBuffer,
    ) -> Result<(), B::Error>;
    fn create_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        resources: &mut [&mut GpuResource<B>],
    ) -> Result<B::BindGroup, B::Error>;
    fn destroy_bind_groups(
        &mut self,
        kernel: &mut B::Kernel,
        bind_groups: &mut [&mut B::BindGroup],
    ) -> Result<(), B::Error>;

    fn create_fences(&mut self, num: u32) -> Result<Vec<B::Fence>, B::Error>;
    fn destroy_fences(&mut self, fences: Vec<B::Fence>) -> Result<(), B::Error>;
    fn wait_for_fences(
        &mut self,
        fences: &mut [&mut B::Fence],
        all: bool,
        timeout_seconds: f32,
    ) -> Result<(), B::Error>;

    fn create_semaphores(
        &mut self,
        timeline: bool,
        num: u32,
    ) -> Result<Vec<B::Semaphore>, B::Error>;
    fn destroy_semaphores(&mut self, semaphores: Vec<B::Semaphore>) -> Result<(), B::Error>;
    fn wait_for_semaphores(
        &mut self,
        semaphores: &mut [&mut B::Semaphore],
        timeout: f32,
    ) -> Result<(), B::Error>;
    fn get_timeline_semaphore_counter(
        &mut self,
        semaphore: &mut B::Semaphore,
    ) -> Result<u64, B::Error>;
}
pub enum GpuResource<'a, B: Backend> {
    Buffer(&'a mut B::Buffer),
}
pub struct CommandSynchronization<'a, B: Backend> {
    waits: &'a mut [&'a mut B::Semaphore],
    resources: &'a mut [&'a mut GpuResource<'a, B>],
    out_fence: Option<&'a mut B::Fence>,
    out_semaphore: Option<&'a mut B::Semaphore>,
}
pub trait CommandRecorder<B: Backend<CommandRecorder = Self>> {
    fn copy_buffer(
        &mut self,
        instance: &mut B::Instance,
        src_buffer: &mut B::Buffer,
        dst_buffer: &mut B::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
        sync: CommandSynchronization<B>,
    ) -> Result<(), B::Error>;
    #[allow(clippy::too_many_arguments)]
    fn dispatch_kernel(
        &mut self,
        instance: &mut B::Instance,
        shader: &mut B::Kernel,
        bind_group: &mut B::BindGroup,
        push_constants: &[u8],
        workgroup_dims: [u32; 3],
        sync: CommandSynchronization<B>,
    ) -> Result<(), B::Error>;
    #[allow(clippy::too_many_arguments)]
    fn dispatch_kernel_indirect(
        &mut self,
        instance: &mut B::Instance,
        shader: &mut B::Kernel,
        bind_group: &mut B::BindGroup,
        push_constants: &[u8],
        indirect_buffer: &mut B::Buffer,
        buffer_offset: u64,
        num_dispatches: u64,
        validate_dispatches: bool,
        sync: CommandSynchronization<B>,
    ) -> Result<(), B::Error>;
    #[allow(clippy::too_many_arguments)]
    fn dispatch_kernel_indirect_count(
        &mut self,
        instance: &mut B::Instance,
        shader: &mut B::Kernel,
        bind_group: &mut B::BindGroup,
        push_constants: &[u8],
        indirect_buffer: &mut B::Buffer,
        buffer_offset: u64,
        count_buffer: &mut B::Buffer,
        count_offset: u64,
        max_dispatches: u64,
        validate_dispatches: bool,
        sync: CommandSynchronization<B>,
    ) -> Result<(), B::Error>;
}
pub trait CompiledKernel<B: Backend<Kernel = Self>> {}
pub trait MappedBuffer<B: Backend<MappedBuffer = Self>> {
    fn readable(&mut self) -> &[u8];
    fn writable(&mut self) -> &mut [u8];
}
pub trait Buffer<B: Backend<Buffer = Self>> {}
pub trait BindGroup<B: Backend<BindGroup = Self>> {}
pub trait PipelineCache<B: Backend<PipelineCache = Self>> {}
/// Only used for when an entire submit completes
pub trait Fence<B: Backend<Fence = Self>> {}
pub trait Semaphore<B: Backend<Semaphore = Self>> {}

pub struct RecorderSubmitInfo<'a, B: Backend> {
    pub command_recorders: &'a mut [&'a mut B::CommandRecorder],
    pub wait_semaphores: &'a mut [&'a mut B::Semaphore],
    pub out_semaphores: &'a mut [&'a mut B::Semaphore],
}
