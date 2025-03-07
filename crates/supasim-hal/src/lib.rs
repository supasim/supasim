#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(feature = "vulkan")]
pub use vulkan::{
    Vulkan, VulkanBindGroup, VulkanBuffer, VulkanCommandRecorder, VulkanError, VulkanFence,
    VulkanInstance, VulkanKernel, VulkanMappedBuffer, VulkanPipelineCache, VulkanSemaphore,
};

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

    type Error: Error<Self>;
}
pub trait BackendInstance<B: Backend<Instance = Self>> {
    fn get_properties(&mut self) -> InstanceProperties;
    fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &shaders::ShaderReflectionInfo,
        cache: Option<&mut B::PipelineCache>,
    ) -> Result<B::Kernel, B::Error>;
    fn create_pipeline_cache(&mut self, initial_data: &[u8]) -> Result<B::PipelineCache, B::Error>;
    fn destroy_pipeline_cache(&mut self, cache: B::PipelineCache) -> Result<(), B::Error>;
    fn get_pipeline_cache_data(&mut self, cache: B::PipelineCache) -> Result<Vec<u8>, B::Error>;
    fn destroy_kernel(&mut self, kernel: B::Kernel) -> Result<(), B::Error>;
    /// Wait for all compute work to complete on the GPU.
    fn wait_for_idle(&mut self) -> Result<(), B::Error>;
    fn create_recorder(&mut self, allow_resubmis: bool) -> Result<B::CommandRecorder, B::Error>;
    fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<B>],
        fence: Option<&mut B::Fence>,
    ) -> Result<(), B::Error>;
    fn destroy_recorder(&mut self, recorder: B::CommandRecorder) -> Result<(), B::Error>;
    fn clear_recorders(&mut self, buffers: &mut [&mut B::CommandRecorder]) -> Result<(), B::Error>;
    fn create_buffer(&mut self, alloc_info: &BufferDescriptor) -> Result<B::Buffer, B::Error>;
    fn destroy_buffer(&mut self, buffer: B::Buffer) -> Result<(), B::Error>;
    fn write_buffer(
        &mut self,
        buffer: &B::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), B::Error>;
    fn map_buffer(
        &mut self,
        buffer: &B::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<B::MappedBuffer, B::Error>;
    fn flush_mapped_buffer(
        &self,
        buffer: &B::Buffer,
        map: &B::MappedBuffer,
    ) -> Result<(), B::Error>;
    fn update_mapped_buffer(
        &self,
        buffer: &B::Buffer,
        map: &B::MappedBuffer,
    ) -> Result<(), B::Error>;
    fn unmap_buffer(&mut self, buffer: &B::Buffer, map: B::MappedBuffer) -> Result<(), B::Error>;
    fn create_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        resources: &mut [GpuResource<B>],
    ) -> Result<B::BindGroup, B::Error>;
    fn destroy_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        bind_group: B::BindGroup,
    ) -> Result<(), B::Error>;

    fn create_fence(&mut self) -> Result<B::Fence, B::Error>;
    fn destroy_fence(&mut self, fence: B::Fence) -> Result<(), B::Error>;
    fn wait_for_fences(
        &mut self,
        fences: &mut [&mut B::Fence],
        all: bool,
        timeout_seconds: f32,
    ) -> Result<(), B::Error>;

    fn create_semaphore(&mut self, timeline: bool) -> Result<B::Semaphore, B::Error>;
    fn destroy_semaphore(&mut self, semaphore: B::Semaphore) -> Result<(), B::Error>;
    fn wait_for_semaphores(
        &mut self,
        semaphores: &mut [(&mut B::Semaphore, u64)],
        all: bool,
        timeout: f32,
    ) -> Result<(), B::Error>;

    fn cleanup_cached_resources(&mut self) -> Result<(), B::Error>;
}
pub enum GpuResource<'a, B: Backend> {
    Buffer {
        buffer: &'a B::Buffer,
        offset: u64,
        size: u64,
    },
}
impl<'a, B: Backend> GpuResource<'a, B> {
    pub fn buffer(buffer: &'a B::Buffer, offset: u64, size: u64) -> Self {
        Self::Buffer {
            buffer,
            offset,
            size,
        }
    }
}
pub struct CommandSynchronization<'a, B: Backend> {
    pub resources_needing_sync: &'a mut [&'a mut GpuResource<'a, B>],
    pub out_semaphore: Option<(&'a mut B::Semaphore, u64)>,
}
pub trait CommandRecorder<B: Backend<CommandRecorder = Self>> {
    fn record_dag(&mut self, dag: &mut daggy::Dag<GpuOperation<B>, ()>) -> Result<(), B::Error>;
    fn record_commands(&mut self, commands: &mut [GpuOperation<B>]) -> Result<(), B::Error>;
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
pub trait Fence<B: Backend<Fence = Self>> {
    fn reset(&mut self, instance: &mut B::Instance) -> Result<(), B::Error>;
    fn get_signalled(&mut self, instance: &mut B::Instance) -> Result<bool, B::Error>;
}
pub trait Semaphore<B: Backend<Semaphore = Self>> {
    fn get_timeline_counter(&mut self, instance: &mut B::Instance) -> Result<u64, B::Error>;
    fn signal(&mut self, instance: &mut B::Instance, signal: u64) -> Result<(), B::Error>;
}

pub struct RecorderSubmitInfo<'a, B: Backend> {
    pub command_recorder: &'a mut B::CommandRecorder,
    pub wait_semaphores: &'a mut [(&'a mut B::Semaphore, u64)],
    pub out_semaphores: &'a mut [(&'a mut B::Semaphore, u64)],
}
#[must_use]
pub trait Error<B: Backend<Error = Self>>: std::error::Error {
    fn is_out_of_device_memory(&self) -> bool;
    fn is_out_of_host_memory(&self) -> bool;
    fn is_timeout(&self) -> bool;
}

pub struct GpuOperation<'a, B: Backend> {
    pub command: GpuCommand<'a, B>,
    pub sync: CommandSynchronization<'a, B>,
    pub validate_indirect: bool,
}
pub enum GpuCommand<'a, B: Backend> {
    CopyBuffer {
        src_buffer: &'a B::Buffer,
        dst_buffer: &'a B::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
    },
    DispatchKernel {
        shader: &'a B::Kernel,
        bind_group: &'a B::BindGroup,
        push_constants: &'a [u8],
        workgroup_dims: [u32; 3],
    },
    DispatchKernelIndirect {
        shader: &'a B::Kernel,
        bind_group: &'a B::BindGroup,
        push_constants: &'a [u8],
        indirect_buffer: &'a B::Buffer,
        buffer_offset: u64,
        num_dispatches: u64,
    },
}
