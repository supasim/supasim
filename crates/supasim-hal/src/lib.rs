#[cfg(feature = "vulkan")]
mod vulkan;
#[cfg(feature = "vulkan")]
pub use vulkan::{
    Vulkan, VulkanBindGroup, VulkanBuffer, VulkanCommandRecorder, VulkanError, VulkanFence,
    VulkanInstance, VulkanKernel, VulkanPipelineCache, VulkanSemaphore,
};

use types::*;

pub trait Backend: Sized + std::fmt::Debug + Clone {
    type Instance: BackendInstance<Self>;
    type Kernel: CompiledKernel<Self>;
    type Buffer: Buffer<Self>;
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
        reflection: &types::ShaderReflectionInfo,
        cache: Option<&mut B::PipelineCache>,
    ) -> Result<B::Kernel, B::Error>;
    fn create_pipeline_cache(&mut self, initial_data: &[u8]) -> Result<B::PipelineCache, B::Error>;
    fn destroy_pipeline_cache(&mut self, cache: B::PipelineCache) -> Result<(), B::Error>;
    fn get_pipeline_cache_data(
        &mut self,
        cache: &mut B::PipelineCache,
    ) -> Result<Vec<u8>, B::Error>;
    fn destroy_kernel(&mut self, kernel: B::Kernel) -> Result<(), B::Error>;
    /// Wait for all compute work to complete on the GPU.
    fn wait_for_idle(&mut self) -> Result<(), B::Error>;
    fn create_recorder(&mut self, allow_resubmits: bool) -> Result<B::CommandRecorder, B::Error>;
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
    fn read_buffer(
        &mut self,
        buffer: &B::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), B::Error>;
    fn map_buffer(
        &mut self,
        buffer: &B::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<*mut u8, B::Error>;
    fn unmap_buffer(&mut self, buffer: &B::Buffer, map: *mut u8) -> Result<(), B::Error>;
    fn create_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        resources: &[GpuResource<B>],
    ) -> Result<B::BindGroup, B::Error>;
    fn update_bind_group(
        &mut self,
        bg: &mut B::BindGroup,
        kernel: &mut B::Kernel,
        resources: &[GpuResource<B>],
    ) -> Result<(), B::Error>;
    fn destroy_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        bind_group: B::BindGroup,
    ) -> Result<(), B::Error>;

    fn create_fence(&mut self) -> Result<B::Fence, B::Error>;
    fn destroy_fence(&mut self, fence: B::Fence) -> Result<(), B::Error>;
    fn wait_for_fences(
        &mut self,
        fences: &[&B::Fence],
        all: bool,
        timeout_seconds: f32,
    ) -> Result<(), B::Error>;

    fn create_semaphore(&mut self) -> Result<B::Semaphore, B::Error>;
    fn destroy_semaphore(&mut self, semaphore: B::Semaphore) -> Result<(), B::Error>;
    fn wait_for_semaphores(
        &mut self,
        semaphores: &[&B::Semaphore],
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
    /// Edges contain the start and length in resources buffer of used resources
    fn record_dag(
        &mut self,
        instance: &mut B::Instance,
        resources: &[&GpuResource<B>],
        dag: &mut daggy::Dag<GpuOperation<B>, (usize, usize)>,
    ) -> Result<(), B::Error>;
    fn record_commands(
        &mut self,
        instance: &mut B::Instance,
        commands: &mut [GpuOperation<B>],
    ) -> Result<(), B::Error>;
}
pub trait CompiledKernel<B: Backend<Kernel = Self>> {}
pub trait Buffer<B: Backend<Buffer = Self>> {}
pub trait BindGroup<B: Backend<BindGroup = Self>> {}
pub trait PipelineCache<B: Backend<PipelineCache = Self>> {}
/// Only used for when an entire submit completes
pub trait Fence<B: Backend<Fence = Self>> {
    fn reset(&mut self, instance: &mut B::Instance) -> Result<(), B::Error>;
    fn get_signalled(&mut self, instance: &mut B::Instance) -> Result<bool, B::Error>;
}
pub trait Semaphore<B: Backend<Semaphore = Self>> {
    fn signal(&mut self, instance: &mut B::Instance) -> Result<(), B::Error>;
}

pub struct RecorderSubmitInfo<'a, B: Backend> {
    pub command_recorder: &'a mut B::CommandRecorder,
    pub wait_semaphores: &'a [&'a B::Semaphore],
    pub signal_semaphores: &'a [&'a B::Semaphore],
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
    },
}
