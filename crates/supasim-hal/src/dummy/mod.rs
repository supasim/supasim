#![allow(unused_variables)]

use crate::*;

#[derive(Clone, Debug)]
pub struct DummyBackend;
impl Backend for DummyBackend {
    type Instance = DummyResource;
    type Kernel = DummyResource;
    type Buffer = DummyResource;
    type CommandRecorder = DummyResource;
    type PipelineCache = DummyResource;
    type BindGroup = DummyResource;
    type Semaphore = DummyResource;
    type Event = DummyResource;
    type Error = DummyResource;
}
#[derive(Clone, Debug)]
pub struct DummyResource;
impl BackendInstance<DummyBackend> for DummyResource {
    fn get_properties(&mut self) -> types::InstanceProperties {
        unreachable!()
    }

    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::ShaderReflectionInfo,
        cache: Option<&mut <DummyBackend as Backend>::PipelineCache>,
    ) -> Result<<DummyBackend as Backend>::Kernel, <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn create_pipeline_cache(
        &mut self,
        initial_data: &[u8],
    ) -> Result<<DummyBackend as Backend>::PipelineCache, <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn destroy_pipeline_cache(
        &mut self,
        cache: <DummyBackend as Backend>::PipelineCache,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn get_pipeline_cache_data(
        &mut self,
        cache: &mut <DummyBackend as Backend>::PipelineCache,
    ) -> Result<Vec<u8>, <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn destroy_kernel(
        &mut self,
        kernel: <DummyBackend as Backend>::Kernel,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn wait_for_idle(&mut self) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn create_recorder(
        &mut self,
        allow_resubmits: bool,
    ) -> Result<<DummyBackend as Backend>::CommandRecorder, <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [crate::RecorderSubmitInfo<DummyBackend>],
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn destroy_recorder(
        &mut self,
        recorder: <DummyBackend as Backend>::CommandRecorder,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn clear_recorders(
        &mut self,
        buffers: &mut [&mut <DummyBackend as Backend>::CommandRecorder],
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn create_buffer(
        &mut self,
        alloc_info: &types::BufferDescriptor,
    ) -> Result<<DummyBackend as Backend>::Buffer, <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn destroy_buffer(
        &mut self,
        buffer: <DummyBackend as Backend>::Buffer,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn write_buffer(
        &mut self,
        buffer: &<DummyBackend as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn read_buffer(
        &mut self,
        buffer: &<DummyBackend as Backend>::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn map_buffer(
        &mut self,
        buffer: &<DummyBackend as Backend>::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<*mut u8, <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn unmap_buffer(
        &mut self,
        buffer: &<DummyBackend as Backend>::Buffer,
        map: *mut u8,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn create_bind_group(
        &mut self,
        kernel: &mut <DummyBackend as Backend>::Kernel,
        resources: &[crate::GpuResource<DummyBackend>],
    ) -> Result<<DummyBackend as Backend>::BindGroup, <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn update_bind_group(
        &mut self,
        bg: &mut <DummyBackend as Backend>::BindGroup,
        kernel: &mut <DummyBackend as Backend>::Kernel,
        resources: &[crate::GpuResource<DummyBackend>],
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn destroy_bind_group(
        &mut self,
        kernel: &mut <DummyBackend as Backend>::Kernel,
        bind_group: <DummyBackend as Backend>::BindGroup,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn create_semaphore(
        &mut self,
    ) -> Result<<DummyBackend as Backend>::Semaphore, <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn destroy_semaphore(
        &mut self,
        semaphore: <DummyBackend as Backend>::Semaphore,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn wait_for_semaphores(
        &mut self,
        semaphores: &[&<DummyBackend as Backend>::Semaphore],
        all: bool,
        timeout: f32,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn create_event(
        &mut self,
    ) -> Result<<DummyBackend as Backend>::Event, <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn destroy_event(
        &mut self,
        event: <DummyBackend as Backend>::Event,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }

    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }
}
impl CompiledKernel<DummyBackend> for DummyResource {}
impl Buffer<DummyBackend> for DummyResource {}
impl CommandRecorder<DummyBackend> for DummyResource {
    unsafe fn record_commands(
        &mut self,
        instance: &mut <DummyBackend as Backend>::Instance,
        commands: &mut [crate::BufferCommand<DummyBackend>],
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }
    unsafe fn record_dag(
        &mut self,
        instance: &mut <DummyBackend as Backend>::Instance,
        resources: &[&crate::GpuResource<DummyBackend>],
        dag: &mut types::Dag<crate::BufferCommand<DummyBackend>>,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }
}
impl PipelineCache<DummyBackend> for DummyResource {}
impl BindGroup<DummyBackend> for DummyResource {}
impl Semaphore<DummyBackend> for DummyResource {
    unsafe fn signal(
        &mut self,
        instance: &mut <DummyBackend as Backend>::Instance,
    ) -> Result<(), <DummyBackend as Backend>::Error> {
        unreachable!()
    }
}
impl Event<DummyBackend> for DummyResource {}
impl Error<DummyBackend> for DummyResource {
    fn is_out_of_device_memory(&self) -> bool {
        unreachable!()
    }
    fn is_out_of_host_memory(&self) -> bool {
        unreachable!()
    }
    fn is_timeout(&self) -> bool {
        unreachable!()
    }
}
impl std::fmt::Display for DummyResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unreachable!()
    }
}
impl std::error::Error for DummyResource {}
