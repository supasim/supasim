#![allow(unused_variables)]

use crate::*;

#[derive(Clone, Debug)]
pub struct Dummy;
impl Backend for Dummy {
    type Instance = DummyResource;
    type Kernel = DummyResource;
    type Buffer = DummyResource;
    type CommandRecorder = DummyResource;
    type KernelCache = DummyResource;
    type BindGroup = DummyResource;
    type Semaphore = DummyResource;
    type Event = DummyResource;
    type Error = DummyResource;
}
impl Dummy {
    pub fn create_instance() -> DummyResource {
        DummyResource
    }
}
#[derive(Clone, Debug)]
pub struct DummyResource;
impl BackendInstance<Dummy> for DummyResource {
    fn get_properties(&mut self) -> types::InstanceProperties {
        InstanceProperties {
            sync_mode: SyncMode::Automatic,
            indirect: true,
            pipeline_cache: true,
            shader_type: ShaderTarget::Spirv {
                version: SpirvVersion::V1_0,
            },
            easily_update_bind_groups: true,
        }
    }

    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::ShaderReflectionInfo,
        cache: Option<&mut <Dummy as Backend>::KernelCache>,
    ) -> Result<<Dummy as Backend>::Kernel, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn create_kernel_cache(
        &mut self,
        initial_data: &[u8],
    ) -> Result<<Dummy as Backend>::KernelCache, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn destroy_kernel_cache(
        &mut self,
        cache: <Dummy as Backend>::KernelCache,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn get_kernel_cache_data(
        &mut self,
        cache: &mut <Dummy as Backend>::KernelCache,
    ) -> Result<Vec<u8>, <Dummy as Backend>::Error> {
        Ok(Vec::new())
    }

    unsafe fn destroy_kernel(
        &mut self,
        kernel: <Dummy as Backend>::Kernel,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn wait_for_idle(&mut self) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn create_recorder(
        &mut self,
    ) -> Result<<Dummy as Backend>::CommandRecorder, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [crate::RecorderSubmitInfo<Dummy>],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn destroy_recorder(
        &mut self,
        recorder: <Dummy as Backend>::CommandRecorder,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn clear_recorders(
        &mut self,
        buffers: &mut [&mut <Dummy as Backend>::CommandRecorder],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn create_buffer(
        &mut self,
        alloc_info: &types::BufferDescriptor,
    ) -> Result<<Dummy as Backend>::Buffer, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn destroy_buffer(
        &mut self,
        buffer: <Dummy as Backend>::Buffer,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn write_buffer(
        &mut self,
        buffer: &<Dummy as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn read_buffer(
        &mut self,
        buffer: &<Dummy as Backend>::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn map_buffer(
        &mut self,
        buffer: &<Dummy as Backend>::Buffer,
    ) -> Result<*mut u8, <Dummy as Backend>::Error> {
        Ok(std::ptr::null_mut())
    }

    unsafe fn unmap_buffer(
        &mut self,
        buffer: &<Dummy as Backend>::Buffer,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn create_bind_group(
        &mut self,
        kernel: &mut <Dummy as Backend>::Kernel,
        resources: &[crate::GpuResource<Dummy>],
    ) -> Result<<Dummy as Backend>::BindGroup, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn update_bind_group(
        &mut self,
        bg: &mut <Dummy as Backend>::BindGroup,
        kernel: &mut <Dummy as Backend>::Kernel,
        resources: &[crate::GpuResource<Dummy>],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn destroy_bind_group(
        &mut self,
        kernel: &mut <Dummy as Backend>::Kernel,
        bind_group: <Dummy as Backend>::BindGroup,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn create_semaphore(
        &mut self,
    ) -> Result<<Dummy as Backend>::Semaphore, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn destroy_semaphore(
        &mut self,
        semaphore: <Dummy as Backend>::Semaphore,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn create_event(
        &mut self,
    ) -> Result<<Dummy as Backend>::Event, <Dummy as Backend>::Error> {
        unreachable!()
    }

    unsafe fn destroy_event(
        &mut self,
        event: <Dummy as Backend>::Event,
    ) -> Result<(), <Dummy as Backend>::Error> {
        unreachable!()
    }

    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
    unsafe fn destroy(self) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
}
impl Kernel<Dummy> for DummyResource {}
impl Buffer<Dummy> for DummyResource {}
impl CommandRecorder<Dummy> for DummyResource {
    unsafe fn record_commands(
        &mut self,
        instance: &mut <Dummy as Backend>::Instance,
        commands: &mut [crate::BufferCommand<Dummy>],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
    unsafe fn record_dag(
        &mut self,
        instance: &mut <Dummy as Backend>::Instance,
        resources: &[&crate::GpuResource<Dummy>],
        dag: &mut types::Dag<crate::BufferCommand<Dummy>>,
    ) -> Result<(), <Dummy as Backend>::Error> {
        unreachable!()
    }
}
impl KernelCache<Dummy> for DummyResource {}
impl BindGroup<Dummy> for DummyResource {}
impl Semaphore<Dummy> for DummyResource {
    unsafe fn wait(
        &mut self,
        instance: &mut <Dummy as Backend>::Instance,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
    unsafe fn is_signalled(
        &mut self,
        instance: &mut <Dummy as Backend>::Instance,
    ) -> Result<bool, <Dummy as Backend>::Error> {
        Ok(false)
    }
}
impl Event<Dummy> for DummyResource {}
impl Error<Dummy> for DummyResource {
    // Error will never be constructed
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
