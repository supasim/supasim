#![allow(unused_variables, dead_code)]

use crate::*;
use ::wgpu;

#[derive(Clone, Copy, Debug)]
pub struct Wgpu;
impl Backend for Wgpu {
    type Instance = WgpuInstance;

    type Kernel = WgpuKernel;

    type Buffer = WgpuBuffer;

    type CommandRecorder = WgpuCommandRecorder;

    type BindGroup = WgpuBindGroup;

    type KernelCache = WgpuKernelCache;

    type Semaphore = WgpuSemaphore;

    type Event = WgpuEvent;

    type Error = WgpuError;
}
pub struct WgpuInstance {
    inner: wgpu::Device,
}
impl BackendInstance<Wgpu> for WgpuInstance {
    fn get_properties(&mut self) -> InstanceProperties {
        todo!()
    }

    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::ShaderReflectionInfo,
        cache: Option<&mut <Wgpu as Backend>::KernelCache>,
    ) -> Result<<Wgpu as Backend>::Kernel, <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn create_pipeline_cache(
        &mut self,
        initial_data: &[u8],
    ) -> Result<<Wgpu as Backend>::KernelCache, <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn destroy_pipeline_cache(
        &mut self,
        cache: <Wgpu as Backend>::KernelCache,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn get_pipeline_cache_data(
        &mut self,
        cache: &mut <Wgpu as Backend>::KernelCache,
    ) -> Result<Vec<u8>, <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn destroy_kernel(
        &mut self,
        kernel: <Wgpu as Backend>::Kernel,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn wait_for_idle(&mut self) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn create_recorder(
        &mut self,
        allow_resubmits: bool,
    ) -> Result<<Wgpu as Backend>::CommandRecorder, <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn destroy_recorder(
        &mut self,
        recorder: <Wgpu as Backend>::CommandRecorder,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn clear_recorders(
        &mut self,
        buffers: &mut [&mut <Wgpu as Backend>::CommandRecorder],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn create_buffer(
        &mut self,
        alloc_info: &BufferDescriptor,
    ) -> Result<<Wgpu as Backend>::Buffer, <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn destroy_buffer(
        &mut self,
        buffer: <Wgpu as Backend>::Buffer,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn write_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn read_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn map_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
        offset: u64,
        size: u64,
    ) -> Result<*mut u8, <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn unmap_buffer(
        &mut self,
        buffer: &<Wgpu as Backend>::Buffer,
        map: *mut u8,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn create_bind_group(
        &mut self,
        kernel: &mut <Wgpu as Backend>::Kernel,
        resources: &[GpuResource<Wgpu>],
    ) -> Result<<Wgpu as Backend>::BindGroup, <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn update_bind_group(
        &mut self,
        bg: &mut <Wgpu as Backend>::BindGroup,
        kernel: &mut <Wgpu as Backend>::Kernel,
        resources: &[GpuResource<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn destroy_bind_group(
        &mut self,
        kernel: &mut <Wgpu as Backend>::Kernel,
        bind_group: <Wgpu as Backend>::BindGroup,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn create_semaphore(
        &mut self,
    ) -> Result<<Wgpu as Backend>::Semaphore, <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn destroy_semaphore(
        &mut self,
        semaphore: <Wgpu as Backend>::Semaphore,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn wait_for_semaphores(
        &mut self,
        semaphores: &[&<Wgpu as Backend>::Semaphore],
        all: bool,
        timeout: f32,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn create_event(
        &mut self,
    ) -> Result<<Wgpu as Backend>::Event, <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn destroy_event(
        &mut self,
        event: <Wgpu as Backend>::Event,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }

    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }
}
pub struct WgpuKernel {
    shader: wgpu::ShaderModule,
    pipeline: wgpu::ComputePipeline,
}
impl Kernel<Wgpu> for WgpuKernel {}
pub struct WgpuBuffer {
    inner: wgpu::Buffer,
}
impl Buffer<Wgpu> for WgpuBuffer {}
pub enum WgpuCommandRecorder {}
impl CommandRecorder<Wgpu> for WgpuCommandRecorder {
    unsafe fn record_commands(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
        commands: &mut [BufferCommand<Wgpu>],
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }
    unsafe fn record_dag(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
        resources: &[&GpuResource<Wgpu>],
        dag: &mut Dag<BufferCommand<Wgpu>>,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }
}
pub struct WgpuBindGroup {
    inner: wgpu::BindGroup,
}
impl BindGroup<Wgpu> for WgpuBindGroup {}
pub struct WgpuKernelCache {
    inner: wgpu::PipelineCache,
}
impl KernelCache<Wgpu> for WgpuKernelCache {}
pub struct WgpuSemaphore {
    inner: wgpu::SubmissionIndex,
}
impl Semaphore<Wgpu> for WgpuSemaphore {
    unsafe fn signal(
        &mut self,
        instance: &mut <Wgpu as Backend>::Instance,
    ) -> Result<(), <Wgpu as Backend>::Error> {
        todo!()
    }
}
pub struct WgpuEvent;
impl Event<Wgpu> for WgpuEvent {}
#[derive(thiserror::Error, Debug)]
pub enum WgpuError {}
impl Error<Wgpu> for WgpuError {
    fn is_out_of_device_memory(&self) -> bool {
        todo!()
    }
    fn is_out_of_host_memory(&self) -> bool {
        todo!()
    }
    fn is_timeout(&self) -> bool {
        todo!()
    }
}
