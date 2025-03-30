use crate::{
    Backend, BackendInstance, BufferCommand, CommandRecorder, GpuResource, RecorderSubmitInfo,
};
use log::info;
use types::{BufferDescriptor, ShaderReflectionInfo, ShaderResourceType, SyncOperations};

unsafe fn create_storage_buf<B: Backend>(
    instance: &mut B::Instance,
    data: &[u8],
) -> Result<B::Buffer, B::Error> {
    unsafe {
        let buf = instance.create_buffer(&BufferDescriptor {
            size: data.len() as u64,
            memory_type: types::MemoryType::UploadDownload,
            mapped_at_creation: false,
            visible_to_renderer: false,
            indirect_capable: false,
            transfer_src: false,
            transfer_dst: false,
            uniform: false,
            needs_flush: true,
        })?;
        instance.write_buffer(&buf, 0, data)?;
        Ok(buf)
    }
}
fn main_test<B: Backend>(mut instance: B::Instance, check_result: bool) -> Result<(), B::Error> {
    unsafe {
        info!("Starting test");
        let mut cache = instance.create_pipeline_cache(&[])?;
        let fun_semaphore = instance.create_semaphore()?;
        let mut kernel = instance.compile_kernel(
            include_bytes!("test_add.spirv"),
            &ShaderReflectionInfo {
                workgroup_size: [1, 1, 1],
                entry_name: "main".to_owned(),
                resources: vec![
                    ShaderResourceType::Buffer,
                    ShaderResourceType::Buffer,
                    ShaderResourceType::Buffer,
                ],
                push_constant_len: 0,
            },
            Some(&mut cache),
        )?;
        let mut kernel2 = instance.compile_kernel(
            include_bytes!("test_double.spirv"),
            &ShaderReflectionInfo {
                workgroup_size: [1, 1, 1],
                entry_name: "main".to_owned(),
                resources: vec![ShaderResourceType::Buffer],
                push_constant_len: 0,
            },
            Some(&mut cache),
        )?;
        let uniform_buf = instance.create_buffer(&BufferDescriptor {
            size: 16,
            memory_type: types::MemoryType::Upload,
            mapped_at_creation: false,
            visible_to_renderer: false,
            indirect_capable: false,
            transfer_src: false,
            transfer_dst: false,
            uniform: true,
            needs_flush: true,
        })?;
        let sb1 = create_storage_buf::<B>(&mut instance, bytemuck::bytes_of(&[5u32, 0, 0, 0]))?;
        let sb2 = create_storage_buf::<B>(&mut instance, bytemuck::bytes_of(&[8u32, 0, 0, 0]))?;
        let sbout = create_storage_buf::<B>(&mut instance, bytemuck::bytes_of(&[2u32, 0, 0, 0]))?;
        let bind_group = instance.create_bind_group(
            &mut kernel,
            &[
                GpuResource::buffer(&sb1, 0, 16),
                GpuResource::buffer(&sb2, 0, 16),
                GpuResource::buffer(&sbout, 0, 16),
            ],
        )?;
        let bind_group2 =
            instance.create_bind_group(&mut kernel2, &[GpuResource::buffer(&sbout, 0, 16)])?;

        let mut recorder = instance.create_recorder(false)?;

        info!("Created things");

        recorder.record_commands(
            &mut instance,
            &mut [
                BufferCommand::DispatchKernel {
                    shader: &kernel,
                    bind_group: &bind_group,
                    push_constants: &[],
                    workgroup_dims: [1, 1, 1],
                },
                BufferCommand::PipelineBarrier {
                    before: SyncOperations::ComputeDispatch,
                    after: SyncOperations::ComputeDispatch,
                },
                BufferCommand::MemoryBarrier {
                    resource: GpuResource::Buffer {
                        buffer: &sbout,
                        offset: 0,
                        size: 16,
                    },
                },
                BufferCommand::DispatchKernel {
                    shader: &kernel2,
                    bind_group: &bind_group2,
                    push_constants: &[],
                    workgroup_dims: [1, 1, 1],
                },
            ],
        )?;
        info!("Recorded commands");
        instance.submit_recorders(std::slice::from_mut(&mut RecorderSubmitInfo {
            command_recorder: &mut recorder,
            wait_semaphores: &mut [],
            signal_semaphores: &[&fun_semaphore],
        }))?;
        info!("Submitted recorders");
        instance.wait_for_semaphores(&[&fun_semaphore], true, 1.0)?;

        let mut res = [3u32, 0, 0, 0];
        instance.read_buffer(&sbout, 0, bytemuck::cast_slice_mut(&mut res))?;
        if check_result && res[0] != 26 {
            panic!("Expected 26, got {}", res[0]);
        }

        info!("Read buffers");

        instance.destroy_recorder(recorder)?;

        instance.destroy_semaphore(fun_semaphore)?;
        instance.destroy_bind_group(&mut kernel, bind_group)?;
        instance.destroy_bind_group(&mut kernel2, bind_group2)?;
        instance.destroy_kernel(kernel)?;
        instance.destroy_kernel(kernel2)?;
        instance.destroy_buffer(uniform_buf)?;
        instance.destroy_buffer(sb1)?;
        instance.destroy_buffer(sb2)?;
        instance.destroy_buffer(sbout)?;
        instance.destroy_pipeline_cache(cache)?;
        instance.destroy()?;
        info!("Destroyed");
        Ok(())
    }
}

#[cfg(feature = "vulkan")]
#[test]
pub fn vulkan_test() {
    use crate::vulkan::Vulkan;
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();
    info!("Vulkan test");
    let instance = Vulkan::create_instance(true).unwrap();
    info!("Created vulkan instance");
    main_test::<Vulkan>(instance, true).unwrap();
}
#[cfg(feature = "wgpu")]
#[test]
pub fn wgpu_test() {
    use crate::wgpu::Wgpu;
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();
    info!("Wgpu test");
    let instance = Wgpu::create_instance(true).unwrap();
    info!("Created wgpu instance");
    main_test::<Wgpu>(instance, true).unwrap();
}
#[test]
pub fn dummy_test() {
    use crate::dummy::Dummy;
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();
    info!("Dummy test");
    let instance = Dummy::create_instance();
    info!("Created dummy instance");
    main_test::<Dummy>(instance, false).unwrap();
}
