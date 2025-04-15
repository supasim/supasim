use std::sync::LazyLock;

use crate::{
    Backend, BackendInstance, BufferCommand, CommandRecorder, GpuResource, RecorderSubmitInfo,
    Semaphore,
};
use log::info;
use types::{BufferDescriptor, ShaderReflectionInfo, ShaderResourceType, SyncOperations};

unsafe fn create_storage_buf<B: Backend>(
    instance: &mut B::Instance,
    size: u64,
) -> Result<B::Buffer, B::Error> {
    unsafe {
        let buf = instance.create_buffer(&BufferDescriptor {
            size,
            memory_type: types::BufferType::Storage,
            visible_to_renderer: false,
            indirect_capable: false,
            uniform: false,
            needs_flush: true,
        })?;
        Ok(buf)
    }
}
fn main_test<B: Backend<Instance = I>, I: crate::BackendInstance<B>>(
    mut instance: I,
    check_result: bool,
) -> Result<(), B::Error> {
    unsafe {
        info!("Starting test");
        let mut cache = if instance.get_properties().pipeline_cache {
            Some(instance.create_kernel_cache(&[])?)
        } else {
            None
        };
        let mut fun_semaphore = instance.create_semaphore()?;
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
            cache.as_mut(),
        )?;
        let mut kernel2 = instance.compile_kernel(
            include_bytes!("test_double.spirv"),
            &ShaderReflectionInfo {
                workgroup_size: [1, 1, 1],
                entry_name: "main".to_owned(),
                resources: vec![ShaderResourceType::Buffer],
                push_constant_len: 0,
            },
            cache.as_mut(),
        )?;
        let upload_buffer = instance.create_buffer(&BufferDescriptor {
            size: 16,
            memory_type: types::BufferType::Upload,
            visible_to_renderer: false,
            indirect_capable: false,
            uniform: false,
            needs_flush: true,
        })?;
        let download_buffer = instance.create_buffer(&BufferDescriptor {
            size: 16,
            memory_type: types::BufferType::Download,
            visible_to_renderer: false,
            indirect_capable: false,
            uniform: false,
            needs_flush: true,
        })?;
        let uniform_buf = instance.create_buffer(&BufferDescriptor {
            size: 16,
            memory_type: types::BufferType::Other,
            visible_to_renderer: false,
            indirect_capable: false,
            uniform: true,
            needs_flush: true,
        })?;
        let sb1 = create_storage_buf::<B>(&mut instance, 16)?;
        let sb2 = create_storage_buf::<B>(&mut instance, 16)?;
        let sbout = create_storage_buf::<B>(&mut instance, 16)?;
        instance.write_buffer(
            &upload_buffer,
            0,
            bytemuck::cast_slice(&[5u32, 8u32, 2u32, 0]),
        )?;
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

        let mut recorder = instance.create_recorder()?;

        info!("Created things");

        recorder.record_commands(
            &mut instance,
            &mut [
                BufferCommand::CopyBuffer {
                    src_buffer: &upload_buffer,
                    dst_buffer: &sb1,
                    src_offset: 0,
                    dst_offset: 0,
                    size: 4,
                },
                BufferCommand::CopyBuffer {
                    src_buffer: &upload_buffer,
                    dst_buffer: &sb2,
                    src_offset: 4,
                    dst_offset: 0,
                    size: 4,
                },
                BufferCommand::CopyBuffer {
                    src_buffer: &upload_buffer,
                    dst_buffer: &sbout,
                    src_offset: 8,
                    dst_offset: 0,
                    size: 4,
                },
                BufferCommand::MemoryBarrier {
                    resource: GpuResource::Buffer {
                        buffer: &sb1,
                        offset: 0,
                        size: 16,
                    },
                },
                BufferCommand::MemoryBarrier {
                    resource: GpuResource::Buffer {
                        buffer: &sb2,
                        offset: 0,
                        size: 16,
                    },
                },
                BufferCommand::MemoryBarrier {
                    resource: GpuResource::Buffer {
                        buffer: &sbout,
                        offset: 0,
                        size: 16,
                    },
                },
                BufferCommand::PipelineBarrier {
                    before: SyncOperations::Transfer,
                    after: SyncOperations::ComputeDispatch,
                },
                BufferCommand::DispatchKernel {
                    kernel: &kernel,
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
                    kernel: &kernel2,
                    bind_group: &bind_group2,
                    push_constants: &[],
                    workgroup_dims: [1, 1, 1],
                },
                BufferCommand::MemoryBarrier {
                    resource: GpuResource::Buffer {
                        buffer: &sbout,
                        offset: 0,
                        size: 16,
                    },
                },
                BufferCommand::PipelineBarrier {
                    before: SyncOperations::ComputeDispatch,
                    after: SyncOperations::Transfer,
                },
                BufferCommand::CopyBuffer {
                    src_buffer: &sbout,
                    dst_buffer: &download_buffer,
                    src_offset: 0,
                    dst_offset: 0,
                    size: 16,
                },
            ],
        )?;
        info!("Recorded commands");
        instance.submit_recorders(std::slice::from_mut(&mut RecorderSubmitInfo {
            command_recorder: &mut recorder,
            wait_semaphore: None,
            signal_semaphore: Some(&fun_semaphore),
        }))?;
        info!("Submitted recorders");
        fun_semaphore.wait(&mut instance)?;

        let mut res = [3u32, 0, 0, 0];
        instance.read_buffer(&download_buffer, 0, bytemuck::cast_slice_mut(&mut res))?;
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
        instance.destroy_buffer(upload_buffer)?;
        instance.destroy_buffer(download_buffer)?;
        if let Some(cache) = cache {
            instance.destroy_kernel_cache(cache)?;
        }
        instance.destroy()?;
        info!("Destroyed");
        Ok(())
    }
}

static INSTANCE_CREATE_LOCK: LazyLock<std::sync::Mutex<()>> =
    LazyLock::new(|| std::sync::Mutex::new(()));

pub fn should_skip(test_backend: &str) -> bool {
    std::env::var(format!("SUPASIM_SKIP_BACKEND_{test_backend}"))
        .is_ok_and(|a| &a != "0" && &a != "false" && !a.is_empty())
}
macro_rules! gpu_test {
    ($func_name:ident, $backend_name:literal, $verify_result:literal, $instance_create:block) => {
        #[test]
        pub fn $func_name() {
            if should_skip($backend_name) {
                return;
            }
            let _ = env_logger::builder()
                .filter_level(log::LevelFilter::Info)
                .try_init();
            info!("{} test", $backend_name);
            let _lock = INSTANCE_CREATE_LOCK.lock().unwrap();
            let instance = $instance_create;
            drop(_lock);
            let instance = instance.expect(&format!("Failed to create {} instance", $backend_name));
            info!("Created {} instance", $backend_name);
            main_test(instance, $verify_result).unwrap();
        }
    };
}
gpu_test!(dummy_test, "DUMMY", false, {
    crate::Dummy::create_instance()
});
#[cfg(feature = "vulkan")]
gpu_test!(vulkan_test, "VULKAN", true, {
    crate::Vulkan::create_instance(true)
});
#[cfg(feature = "wgpu")]
gpu_test!(wgpu_vulkan_test, "WGPU_VULKAN", true, {
    crate::wgpu::Wgpu::create_instance(true, wgpu::Backends::VULKAN, None)
});
#[cfg(feature = "wgpu")]
#[cfg(target_vendor = "apple")]
gpu_test!(wgpu_metal_test, "WGPU_METAL", true, {
    crate::wgpu::Wgpu::create_instance(true, wgpu::Backends::METAL)
});
#[cfg(feature = "wgpu")]
#[cfg(target_os = "windows")]
gpu_test!(wgpu_dx12_test, "WGPU_DX12", true, {
    crate::wgpu::Wgpu::create_instance(true, wgpu::Backends::DX12)
});
/*#[cfg(feature = "wgpu")]
#[test]
pub fn wgpu_test() {
    use crate::wgpu::Wgpu;
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .try_init();
    info!("Wgpu test");
    let _lock = INSTANCE_CREATE_LOCK.lock().unwrap();
    for backend in [
        (wgpu::Backend::Vulkan, "VULKAN"),
        #[cfg(target_vendor = "apple")]
        (wgpu::Backend::Metal, "METAL"),
        #[cfg(target_os = "windows")]
        (wgpu::Backend::Dx12, "DX12"),
    ] {

    }
    let instance = Wgpu::create_instance(true).unwrap();
    drop(_lock);
    if std::env::var("SUPASIM_WGPU_TEST")
    info!("Created wgpu instance");
    main_test::<Wgpu>(instance, true).unwrap();
}*/
