use std::any::TypeId;
/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 SupaMaggie70 (Magnus Larsson)


  SupaSim is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 3
  of the License, or (at your option) any later version.

  SupaSim is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
END LICENSE */
use std::sync::LazyLock;

use crate as hal;
use crate::{
    Backend, BackendInstance, BufferCommand, CommandRecorder, GpuResource, RecorderSubmitInfo,
    Semaphore,
};
use log::info;
use shaders::ShaderCompileOptions;
use types::{HalBufferDescriptor, SyncOperations};

unsafe fn create_storage_buf<B: Backend>(
    instance: &mut B::Instance,
    size: u64,
) -> Result<B::Buffer, B::Error> {
    unsafe {
        let buf = instance.create_buffer(&HalBufferDescriptor {
            size,
            memory_type: types::HalBufferType::Storage,
            visible_to_renderer: false,
            indirect_capable: false,
            uniform: false,
            needs_flush: true,
        })?;
        Ok(buf)
    }
}

static INSTANCE_CREATE_LOCK: LazyLock<std::sync::Mutex<()>> =
    LazyLock::new(|| std::sync::Mutex::new(()));

fn hal_comprehensive<B: Backend>(mut instance: B::Instance) -> Result<(), B::Error> {
    unsafe {
        let _lock = if let Ok(lock) = INSTANCE_CREATE_LOCK.lock() {
            lock
        } else {
            return Ok(());
        };
        let _ = env_logger::builder()
            .filter_level(log::LevelFilter::Info)
            .try_init();
        dev_utils::setup_trace_printer_if_env();
        info!("Starting test");
        let mut cache = if instance.get_properties().pipeline_cache {
            Some(instance.create_kernel_cache(&[])?)
        } else {
            None
        };
        let mut fun_semaphore = instance.create_semaphore()?;
        let shader_compiler = shaders::GlobalState::new_from_env().unwrap();
        let mut add_code = Vec::new();
        let mut double_code = Vec::new();
        info!("Compiling kernels");
        let add_reflection = shader_compiler
            .compile_shader(ShaderCompileOptions {
                target: instance.get_properties().shader_type,
                source: shaders::ShaderSource::Memory(include_bytes!(
                    "../../../../kernels/test_add.slang"
                )),
                dest: shaders::ShaderDest::Memory(&mut add_code),
                entry: "add",
                include: None,
                fp_mode: shaders::ShaderFpMode::Precise,
                opt_level: shaders::OptimizationLevel::Maximal,
                stability: shaders::StabilityGuarantee::ExtraValidation,
                minify: false,
            })
            .unwrap();
        let double_reflection = shader_compiler
            .compile_shader(ShaderCompileOptions {
                target: instance.get_properties().shader_type,
                source: shaders::ShaderSource::Memory(include_bytes!(
                    "../../../../kernels/test_double.slang"
                )),
                dest: shaders::ShaderDest::Memory(&mut double_code),
                entry: "double",
                include: None,
                fp_mode: shaders::ShaderFpMode::Precise,
                opt_level: shaders::OptimizationLevel::Maximal,
                stability: shaders::StabilityGuarantee::ExtraValidation,
                minify: false,
            })
            .unwrap();
        drop(shader_compiler);
        let mut kernel = instance.compile_kernel(&add_code, &add_reflection, cache.as_mut())?;
        let mut kernel2 =
            instance.compile_kernel(&double_code, &double_reflection, cache.as_mut())?;
        info!("Kernels compiled");
        let upload_buffer = instance.create_buffer(&HalBufferDescriptor {
            size: 16,
            memory_type: types::HalBufferType::Upload,
            visible_to_renderer: false,
            indirect_capable: false,
            uniform: false,
            needs_flush: true,
        })?;
        let download_buffer = instance.create_buffer(&HalBufferDescriptor {
            size: 16,
            memory_type: types::HalBufferType::Download,
            visible_to_renderer: false,
            indirect_capable: false,
            uniform: false,
            needs_flush: true,
        })?;
        let uniform_buf = instance.create_buffer(&HalBufferDescriptor {
            size: 16,
            memory_type: types::HalBufferType::Other,
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
                    len: 4,
                },
                BufferCommand::CopyBuffer {
                    src_buffer: &upload_buffer,
                    dst_buffer: &sb2,
                    src_offset: 4,
                    dst_offset: 0,
                    len: 4,
                },
                BufferCommand::CopyBuffer {
                    src_buffer: &upload_buffer,
                    dst_buffer: &sbout,
                    src_offset: 8,
                    dst_offset: 0,
                    len: 4,
                },
                BufferCommand::MemoryBarrier {
                    resource: GpuResource::Buffer {
                        buffer: &sb1,
                        offset: 0,
                        len: 16,
                    },
                },
                BufferCommand::MemoryBarrier {
                    resource: GpuResource::Buffer {
                        buffer: &sb2,
                        offset: 0,
                        len: 16,
                    },
                },
                BufferCommand::MemoryBarrier {
                    resource: GpuResource::Buffer {
                        buffer: &sbout,
                        offset: 0,
                        len: 16,
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
                        len: 16,
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
                        len: 16,
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
                    len: 16,
                },
            ],
        )?;
        info!("Recorded commands");
        instance.submit_recorders(std::slice::from_mut(&mut RecorderSubmitInfo {
            command_recorder: &mut recorder,
            wait_semaphores: &mut [],
            signal_semaphore: Some(&fun_semaphore),
        }))?;
        info!("Submitted recorders");
        fun_semaphore.wait(&mut instance)?;

        let mut res = [3u32, 0, 0, 0];
        instance.read_buffer(&download_buffer, 0, bytemuck::cast_slice_mut(&mut res))?;
        if TypeId::of::<B>() != TypeId::of::<crate::Dummy>() {
            assert_eq!(res[0], 26);
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

dev_utils::all_backend_tests!(hal_comprehensive);
