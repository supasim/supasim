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
use std::any::TypeId;
use std::sync::LazyLock;

use crate::{self as hal, HalBufferSlice};
use crate::{
    Backend, BackendInstance, BufferCommand, CommandRecorder, RecorderSubmitInfo, Semaphore,
};
use kernels::KernelCompileOptions;
use log::info;
use types::{HalBufferDescriptor, SyncOperations};

unsafe fn create_storage_buf<B: Backend>(
    instance: &mut B::Instance,
    size: u64,
) -> Result<B::Buffer, B::Error> {
    unsafe {
        let buf = instance.create_buffer(&HalBufferDescriptor {
            size,
            memory_type: types::HalBufferType::Storage,
            min_alignment: 16,
            can_export: false,
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
        let fun_semaphore = instance.create_semaphore()?;
        let kernel_compiler = kernels::GlobalState::new_from_env().unwrap();
        let mut add_code = Vec::new();
        let mut double_code = Vec::new();
        info!("Compiling kernels into code");
        let add_reflection = kernel_compiler
            .compile_kernel(KernelCompileOptions {
                target: instance.get_properties().kernel_lang,
                source: kernels::KernelSource::Memory(include_bytes!(
                    "../../../../kernels/test_add.slang"
                )),
                dest: kernels::KernelDest::Memory(&mut add_code),
                entry: "add",
                include: None,
                fp_mode: kernels::KernelFpMode::Precise,
                opt_level: kernels::OptimizationLevel::Maximal,
                stability: kernels::StabilityGuarantee::ExtraValidation,
                minify: false,
            })
            .unwrap();
        let double_reflection = kernel_compiler
            .compile_kernel(KernelCompileOptions {
                target: instance.get_properties().kernel_lang,
                source: kernels::KernelSource::Memory(include_bytes!(
                    "../../../../kernels/test_double.slang"
                )),
                dest: kernels::KernelDest::Memory(&mut double_code),
                entry: "double",
                include: None,
                fp_mode: kernels::KernelFpMode::Precise,
                opt_level: kernels::OptimizationLevel::Maximal,
                stability: kernels::StabilityGuarantee::ExtraValidation,
                minify: false,
            })
            .unwrap();
        assert_eq!(add_reflection.buffers, vec![false, false, true]);
        assert_eq!(double_reflection.buffers, vec![true]);
        drop(kernel_compiler);
        info!("Constructing kernel objects");
        let mut kernel = instance.compile_kernel(&add_code, &add_reflection, cache.as_mut())?;
        let mut kernel2 =
            instance.compile_kernel(&double_code, &double_reflection, cache.as_mut())?;
        info!("Kernels compiled");
        let mut upload_buffer = instance.create_buffer(&HalBufferDescriptor {
            size: 16,
            memory_type: types::HalBufferType::Upload,
            min_alignment: 16,
            can_export: false,
        })?;
        let mut download_buffer = instance.create_buffer(&HalBufferDescriptor {
            size: 16,
            memory_type: types::HalBufferType::Download,
            min_alignment: 16,
            can_export: false,
        })?;
        let sb1 = create_storage_buf::<B>(&mut instance, 16)?;
        let sb2 = create_storage_buf::<B>(&mut instance, 16)?;
        let sbout = create_storage_buf::<B>(&mut instance, 16)?;
        instance.write_buffer(
            &mut upload_buffer,
            0,
            bytemuck::cast_slice(&[5u32, 8u32, 2u32, 0]),
        )?;
        let bind_group = instance.create_bind_group(
            &mut kernel,
            &[
                HalBufferSlice {
                    buffer: &sb1,
                    offset: 0,
                    len: 16,
                },
                HalBufferSlice {
                    buffer: &sb2,
                    offset: 0,
                    len: 16,
                },
                HalBufferSlice {
                    buffer: &sbout,
                    offset: 0,
                    len: 16,
                },
            ],
        )?;
        let bind_group2 = instance.create_bind_group(
            &mut kernel2,
            &[HalBufferSlice {
                buffer: &sbout,
                offset: 0,
                len: 16,
            }],
        )?;

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
                    buffer: HalBufferSlice {
                        buffer: &sb1,
                        offset: 0,
                        len: 16,
                    },
                },
                BufferCommand::MemoryBarrier {
                    buffer: HalBufferSlice {
                        buffer: &sb2,
                        offset: 0,
                        len: 16,
                    },
                },
                BufferCommand::MemoryBarrier {
                    buffer: HalBufferSlice {
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
                    buffer: HalBufferSlice {
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
                    buffer: HalBufferSlice {
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
            wait_semaphore: None,
            signal_semaphore: Some(&fun_semaphore),
        }))?;
        info!("Submitted recorders");
        fun_semaphore.wait()?;

        let mut res = [3u32, 0, 0, 0];
        instance.read_buffer(&mut download_buffer, 0, bytemuck::cast_slice_mut(&mut res))?;
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
