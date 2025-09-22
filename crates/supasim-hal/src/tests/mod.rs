/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

#![allow(unused)]

use std::sync::LazyLock;

use crate::{self as hal, BindGroup, Buffer, Device, HalBufferSlice, Kernel};
use crate::{
    Backend, BackendInstance, BufferCommand, CommandRecorder, RecorderSubmitInfo, Semaphore, Stream,
};
use kernels::KernelCompileOptions;
use log::info;
use types::{HalBufferDescriptor, SyncOperations};

unsafe fn create_storage_buf<B: Backend>(
    device: &mut B::Device,
    size: u64,
) -> Result<B::Buffer, B::Error> {
    unsafe {
        let buf = device.create_buffer(&HalBufferDescriptor {
            size,
            memory_type: types::HalBufferType::Storage,
            min_alignment: 16,
        })?;
        Ok(buf)
    }
}

static INSTANCE_CREATE_LOCK: LazyLock<std::sync::Mutex<()>> =
    LazyLock::new(|| std::sync::Mutex::new(()));

fn hal_comprehensive<B: Backend>(descriptor: crate::InstanceDescriptor<B>) -> Result<(), B::Error> {
    let mut instance = descriptor.instance;
    let device = descriptor.devices.into_iter().next().unwrap();
    let crate::StreamDescriptor {
        mut stream,
        stream_type: crate::StreamType::ComputeAndTransfer,
    } = device.streams.into_iter().next().unwrap()
    else {
        panic!("Expected compute and transfer queue to be first")
    };

    let mut device = device.device;
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

        let kernel = instance.compile_kernel(super::KernelDescriptor {
            reflection: add_reflection,
            binary: &add_code,
        })?;
        let kernel2 = instance.compile_kernel(super::KernelDescriptor {
            reflection: double_reflection,
            binary: &double_code,
        })?;
        info!("Kernels compiled");
        let mut upload_buffer = device.create_buffer(&HalBufferDescriptor {
            size: 16,
            memory_type: types::HalBufferType::Upload,
            min_alignment: 16,
        })?;
        let mut download_buffer = device.create_buffer(&HalBufferDescriptor {
            size: 16,
            memory_type: types::HalBufferType::Download,
            min_alignment: 16,
        })?;
        let sb1 = create_storage_buf::<B>(&mut device, 16)?;
        let sb2 = create_storage_buf::<B>(&mut device, 16)?;
        let sbout = create_storage_buf::<B>(&mut device, 16)?;
        upload_buffer.write(&device, 0, bytemuck::cast_slice(&[5u32, 8u32, 2u32, 0]))?;
        let bind_group = stream.create_bind_group(
            &device,
            &kernel,
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
        let bind_group2 = stream.create_bind_group(
            &device,
            &kernel2,
            &[HalBufferSlice {
                buffer: &sbout,
                offset: 0,
                len: 16,
            }],
        )?;

        let mut recorder = stream.create_recorder()?;

        info!("Created things");

        recorder.record_commands(
            &stream,
            &[
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
        stream.submit_recorders(std::slice::from_mut(&mut RecorderSubmitInfo {
            command_recorder: &mut recorder,
            wait_semaphores: &[],
            signal_semaphore: Some(&fun_semaphore),
        }))?;
        info!("Submitted recorders");
        fun_semaphore.wait(&instance)?;

        let mut res = [3u32, 0, 0, 0];
        download_buffer.read(&device, 0, bytemuck::cast_slice_mut(&mut res))?;
        assert_eq!(res[0], 26);

        info!("Read buffers");

        recorder.destroy(&stream)?;
        fun_semaphore.destroy(&instance)?;
        bind_group.destroy(&stream, &kernel)?;
        bind_group2.destroy(&stream, &kernel2)?;
        kernel.destroy(&instance)?;
        kernel2.destroy(&instance)?;
        sb1.destroy(&device)?;
        sb2.destroy(&device)?;
        sbout.destroy(&device)?;
        upload_buffer.destroy(&device)?;
        download_buffer.destroy(&device)?;
        stream.destroy(&mut device)?;
        device.destroy(&mut instance)?;
        instance.destroy()?;
        info!("Destroyed");
        Ok(())
    }
}

dev_utils::all_backend_tests!(hal_comprehensive);
