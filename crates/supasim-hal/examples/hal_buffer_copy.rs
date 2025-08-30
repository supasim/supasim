/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use supasim_hal::{
    Backend, BackendInstance, Buffer, BufferCommand, CommandRecorder, Device, InstanceDescriptor,
    RecorderSubmitInfo, Stream, StreamDescriptor, StreamType,
};
use types::HalBufferDescriptor;

pub fn example<B: Backend>(desc: InstanceDescriptor<B>) {
    let mut instance = desc.instance;
    let device = desc.devices.into_iter().next().unwrap();
    let StreamDescriptor {
        mut stream,
        stream_type: StreamType::ComputeAndTransfer,
    } = device.streams.into_iter().next().unwrap()
    else {
        panic!("Expected compute and transfer queue to be first")
    };

    let mut device = device.device;
    unsafe {
        let mut buffer1 = device
            .create_buffer(&HalBufferDescriptor {
                size: 16,
                memory_type: types::HalBufferType::Upload,
                min_alignment: 16,
            })
            .unwrap();
        let buffer2 = device
            .create_buffer(&HalBufferDescriptor {
                size: 16,
                memory_type: types::HalBufferType::Storage,
                min_alignment: 16,
            })
            .unwrap();
        let mut buffer3 = device
            .create_buffer(&HalBufferDescriptor {
                size: 16,
                memory_type: types::HalBufferType::Download,
                min_alignment: 16,
            })
            .unwrap();
        buffer1
            .write(&device, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))
            .unwrap();
        let mut recorder = stream.create_recorder().unwrap();
        recorder
            .record_commands(
                &stream,
                &[
                    BufferCommand::CopyBuffer {
                        src_buffer: &buffer1,
                        dst_buffer: &buffer2,
                        src_offset: 0,
                        dst_offset: 0,
                        len: 16,
                    },
                    BufferCommand::PipelineBarrier {
                        before: types::SyncOperations::Transfer,
                        after: types::SyncOperations::Transfer,
                    },
                    BufferCommand::MemoryBarrier {
                        buffer: supasim_hal::HalBufferSlice {
                            buffer: &buffer2,
                            offset: 0,
                            len: 16,
                        },
                    },
                    BufferCommand::CopyBuffer {
                        src_buffer: &buffer2,
                        dst_buffer: &buffer3,
                        src_offset: 0,
                        dst_offset: 0,
                        len: 16,
                    },
                ],
            )
            .unwrap();
        stream
            .submit_recorders(&mut [RecorderSubmitInfo {
                command_recorder: &mut recorder,
                wait_semaphores: &[],
                signal_semaphore: None,
            }])
            .unwrap();
        stream.wait_for_idle().unwrap();
        let mut data = [0u32; 4];
        buffer3
            .read(&device, 0, bytemuck::cast_slice_mut(&mut data))
            .unwrap();
        assert!(data == [1, 2, 3, 4]);

        recorder.destroy(&stream).unwrap();
        buffer1.destroy(&device).unwrap();
        buffer2.destroy(&device).unwrap();
        buffer3.destroy(&device).unwrap();
        stream.destroy(&mut device).unwrap();
        device.destroy(&mut instance).unwrap();
        instance.destroy().unwrap();
    }
}
pub fn main() {
    let instance = supasim_hal::Wgpu::setup_default_descriptor().unwrap();
    example::<supasim_hal::Wgpu>(instance);
}
