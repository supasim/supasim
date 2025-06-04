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
use supasim_hal::{Backend, BackendInstance, BufferCommand, CommandRecorder, RecorderSubmitInfo};
use types::HalBufferDescriptor;

pub fn example<B: Backend>(mut instance: B::Instance) {
    unsafe {
        let mut buffer1 = instance
            .create_buffer(&HalBufferDescriptor {
                size: 16,
                memory_type: types::HalBufferType::Upload,
                visible_to_renderer: false,
                indirect_capable: false,
                uniform: false,
                needs_flush: false,
            })
            .unwrap();
        let buffer2 = instance
            .create_buffer(&HalBufferDescriptor {
                size: 16,
                memory_type: types::HalBufferType::Storage,
                visible_to_renderer: false,
                indirect_capable: false,
                uniform: false,
                needs_flush: false,
            })
            .unwrap();
        let mut buffer3 = instance
            .create_buffer(&HalBufferDescriptor {
                size: 16,
                memory_type: types::HalBufferType::Download,
                visible_to_renderer: false,
                indirect_capable: false,
                uniform: false,
                needs_flush: false,
            })
            .unwrap();
        instance
            .write_buffer(&mut buffer1, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))
            .unwrap();
        let mut recorder = instance.create_recorder().unwrap();
        recorder
            .record_commands(
                &mut instance,
                &mut [
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
        instance
            .submit_recorders(&mut [RecorderSubmitInfo {
                command_recorder: &mut recorder,
                wait_semaphores: &mut [],
                signal_semaphore: None,
            }])
            .unwrap();
        instance.wait_for_idle().unwrap();
        let mut data = [0u32; 4];
        instance
            .read_buffer(&mut buffer3, 0, bytemuck::cast_slice_mut(&mut data))
            .unwrap();
        assert!(data == [1, 2, 3, 4,]);
    }
}
pub fn main() {
    let instance = supasim_hal::Wgpu::create_instance(true, wgpu::Backends::PRIMARY, None).unwrap();
    example::<supasim_hal::Wgpu>(instance);
}
