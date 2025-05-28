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
use supasim_hal::{BackendInstance, BufferCommand, CommandRecorder, RecorderSubmitInfo};
use types::BufferDescriptor;

pub fn main() {
    unsafe {
        let mut instance = supasim_hal::Vulkan::create_instance(true).unwrap();
        let buffer1 = instance
            .create_buffer(&BufferDescriptor {
                size: 16,
                memory_type: types::BufferType::Upload,
                visible_to_renderer: false,
                indirect_capable: false,
                uniform: false,
                needs_flush: false,
            })
            .unwrap();
        let buffer2 = instance
            .create_buffer(&BufferDescriptor {
                size: 16,
                memory_type: types::BufferType::Storage,
                visible_to_renderer: false,
                indirect_capable: false,
                uniform: false,
                needs_flush: false,
            })
            .unwrap();
        let buffer3 = instance
            .create_buffer(&BufferDescriptor {
                size: 16,
                memory_type: types::BufferType::Download,
                visible_to_renderer: false,
                indirect_capable: false,
                uniform: false,
                needs_flush: false,
            })
            .unwrap();
        instance
            .write_buffer(&buffer1, 0, bytemuck::cast_slice(&[1u32, 2, 3, 4]))
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
                        resource: supasim_hal::GpuResource::Buffer {
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
            .read_buffer(&buffer3, 0, bytemuck::cast_slice_mut(&mut data))
            .unwrap();
        assert!(data == [1, 2, 3, 4,]);
    }
}
