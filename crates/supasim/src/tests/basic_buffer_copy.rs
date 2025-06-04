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
use crate::*;
use std::any::TypeId;

pub fn basic_buffer_copy<Backend: hal::Backend>(hal: Backend::Instance) -> Result<(), ()> {
    // Dummy test won't be necessary here
    if TypeId::of::<Backend>() == TypeId::of::<hal::Dummy>() {
        return Ok(());
    }
    println!("Hello, world!");
    dev_utils::setup_trace_printer_if_env();
    let instance = SupaSimInstance::<Backend>::from_hal(hal);
    let buffer1 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: BufferType::Upload,
            contents_align: 4,
            priority: 0.0,
        })
        .unwrap();
    let buffer2 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: BufferType::Storage,
            contents_align: 4,
            priority: 0.0,
        })
        .unwrap();
    let buffer3 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: BufferType::Download,
            contents_align: 4,
            priority: 0.0,
        })
        .unwrap();
    let slices = [
        &BufferSlice::entire_buffer(&buffer1, true).unwrap(),
        &BufferSlice::entire_buffer(&buffer3, false).unwrap(),
    ];
    instance
        .access_buffers(
            Box::new(|buffers| {
                buffers[0]
                    .writeable::<u32>()
                    .unwrap()
                    .clone_from_slice(&[1, 2, 3, 4]);
                Ok(())
            }),
            &slices[..],
        )
        .unwrap();
    let slices = [
        &BufferSlice::entire_buffer(&buffer1, false).unwrap(),
        &BufferSlice::entire_buffer(&buffer3, false).unwrap(),
    ];
    let recorder = instance.create_recorder().unwrap();
    recorder
        .copy_buffer(buffer1.clone(), buffer2.clone(), 0, 0, 16)
        .unwrap();
    recorder
        .copy_buffer(buffer2.clone(), buffer3.clone(), 0, 0, 16)
        .unwrap();
    instance.submit_commands(&mut [recorder]).unwrap();
    instance.wait_for_idle(1.0).unwrap();
    instance
        .access_buffers(
            Box::new(|buffers| {
                println!("{:?}", buffers[0].readable::<u32>().unwrap());
                println!("{:?}", buffers[1].readable::<u32>().unwrap());
                assert_eq!(buffers[1].readable::<u32>().unwrap(), [1, 2, 3, 4]);
                Ok(())
            }),
            &slices[..],
        )
        .unwrap();
    Ok(())
}

dev_utils::all_backend_tests!(basic_buffer_copy);
