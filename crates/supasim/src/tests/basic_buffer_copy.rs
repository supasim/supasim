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
    if TypeId::of::<Backend>() == TypeId::of::<hal::Dummy>() {
        // We don't want to "test" on the dummy backend, where nothing happens so the result will be wrong
        return Ok(());
    }
    println!("Hello, world!");
    dev_utils::setup_trace_printer_if_env();
    let instance = SupaSimInstance::<Backend>::from_hal(hal);
    let upload_buffer = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: BufferType::Upload,
            contents_align: 4,
            priority: 0.0,
        })
        .unwrap();
    let gpu_buffer = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: BufferType::Gpu,
            contents_align: 4,
            priority: 0.0,
        })
        .unwrap();
    let download_buffer = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: BufferType::Download,
            contents_align: 4,
            priority: 0.0,
        })
        .unwrap();
    upload_buffer.write::<u32>(0, &[1, 2, 3, 4]).unwrap();
    let recorder = instance.create_recorder().unwrap();
    recorder
        .copy_buffer(&upload_buffer, &gpu_buffer, 0, 0, 16)
        .unwrap();
    recorder
        .copy_buffer(&gpu_buffer, &download_buffer, 0, 0, 16)
        .unwrap();
    instance.submit_commands(&mut [recorder]).unwrap();
    instance.wait_for_idle(1.0).unwrap();
    assert_eq!(
        download_buffer
            .access(0, 16, false)
            .unwrap()
            .readable::<u32>()
            .unwrap(),
        [1, 2, 3, 4]
    );
    Ok(())
}

dev_utils::all_backend_tests!(basic_buffer_copy);
