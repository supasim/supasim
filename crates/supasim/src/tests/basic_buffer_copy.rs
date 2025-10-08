/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use crate::*;

pub fn basic_buffer_copy<Backend: hal::Backend>(
    hal: hal::InstanceDescriptor<Backend>,
) -> Result<(), ()> {
    {
        // Test specific stuff
        println!("Hello, world!");
        dev_utils::setup_trace_printer_if_env();

        let instance = Instance::<Backend>::from_hal(hal);
        let upload_buffer = instance
            .create_buffer(&BufferDescriptor {
                size: 16,
                contents_align: 4,
                priority: 0.0,
                preferred_device_index: None,
            })
            .unwrap();
        let gpu_buffer = instance
            .create_buffer(&BufferDescriptor {
                size: 16,
                contents_align: 4,
                priority: 0.0,
                preferred_device_index: None,
            })
            .unwrap();
        let download_buffer = instance
            .create_buffer(&BufferDescriptor {
                size: 16,
                contents_align: 4,
                priority: 0.0,
                preferred_device_index: None,
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
    }
    Ok(())
}

dev_utils::all_backend_tests!(basic_buffer_copy);
