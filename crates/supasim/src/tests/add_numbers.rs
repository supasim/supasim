/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use crate::*;
pub fn add_numbers<Backend: hal::Backend>(hal: hal::InstanceDescriptor<Backend>) -> Result<(), ()> {
    // Test specific stuff
    println!("Hello, world!");
    dev_utils::setup_trace_printer_if_env();

    // Create the instance
    let instance: Instance<Backend> = Instance::from_hal(hal);

    // Create the buffers
    let download_buffer = instance
        .create_buffer(&crate::BufferDescriptor {
            size: 16,
            buffer_type: BufferType::Download,
            contents_align: 4,
            ..Default::default()
        })
        .unwrap();
    let buffer1 = instance
        .create_buffer(&crate::BufferDescriptor {
            size: 16,
            contents_align: 4,
            ..Default::default()
        })
        .unwrap();
    let buffer2 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            contents_align: 4,
            ..Default::default()
        })
        .unwrap();
    let buffer3 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            contents_align: 4,
            ..Default::default()
        })
        .unwrap();

    // Compile the kernels
    let global_state = kernels::GlobalState::new_from_env().unwrap();
    let mut spirv = Vec::new();
    let reflection_info = global_state
        .compile_kernel(kernels::KernelCompileOptions {
            target: instance.properties().unwrap().kernel_lang,
            source: kernels::KernelSource::Memory(include_bytes!(
                "../../../../kernels/test_add.slang"
            )),
            dest: kernels::KernelDest::Memory(&mut spirv),
            entry: "add",
            include: None,
            fp_mode: kernels::KernelFpMode::Precise,
            opt_level: kernels::OptimizationLevel::Maximal,
            stability: kernels::StabilityGuarantee::ExtraValidation,
            minify: false,
        })
        .unwrap();
    assert!(reflection_info.buffers == vec![false, false, true]);
    let kernel = instance
        .compile_raw_kernel(&spirv, reflection_info)
        .unwrap();

    // Record and submit commands
    //
    // Command summary:
    // * Copy data from upload buffer into 3 used buffers
    // * Buffer3 gets the result of adding from 1 and 2
    // * Buffer3 gets copied into download buffer
    // * It should have value [8,10,12,14]

    let recorder = instance.create_recorder().unwrap();
    recorder
        .write_buffer::<u32>(&buffer1, 0, &[1, 2, 3, 4])
        .unwrap();
    recorder
        .write_buffer::<u32>(&buffer2, 0, &[5, 6, 7, 8])
        .unwrap();
    recorder
        .write_buffer::<u32>(&buffer3, 0, &[1, 1, 1, 1])
        .unwrap();
    recorder
        .dispatch_kernel(
            &kernel,
            &[
                &BufferSlice::entire_buffer(&buffer1, false).unwrap(),
                &BufferSlice::entire_buffer(&buffer2, false).unwrap(),
                &BufferSlice::entire_buffer(&buffer3, true).unwrap(),
            ],
            [4, 1, 1],
        )
        .unwrap();
    recorder
        .copy_buffer(&buffer3, &download_buffer, 0, 0, 16)
        .unwrap();
    instance.submit_commands(&mut [recorder]).unwrap();

    // Check the result from the download buffer
    assert_eq!(
        download_buffer
            .access(0, 16, false)
            .unwrap()
            .readable::<u32>()
            .unwrap(),
        [6, 8, 10, 12]
    );
    Ok(())
}

dev_utils::all_backend_tests!(add_numbers);
