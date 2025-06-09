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
pub fn add_numbers<Backend: hal::Backend>(hal: Backend::Instance) -> Result<(), ()> {
    // Test specific stuff
    if TypeId::of::<Backend>() == TypeId::of::<hal::Dummy>() {
        // We don't want to "test" on the dummy backend, where nothing happens so the result will be wrong
        return Ok(());
    }
    println!("Hello, world!");
    dev_utils::setup_trace_printer_if_env();

    // Create the instance
    let instance: SupaSimInstance<Backend> = SupaSimInstance::from_hal(hal);

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
    let cache: Option<KernelCache<Backend>> =
        if instance.properties().unwrap().supports_pipeline_cache {
            Some(instance.create_kernel_cache(&[]).unwrap())
        } else {
            None
        };

    // Compile the kernels
    let global_state = kernels::GlobalState::new_from_env().unwrap();
    let mut spirv = Vec::new();
    let mut reflection_info = global_state
        .compile_kernel(kernels::KernelCompileOptions {
            target: types::KernelTarget::Spirv {
                version: kernels::SpirvVersion::V1_2,
            },
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
    // Reflection isn't working yet so this is a temporary workaround
    reflection_info.num_buffers = 3;
    let kernel = instance
        .compile_raw_kernel(&spirv, reflection_info, cache.as_ref())
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
