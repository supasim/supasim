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
use std::time::Duration;
use supasim::{BufferDescriptor, BufferSlice, Instance, shaders};
use tracing_subscriber::filter::LevelFilter;

pub fn main_test<Backend: supasim::hal::Backend>(hal: Backend::Instance) {
    println!("Hello, world!");
    tracing_subscriber::fmt()
        .with_max_level(LevelFilter::TRACE)
        .json()
        .flatten_event(true)
        .init();
    let instance: Instance<Backend> = Instance::from_hal(hal);
    let upload_buffer = instance
        .create_buffer(&BufferDescriptor {
            size: 64,
            buffer_type: supasim::BufferType::Upload,
            contents_align: 4,
            ..Default::default()
        })
        .unwrap();
    let download_buffer = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: supasim::BufferType::Download,
            contents_align: 4,
            ..Default::default()
        })
        .unwrap();
    let buffer1 = instance
        .create_buffer(&BufferDescriptor {
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
    let cache = instance.create_kernel_cache(&[]).unwrap();
    let global_state = shaders::GlobalState::new_from_env().unwrap();
    let mut spirv = Vec::new();
    let reflection_info = global_state
        .compile_shader(shaders::ShaderCompileOptions {
            target: types::ShaderTarget::Spirv {
                version: shaders::SpirvVersion::V1_2,
            },
            source: shaders::ShaderSource::Memory(include_bytes!("add_numbers.slang")),
            dest: shaders::ShaderDest::Memory(&mut spirv),
            entry: "main",
            include: None,
            fp_mode: shaders::ShaderFpMode::Precise,
            opt_level: shaders::OptimizationLevel::Maximal,
            stability: shaders::StabilityGuarantee::ExtraValidation,
            minify: false,
        })
        .unwrap();
    let kernel = instance
        .compile_kernel(&spirv, reflection_info, Some(&cache))
        .unwrap();
    let recorder = instance.create_recorder().unwrap();
    let buffers = [&BufferSlice {
        buffer: upload_buffer.clone(),
        start: 0,
        len: 64,
        needs_mut: true,
    }];
    instance
        .access_buffers(
            Box::new(|buffers| {
                buffers[0]
                    .writeable::<u32>()
                    .unwrap()
                    .clone_from_slice(&[1, 2, 3, 4, 5, 6, 7, 8, 1, 1, 1, 1, 22, 22, 22, 22]);
                Ok(())
            }),
            &buffers,
        )
        .unwrap();
    instance
        .access_buffers(
            Box::new(|buffers| {
                println!("Buffer 0: {:?}", buffers[0].readable::<u32>());
                Ok(())
            }),
            &buffers,
        )
        .unwrap();
    /*instance
    .access_buffers(
        Box::new(|buffers| {
            buffers[0]
                .writeable::<u32>()?
                .copy_from_slice(&[1, 2, 3, 4]);
            buffers[1]
                .writeable::<u32>()?
                .copy_from_slice(&[5, 6, 7, 8]);
            buffers[2]
                .writeable::<u32>()?
                .copy_from_slice(&[1, 1, 1, 1]);
            Ok(())
        }),
        &buffers,
    )
    .unwrap();*/
    recorder
        .copy_buffer(upload_buffer.clone(), buffer1.clone(), 0, 0, 16)
        .unwrap();
    recorder
        .copy_buffer(upload_buffer.clone(), buffer2.clone(), 16, 0, 16)
        .unwrap();
    recorder
        .copy_buffer(upload_buffer.clone(), buffer3.clone(), 32, 0, 16)
        .unwrap();
    recorder
        .copy_buffer(upload_buffer.clone(), download_buffer.clone(), 48, 0, 16)
        .unwrap();
    let buffers = [
        &BufferSlice::entire_buffer(&buffer1, false).unwrap(),
        &BufferSlice::entire_buffer(&buffer2, false).unwrap(),
        &BufferSlice::entire_buffer(&buffer3, true).unwrap(),
    ];
    recorder
        .dispatch_kernel(kernel.clone(), &buffers, [4, 1, 1])
        .unwrap();
    recorder
        .copy_buffer(buffer3.clone(), download_buffer.clone(), 0, 0, 16)
        .unwrap();
    instance.submit_commands(&mut [recorder]).unwrap();
    let buffers = [
        &BufferSlice::entire_buffer(&download_buffer, false).unwrap(),
        &BufferSlice::entire_buffer(&upload_buffer, false).unwrap(),
    ];
    std::thread::sleep(Duration::from_secs(2));
    instance
        .access_buffers(
            Box::new(|buffers| {
                for buffer in buffers {
                    println!("{:?}", buffer.readable::<u32>()?);
                }
                Ok(())
            }),
            &buffers,
        )
        .unwrap();
    // If all goes well, these will be cleaned up

    instance.destroy().unwrap();
}
pub fn main() {
    if true {
        let instance =
            hal::Wgpu::create_instance(true, hal::wgpu::wgpu::Backends::PRIMARY, None).unwrap();
        main_test::<hal::Wgpu>(instance);
    } else {
        let instance = hal::Vulkan::create_instance(true).unwrap();
        main_test::<hal::Vulkan>(instance);
    }
}
