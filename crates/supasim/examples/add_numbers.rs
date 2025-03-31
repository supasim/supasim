use supasim::{BufferDescriptor, Instance};
use supasim::{BufferSlice, shaders};

pub fn main() {
    println!("Hello, world!");
    let instance: Instance<supasim::hal::vulkan::Vulkan> =
        Instance::from_hal(supasim::hal::vulkan::Vulkan::create_instance(true).unwrap());
    let buffer1 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            memory_type: types::MemoryType::Upload,
            transfer_src: false,
            transfer_dst: true,
            contents_align: 4,
            ..Default::default()
        })
        .unwrap();
    let buffer2 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            memory_type: types::MemoryType::Upload,
            transfer_src: false,
            transfer_dst: true,
            contents_align: 4,
            ..Default::default()
        })
        .unwrap();
    let buffer3 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            memory_type: types::MemoryType::Download,
            transfer_src: true,
            transfer_dst: false,
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
    let recorder = instance.create_recorder(false).unwrap();
    let buffers = [
        &BufferSlice::entire_buffer(&buffer1, false).unwrap(),
        &BufferSlice::entire_buffer(&buffer2, false).unwrap(),
        &BufferSlice::entire_buffer(&buffer3, true).unwrap(),
    ];
    instance
        .access_buffers(
            Box::new(|buffers| {
                buffers[0].write::<u32>()?.copy_from_slice(&[1, 2, 3, 4]);
                buffers[1].write::<u32>()?.copy_from_slice(&[5, 6, 7, 8]);
                buffers[2].write::<u32>()?.copy_from_slice(&[1, 1, 1, 1]);
                Ok(())
            }),
            &buffers,
        )
        .unwrap();
    recorder
        .dispatch_kernel(kernel.clone(), &buffers, [1, 1, 1], false)
        .unwrap();
    instance.submit_commands(&[recorder]).unwrap();
    instance
        .access_buffers(
            Box::new(|buffers| {
                println!("{:?}", buffers[2].read::<u32>()?);
                Ok(())
            }),
            &buffers,
        )
        .unwrap();
    // If all goes well, these will be cleaned up

    instance.destroy().unwrap();
}
