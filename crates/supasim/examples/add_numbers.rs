use supasim::shaders;
use supasim::{BufferDescriptor, Instance, ShaderReflectionInfo};

pub fn main() {
    println!("Hello, world!");
    let instance: Instance<supasim::hal::Vulkan> =
        Instance::from_hal(supasim::hal::Vulkan::create_instance(true).unwrap());
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
    // If all goes well, these will be cleaned up

    instance.destroy().unwrap();
}
