pub fn main() {
    println!("Hello, world!");
    let instance = supasim::hal::Vulkan::create_instance(true).unwrap();
    instance.destroy();
}
