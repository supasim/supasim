use supasim::Backend;
use supasim::Instance;

pub struct AppState<B: hal::Backend> {
    instance: Instance<B>,
}
impl<B: hal::Backend> AppState<B> {
    pub fn new(instance: Instance<B>) -> Self {
        Self { instance }
    }
    pub fn run(&mut self) {}
}

pub fn main() {
    dev_utils::setup_trace_printer_if_env();
    let backend = match std::env::var("BACKEND") {
        Ok(b) => match b.as_str() {
            "vulkan" => Backend::Vulkan,
            #[cfg(target_vendor = "apple")]
            "metal" => Backend::Metal,
            _ => Backend::Wgpu,
        },
        Err(_) => Backend::Wgpu,
    };
    match backend {
        Backend::Wgpu => {
            println!("Selected wgpu backend");
            let mut state = AppState::new(Instance::from_hal(
                hal::Wgpu::create_instance(true, wgpu::Backends::PRIMARY, None).unwrap(),
            ));
            state.run();
        }
        Backend::Vulkan => {
            println!("Selected vulkan backend");
            let mut state = AppState::new(Instance::from_hal(
                hal::Vulkan::create_instance(true).unwrap(),
            ));
            state.run();
        }
        #[cfg(target_vendor = "apple")]
        Backend::Metal => {
            use hal::Backend;

            println!("Selected metal backend");
            let mut state = AppState::new(Instance::from_hal(
                hal::Metal::setup_default_descriptor().unwrap(),
            ));
            state.run();
        }
        #[cfg(not(target_vendor = "apple"))]
        Backend::Metal => unreachable!(),
    }
}
