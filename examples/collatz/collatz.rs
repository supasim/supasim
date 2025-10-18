use bytemuck::Pod;
use bytemuck::Zeroable;
use supasim::Backend;
use supasim::BufferDescriptor;
use supasim::Instance;

#[repr(C)]
#[derive(Pod, Debug, Clone, Copy, Zeroable)]
struct GlobalData {
    min_limit: u64,
    max_limit: u64,
    num_unsolved: u64,
    new_num_unsolved: u64,
    smallest_unsolved: u64,

    dispatch_size: [u32; 3],
    unsolved_buffer_size: u32,
}

#[repr(C)]
#[derive(Pod, Debug, Clone, Copy, Zeroable)]
struct UnsolvedElement {
    number: u64,
    current_value: u64,
    attempts_so_far: u64,
}

pub struct AppState<B: hal::Backend> {
    instance: Instance<B>,
}
impl<B: hal::Backend> AppState<B> {
    pub fn new(instance: Instance<B>) -> Self {
        Self { instance }
    }
    pub fn run(&mut self) {
        const WORKGROUP_DIM: u32 = 4;
        const UNSOLVED_BUFFER_SIZE: u32 = WORKGROUP_DIM * WORKGROUP_DIM * WORKGROUP_DIM * 4;
        let kernel1 = self
            .instance
            .compile_slang_kernel(include_str!("collatz.slang"), "collatzMain")
            .unwrap();
        let kernel2 = self
            .instance
            .compile_slang_kernel(include_str!("collatz.slang"), "sortUnsolved1")
            .unwrap();
        let kernel3 = self
            .instance
            .compile_slang_kernel(include_str!("collatz.slang"), "sortUnsolved2")
            .unwrap();
        let global_state_size = size_of::<GlobalData>().next_multiple_of(64) as u64;
        let global_state = self
            .instance
            .create_buffer(&BufferDescriptor {
                size: global_state_size,
                buffer_type: supasim::BufferType::Gpu,
                contents_align: 8,
                priority: 0.0,
                can_export: false,
            })
            .unwrap();
        let unsolved_buffer = self
            .instance
            .create_buffer(&BufferDescriptor {
                size: UNSOLVED_BUFFER_SIZE as u64 * size_of::<UnsolvedElement>() as u64,
                buffer_type: supasim::BufferType::Gpu,
                contents_align: 8,
                priority: 0.0,
                can_export: false,
            })
            .unwrap();
        let download_buffers: Vec<_> = std::iter::repeat_with(|| {
            self.instance
                .create_buffer(&BufferDescriptor {
                    size: global_state_size,
                    buffer_type: supasim::BufferType::Gpu,
                    contents_align: 8,
                    priority: 0.0,
                    can_export: false,
                })
                .unwrap()
        })
        .take(2)
        .collect();

        let setup_recorder = self.instance.create_recorder().unwrap();
        let initial_global_data = GlobalData {
            min_limit: 2,
            max_limit: 2,
            num_unsolved: 0,
            new_num_unsolved: 0,
            smallest_unsolved: u64::MAX,
            dispatch_size: [WORKGROUP_DIM, WORKGROUP_DIM, WORKGROUP_DIM],
            unsolved_buffer_size: UNSOLVED_BUFFER_SIZE,
        };
        setup_recorder
            .write_buffer(&global_state, 0, bytemuck::bytes_of(&initial_global_data))
            .unwrap();
        setup_recorder
            .write_buffer(
                &download_buffers[0],
                0,
                bytemuck::bytes_of(&initial_global_data),
            )
            .unwrap();
        setup_recorder
            .write_buffer(
                &download_buffers[1],
                0,
                bytemuck::bytes_of(&initial_global_data),
            )
            .unwrap();
        self.instance
            .submit_commands(&mut [setup_recorder])
            .unwrap();

        let mut current_iteration = 0;

        loop {
            let recorder = self.instance.create_recorder().unwrap();
            let buffers_to_use = [
                &global_state.slice(.., true),
                &unsolved_buffer.slice(.., true),
            ];
            recorder
                .dispatch_kernel(
                    &kernel1,
                    &buffers_to_use,
                    [WORKGROUP_DIM, WORKGROUP_DIM, WORKGROUP_DIM],
                )
                .unwrap();
            recorder
                .dispatch_kernel(
                    &kernel2,
                    &buffers_to_use,
                    [WORKGROUP_DIM, WORKGROUP_DIM, WORKGROUP_DIM],
                )
                .unwrap();
            recorder
                .dispatch_kernel(
                    &kernel3,
                    &buffers_to_use,
                    [WORKGROUP_DIM, WORKGROUP_DIM, WORKGROUP_DIM],
                )
                .unwrap();
            recorder
                .copy_buffer(
                    &global_state,
                    &download_buffers[current_iteration % 2],
                    0,
                    0,
                    global_state_size,
                )
                .unwrap();
            self.instance.submit_commands(&mut [recorder]).unwrap();

            current_iteration += 1;

            let mut state = GlobalData::zeroed();
            download_buffers[current_iteration % 2]
                .read(0, std::slice::from_mut(&mut state))
                .unwrap();
            println!(
                "Iteration {current_iteration} raised lowest integer to {}",
                state.min_limit
            );
        }
    }
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
