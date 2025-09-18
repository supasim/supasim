/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use rand::random;
use std::sync::Arc;
use supasim::{Backend, Instance};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::Window,
};

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BuddhabrotInputOptions {
    width: u32,
    height: u32,
    iteration_set_index: u32,
    skip_last_points: u32,
    workgroup_dim: u64,
    random_seed: u64,
}

pub struct AppState<B: hal::Backend> {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,
    instance: Instance<B>,
    supasim_buffer: supasim::Buffer<B>,
    supasim_temp_buffer: supasim::Buffer<B>,
    supasim_width_height_buffer: supasim::Buffer<B>,
    run_kernel: supasim::Kernel<B>,
    finalize_kernel: supasim::Kernel<B>,
    _features: wgpu::Features,
    just_resized: bool,

    width_height_buffer: wgpu::Buffer,
    render_uniform_bind_group: wgpu::BindGroup,
    max_uniform_bind_group: wgpu::BindGroup,
    buffer_bind_group_layout: wgpu::BindGroupLayout,
    wgpu_device_buffer: wgpu::Buffer,

    max_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,

    workgroup_dim: u32,
    iterations: u32,
    skip_last_points: u32,
}
impl<B: hal::Backend> AppState<B> {
    pub async fn new(window: Arc<Window>, hal_desc: hal::InstanceDescriptor<B>) -> Self {
        const DEFAULT_WORKGROUP_DIM: u32 = 4;
        let workgroup_dim = match std::env::var("WORKGROUP_DIM") {
            Ok(w) => match w.parse::<u32>() {
                Ok(v) => v.clamp(1, 16),
                Err(_) => DEFAULT_WORKGROUP_DIM,
            },
            Err(_) => DEFAULT_WORKGROUP_DIM,
        };
        println!(
            "Running with {workgroup_dim}x{workgroup_dim}x{workgroup_dim} workgroup dimensions"
        );
        const DEFAULT_ITERATION_COUNT: u32 = 4;
        // The number of sets of 512 iterations
        let iterations = match std::env::var("ITERATION_SETS") {
            Ok(w) => match w.parse::<u32>() {
                Ok(v) => v.max(1),
                Err(_) => DEFAULT_ITERATION_COUNT,
            },
            Err(_) => DEFAULT_ITERATION_COUNT,
        };
        println!("Rendering with {} iterations", 512 * iterations);
        const DEFAULT_SKIP_LAST_POINTS: u32 = 0;
        let skip_last_points = match std::env::var("SKIP_LAST_POINTS") {
            Ok(w) => match w.parse::<u32>() {
                Ok(v) => v,
                Err(_) => DEFAULT_SKIP_LAST_POINTS,
            },
            Err(_) => DEFAULT_SKIP_LAST_POINTS,
        };
        println!("Skipping last {skip_last_points} points");

        let size = window.inner_size();

        let wgpu_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::PRIMARY,
            flags: wgpu::InstanceFlags::all(),
            ..Default::default()
        });
        let surface = wgpu_instance.create_surface(window.clone()).unwrap();
        let adapter = wgpu_instance
            .request_adapter(&wgpu::RequestAdapterOptions {
                power_preference: wgpu::PowerPreference::HighPerformance,
                force_fallback_adapter: false,
                compatible_surface: Some(&surface),
            })
            .await
            .unwrap();
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features: wgpu::Features::empty(),
                required_limits: wgpu::Limits::defaults(),
                memory_hints: Default::default(),
                trace: wgpu::Trace::Off,
            })
            .await
            .unwrap();
        let surface_caps = surface.get_capabilities(&adapter);
        let surface_format = surface_caps
            .formats
            .iter()
            .copied()
            .find(|f| f.is_srgb())
            .unwrap_or(surface_caps.formats[0]);
        let alpha_mode = surface_caps
            .alpha_modes
            .iter()
            .copied()
            .find(|a| *a == wgpu::CompositeAlphaMode::Opaque)
            .unwrap_or(wgpu::CompositeAlphaMode::Auto);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: wgpu::PresentMode::Fifo,
            alpha_mode,
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let global_state = kernels::GlobalState::new_from_env().unwrap();
        let mut shader_binary = Vec::new();
        let instance = Instance::from_hal(hal_desc);
        let workgroup_size = [16, 16, 1];
        let mut compile_kernel = |entry: &str| {
            shader_binary.clear();
            let reflection_info = global_state
                .compile_kernel(supasim::kernels::KernelCompileOptions {
                    target: instance.properties().unwrap().kernel_lang,
                    source: kernels::KernelSource::Memory(include_bytes!("buddhabrot.slang")),
                    dest: kernels::KernelDest::Memory(&mut shader_binary),
                    entry,
                    include: None,
                    fp_mode: kernels::KernelFpMode::Precise,
                    opt_level: kernels::OptimizationLevel::Standard,
                    stability: kernels::StabilityGuarantee::Stable,
                    minify: true,
                })
                .unwrap();
            assert_eq!(reflection_info.buffers, vec![false, true, true]);
            instance
                .compile_raw_kernel(&shader_binary, reflection_info)
                .unwrap()
        };
        let run_kernel = compile_kernel("Run");
        let finalize_kernel = compile_kernel("Finalize");

        let width_height_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 8,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::COPY_DST,
            mapped_at_creation: false,
        });
        let max_value_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size: 4,
            usage: wgpu::BufferUsages::UNIFORM | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });

        let buffer_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[wgpu::BindGroupLayoutEntry {
                    binding: 0,
                    visibility: wgpu::ShaderStages::all(),
                    ty: wgpu::BindingType::Buffer {
                        ty: wgpu::BufferBindingType::Storage { read_only: true },
                        has_dynamic_offset: false,
                        min_binding_size: None,
                    },
                    count: None,
                }],
            });
        let max_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::COMPUTE,
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Storage { read_only: false },
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let max_pipeline = device.create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        &max_uniform_bind_group_layout,
                        &buffer_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                }),
            ),
            module: &device.create_shader_module(wgpu::ShaderModuleDescriptor {
                label: None,
                source: wgpu::ShaderSource::Wgsl(include_str!("buddhabrot_max.wgsl").into()),
            }),
            entry_point: Some("find_max"),
            cache: None,
            compilation_options: wgpu::PipelineCompilationOptions::default(),
        });

        let render_uniform_bind_group_layout =
            device.create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                label: None,
                entries: &[
                    wgpu::BindGroupLayoutEntry {
                        binding: 0,
                        visibility: wgpu::ShaderStages::all(),
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                    wgpu::BindGroupLayoutEntry {
                        binding: 1,
                        visibility: wgpu::ShaderStages::all(),
                        ty: wgpu::BindingType::Buffer {
                            ty: wgpu::BufferBindingType::Uniform,
                            has_dynamic_offset: false,
                            min_binding_size: None,
                        },
                        count: None,
                    },
                ],
            });
        let render_shader = device.create_shader_module(wgpu::ShaderModuleDescriptor {
            label: None,
            source: wgpu::ShaderSource::Wgsl(include_str!("buddhabrot_render.wgsl").into()),
        });
        let render_pipeline = device.create_render_pipeline(&wgpu::RenderPipelineDescriptor {
            label: None,
            layout: Some(
                &device.create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                    label: None,
                    bind_group_layouts: &[
                        &render_uniform_bind_group_layout,
                        &buffer_bind_group_layout,
                    ],
                    push_constant_ranges: &[],
                }),
            ),
            vertex: wgpu::VertexState {
                module: &render_shader,
                entry_point: Some("render_vs"),
                compilation_options: Default::default(),
                buffers: &[],
            },
            primitive: Default::default(),
            depth_stencil: None,
            multisample: Default::default(),
            fragment: Some(wgpu::FragmentState {
                module: &render_shader,
                entry_point: Some("render_fs"),
                compilation_options: Default::default(),
                targets: &[Some(wgpu::ColorTargetState {
                    format: surface_format,

                    blend: None,
                    write_mask: wgpu::ColorWrites::ALL,
                })],
            }),
            multiview: None,
            cache: None,
        });
        let max_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &max_uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &width_height_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &max_value_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });
        let render_uniform_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &render_uniform_bind_group_layout,
            entries: &[
                wgpu::BindGroupEntry {
                    binding: 0,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &width_height_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
                wgpu::BindGroupEntry {
                    binding: 1,
                    resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                        buffer: &max_value_buffer,
                        offset: 0,
                        size: None,
                    }),
                },
            ],
        });

        let supasim_temp_buffer = instance
            .create_buffer(&supasim::BufferDescriptor {
                size: 24
                    * workgroup_size[0] as u64
                    * workgroup_size[1] as u64
                    * workgroup_size[2] as u64
                    * workgroup_dim as u64
                    * workgroup_dim as u64
                    * workgroup_dim as u64,
                buffer_type: supasim::BufferType::Gpu,
                contents_align: 8,
                priority: 1.0,
                can_export: false,
            })
            .unwrap();

        let (supasim_buffer, wgpu_device_buffer) =
            Self::create_buffer(config.width, config.height, &instance, &device);

        let supasim_width_height_buffer = instance
            .create_buffer(&supasim::BufferDescriptor {
                size: size_of::<BuddhabrotInputOptions>() as u64,
                buffer_type: supasim::BufferType::Gpu,
                contents_align: 4,
                priority: 1.0,
                can_export: false,
            })
            .unwrap();

        // Layouts:
        // Buffer bind group - just one buffer, the main imported buffer
        // Max uniform bind group - width/height, max value
        // Render uniform bind group - width/height, max value

        Self {
            surface,
            device,
            queue,
            config,
            is_surface_configured: false,
            window,
            instance,
            supasim_buffer,
            _features: wgpu::Features::empty(),
            just_resized: true,
            run_kernel,
            finalize_kernel,
            supasim_width_height_buffer,
            max_pipeline,
            render_pipeline,

            width_height_buffer,
            render_uniform_bind_group,
            max_uniform_bind_group,
            buffer_bind_group_layout,
            wgpu_device_buffer,

            workgroup_dim,
            supasim_temp_buffer,
            iterations,
            skip_last_points,
        }
    }
    pub fn create_buffer(
        width: u32,
        height: u32,
        instance: &Instance<B>,
        device: &wgpu::Device,
    ) -> (supasim::Buffer<B>, wgpu::Buffer) {
        let size = width as u64 * height as u64 * 4;
        let supasim_buffer = instance
            .create_buffer(&supasim::BufferDescriptor {
                size,
                buffer_type: supasim::BufferType::Gpu,
                contents_align: 4,
                priority: 1.0,
                can_export: true,
            })
            .unwrap();
        let device_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            label: None,
            size,
            usage: wgpu::BufferUsages::COPY_DST | wgpu::BufferUsages::STORAGE,
            mapped_at_creation: false,
        });
        (supasim_buffer, device_buffer)
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            let (supasim_buffer, wgpu_device_buffer) =
                Self::create_buffer(width, height, &self.instance, &self.device);
            self.supasim_buffer = supasim_buffer;
            self.wgpu_device_buffer = wgpu_device_buffer;
            self.just_resized = true;
            self.queue.write_buffer(
                &self.width_height_buffer,
                0,
                bytemuck::cast_slice(&[self.config.width, self.config.height]),
            );
        }
    }

    fn update(&mut self) {
        let recorder = self.instance.create_recorder().unwrap();
        if self.just_resized {
            recorder
                .zero_memory(
                    &self.supasim_buffer,
                    0,
                    self.config.width as u64 * self.config.height as u64 * 4,
                )
                .unwrap();
            self.just_resized = false;
        }

        let workgroup_dims = [self.workgroup_dim, self.workgroup_dim, self.workgroup_dim];
        let buffers = [
            &self.supasim_width_height_buffer.slice(.., false),
            &self.supasim_buffer.slice(.., true),
            &self.supasim_temp_buffer.slice(.., true),
        ];
        let random_seed = random::<u64>();
        for i in 0..self.iterations {
            // Update set iteration index
            recorder
                .write_buffer(
                    &self.supasim_width_height_buffer,
                    0,
                    &[BuddhabrotInputOptions {
                        width: self.config.width,
                        height: self.config.height,
                        random_seed,
                        iteration_set_index: i,
                        skip_last_points: self.skip_last_points,
                        workgroup_dim: self.workgroup_dim as u64,
                    }],
                )
                .unwrap();
            recorder
                .dispatch_kernel(&self.run_kernel, &buffers, workgroup_dims)
                .unwrap();
        }
        for i in 0..self.iterations {
            // Update set iteration index
            recorder
                .write_buffer(
                    &self.supasim_width_height_buffer,
                    0,
                    &[BuddhabrotInputOptions {
                        width: self.config.width,
                        height: self.config.height,
                        random_seed,
                        iteration_set_index: i,
                        skip_last_points: self.skip_last_points,
                        workgroup_dim: self.workgroup_dim as u64,
                    }],
                )
                .unwrap();
            recorder
                .dispatch_kernel(&self.finalize_kernel, &buffers, workgroup_dims)
                .unwrap();
        }
        let download_buffer = self
            .instance
            .create_buffer(&supasim::BufferDescriptor {
                size: self.config.width as u64 * self.config.height as u64 * 4,
                buffer_type: supasim::BufferType::Download,
                contents_align: 4,
                priority: 1.0,
                can_export: false,
            })
            .unwrap();
        recorder
            .copy_buffer(
                &self.supasim_buffer,
                &download_buffer,
                0,
                0,
                self.config.width as u64 * self.config.height as u64 * 4,
            )
            .unwrap();
        self.instance.submit_commands(&mut [recorder]).unwrap();
        {
            let access = download_buffer
                .access(
                    0,
                    self.config.width as u64 * self.config.height as u64 * 4,
                    false,
                )
                .unwrap();
            let readable = access.readable::<u8>().unwrap();
            self.queue
                .write_buffer(&self.wgpu_device_buffer, 0, readable);
        }
    }

    pub fn render(&mut self) -> bool {
        self.window.request_redraw();

        if !self.is_surface_configured {
            return false;
        }

        self.update();

        let output = self.surface.get_current_texture().unwrap();
        let output_view = output
            .texture
            .create_view(&wgpu::TextureViewDescriptor::default());
        let buffer_bind_group = self.device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: &self.buffer_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &self.wgpu_device_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });

        let mut encoder = self
            .device
            .create_command_encoder(&wgpu::CommandEncoderDescriptor::default());
        {
            const MAX_PIPELINE_WORKGROUP_SIZE: u32 = 256;
            let mut max_pass = encoder.begin_compute_pass(&wgpu::ComputePassDescriptor {
                label: None,
                timestamp_writes: None,
            });
            max_pass.set_pipeline(&self.max_pipeline);
            max_pass.set_bind_group(0, Some(&self.max_uniform_bind_group), &[]);
            max_pass.set_bind_group(1, Some(&buffer_bind_group), &[]);
            max_pass.dispatch_workgroups(
                (self.config.width * self.config.height).div_ceil(MAX_PIPELINE_WORKGROUP_SIZE),
                1,
                1,
            );
        }
        {
            let mut render_pass = encoder.begin_render_pass(&wgpu::RenderPassDescriptor {
                label: None,
                color_attachments: &[Some(wgpu::RenderPassColorAttachment {
                    view: &output_view,
                    resolve_target: None,
                    ops: wgpu::Operations {
                        load: wgpu::LoadOp::Clear(wgpu::Color {
                            r: 0.1,
                            g: 0.2,
                            b: 0.3,
                            a: 1.0,
                        }),
                        store: wgpu::StoreOp::Store,
                    },
                    depth_slice: None,
                })],
                depth_stencil_attachment: None,
                occlusion_query_set: None,
                timestamp_writes: None,
            });
            render_pass.set_pipeline(&self.render_pipeline);
            render_pass.set_bind_group(0, Some(&self.render_uniform_bind_group), &[]);
            render_pass.set_bind_group(1, Some(&buffer_bind_group), &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        output.present();
        true
    }
}

pub struct App<B: hal::Backend> {
    state: Option<AppState<B>>,
    desc: Option<hal::InstanceDescriptor<B>>,
}
impl<B: hal::Backend> ApplicationHandler<AppState<B>> for App<B> {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = Window::default_attributes();

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());
        window.set_title("SupaSim Buddhabrot Demo");
        let i = std::mem::take(&mut self.desc).unwrap();
        self.state = Some(pollster::block_on(AppState::new(window, i)));
    }
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppState<B>) {
        self.state = Some(event);
    }
    fn window_event(
        &mut self,
        event_loop: &winit::event_loop::ActiveEventLoop,
        window_id: winit::window::WindowId,
        event: winit::event::WindowEvent,
    ) {
        assert!(window_id == self.state.as_ref().unwrap().window.id());
        let state = match &mut self.state {
            Some(canvas) => canvas,
            None => return,
        };

        match event {
            WindowEvent::CloseRequested => event_loop.exit(),
            WindowEvent::Resized(size) => state.resize(size.width, size.height),
            WindowEvent::RedrawRequested => {
                if !state.render() {
                    let size = state.window.inner_size();
                    state.resize(size.width, size.height);
                }
            }
            _ => (),
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
            let event_loop = EventLoop::with_user_event().build().unwrap();
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            let mut app = App::<hal::Wgpu> {
                state: None,
                desc: Some(
                    hal::Wgpu::create_instance(true, wgpu::Backends::PRIMARY, None).unwrap(),
                ),
            };
            event_loop.run_app(&mut app).unwrap();
        }
        Backend::Vulkan => {
            println!("Selected vulkan backend");
            let event_loop = EventLoop::with_user_event().build().unwrap();
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            let mut app = App::<hal::Vulkan> {
                state: None,
                desc: Some(hal::Vulkan::create_instance(true).unwrap()),
            };
            event_loop.run_app(&mut app).unwrap();
        }
        #[cfg(target_vendor = "apple")]
        Backend::Metal => {
            use hal::Backend;

            println!("Selected metal backend");
            let event_loop = EventLoop::with_user_event().build().unwrap();
            event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
            let mut app = App::<hal::Metal> {
                state: None,
                desc: Some(hal::Metal::setup_default_descriptor().unwrap()),
            };
            event_loop.run_app(&mut app).unwrap();
        }
        #[cfg(not(target_vendor = "apple"))]
        Backend::Metal => unreachable!(),
    }
}
