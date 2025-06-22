//! Lots of this code is just stolen from https://sotrh.github.io/learn-wgpu/ lol

use rand::random;
use std::sync::Arc;
use supasim::{SupaSimInstance, wgpu};

use winit::{
    application::ApplicationHandler,
    event::WindowEvent,
    event_loop::{ActiveEventLoop, EventLoop},
    window::Window,
};

pub type Backend = supasim::hal::Wgpu;

#[repr(C)]
#[derive(Clone, Copy, Debug, bytemuck::Pod, bytemuck::Zeroable)]
struct BuddhabrotInputOptions {
    width: u32,
    height: u32,
    random_seed: u64,
}

pub struct AppState {
    surface: wgpu::Surface<'static>,
    device: wgpu::Device,
    queue: wgpu::Queue,
    config: wgpu::SurfaceConfiguration,
    is_surface_configured: bool,
    window: Arc<Window>,
    instance: SupaSimInstance<Backend>,
    supasim_buffer: supasim::Buffer<Backend>,
    supasim_width_height_buffer: supasim::Buffer<Backend>,
    kernel: supasim::Kernel<Backend>,
    shared_buffer: wgpu::Buffer,
    features: wgpu::Features,
    just_resized: bool,

    width_height_buffer: wgpu::Buffer,
    render_uniform_bind_group: wgpu::BindGroup,
    max_uniform_bind_group: wgpu::BindGroup,
    buffer_bind_group: wgpu::BindGroup,
    buffer_bind_group_layout: wgpu::BindGroupLayout,

    max_pipeline: wgpu::ComputePipeline,
    render_pipeline: wgpu::RenderPipeline,
}
impl AppState {
    pub async fn new(window: Arc<Window>) -> Self {
        let size = window.inner_size();

        let wgpu_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
            backends: wgpu::Backends::VULKAN,
            flags: wgpu::InstanceFlags::all(),
            ..Default::default()
        });
        let surface = wgpu_instance.create_surface(window.clone()).unwrap();
        let required_features = if cfg!(windows) {
            wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32
        } else {
            wgpu::Features::VULKAN_EXTERNAL_MEMORY_FD
        };
        let adapter = match wgpu_instance
            .enumerate_adapters(wgpu::Backends::VULKAN)
            .into_iter()
            .find(|a| a.features().contains(required_features))
        {
            Some(adapter) => adapter,
            None => {
                panic!("No wpgu adpters have export memory property");
            }
        };
        let (device, queue) = adapter
            .request_device(&wgpu::DeviceDescriptor {
                label: None,
                required_features,
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
            .find(|f| f.is_srgb())
            .copied()
            .unwrap_or(surface_caps.formats[0]);
        let config = wgpu::SurfaceConfiguration {
            usage: wgpu::TextureUsages::RENDER_ATTACHMENT,
            format: surface_format,
            width: size.width,
            height: size.height,
            present_mode: surface_caps.present_modes[0],
            alpha_mode: surface_caps.alpha_modes[0],
            view_formats: vec![],
            desired_maximum_frame_latency: 2,
        };

        let instance = SupaSimInstance::from_hal(
            Backend::create_instance(true, wgpu::Backends::VULKAN, None).unwrap(),
        );

        let global_state = kernels::GlobalState::new_from_env().unwrap();
        let mut spirv = Vec::new();
        global_state
            .compile_kernel(supasim::kernels::KernelCompileOptions {
                target: instance.properties().unwrap().kernel_lang,
                source: kernels::KernelSource::Memory(include_bytes!("buddhabrot.slang")),
                dest: kernels::KernelDest::Memory(&mut spirv),
                entry: "Main",
                include: None,
                fp_mode: kernels::KernelFpMode::Precise,
                opt_level: kernels::OptimizationLevel::Standard,
                stability: kernels::StabilityGuarantee::Stable,
                minify: true,
            })
            .unwrap();

        let kernel = instance
            .compile_raw_kernel(
                &spirv,
                supasim::KernelReflectionInfo {
                    workgroup_size: [32, 32, 1],
                    num_buffers: 2,
                },
                None,
            )
            .unwrap();

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

        let (supasim_buffer, shared_buffer, buffer_bind_group) = Self::create_buffer(
            &device,
            required_features,
            config.width,
            config.height,
            &instance,
            &buffer_bind_group_layout,
        );

        let supasim_width_height_buffer = instance
            .create_buffer(&supasim::BufferDescriptor {
                size: 16,
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
            shared_buffer,
            features: required_features,
            just_resized: true,
            kernel,
            supasim_width_height_buffer,
            max_pipeline,
            render_pipeline,

            width_height_buffer,
            buffer_bind_group,
            render_uniform_bind_group,
            max_uniform_bind_group,
            buffer_bind_group_layout,
        }
    }
    pub fn create_buffer(
        device: &wgpu::Device,
        features: wgpu::Features,
        width: u32,
        height: u32,
        instance: &SupaSimInstance<Backend>,
        buffer_bind_group_layout: &wgpu::BindGroupLayout,
    ) -> (supasim::Buffer<Backend>, wgpu::Buffer, wgpu::BindGroup) {
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
        /*let shared_buffer = unsafe {
            supasim_buffer
                .export_to_wgpu(WgpuDeviceExportInfo {
                    device: device.clone(),
                    features,
                    backend: wgpu::Backend::Vulkan,
                    usages: wgpu::BufferUsages::STORAGE
                        | wgpu::BufferUsages::COPY_DST
                        | wgpu::BufferUsages::COPY_SRC,
                })
                .unwrap()
        };*/
        let shared_buffer = device.create_buffer(&wgpu::BufferDescriptor {
            usage: wgpu::BufferUsages::STORAGE
                | wgpu::BufferUsages::COPY_DST
                | wgpu::BufferUsages::COPY_SRC,
            label: None,
            size,
            mapped_at_creation: false,
        });
        let buffer_bind_group = device.create_bind_group(&wgpu::BindGroupDescriptor {
            label: None,
            layout: buffer_bind_group_layout,
            entries: &[wgpu::BindGroupEntry {
                binding: 0,
                resource: wgpu::BindingResource::Buffer(wgpu::BufferBinding {
                    buffer: &shared_buffer,
                    offset: 0,
                    size: None,
                }),
            }],
        });
        (supasim_buffer, shared_buffer, buffer_bind_group)
    }
    pub fn resize(&mut self, width: u32, height: u32) {
        if width > 0 && height > 0 {
            self.config.width = width;
            self.config.height = height;
            self.surface.configure(&self.device, &self.config);
            self.is_surface_configured = true;

            let (supasim_buffer, shared_buffer, buffer_bind_group) = Self::create_buffer(
                &self.device,
                self.features,
                width,
                height,
                &self.instance,
                &self.buffer_bind_group_layout,
            );
            self.supasim_buffer = supasim_buffer;
            self.shared_buffer = shared_buffer;
            self.buffer_bind_group = buffer_bind_group;
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
        let random_seed = random::<u64>();
        recorder
            .write_buffer(
                &self.supasim_width_height_buffer,
                0,
                &[BuddhabrotInputOptions {
                    width: self.config.width,
                    height: self.config.height,
                    random_seed,
                }],
            )
            .unwrap();
        recorder
            .dispatch_kernel(
                &self.kernel,
                &[
                    &self.supasim_width_height_buffer.slice(.., false),
                    &self.supasim_buffer.slice(.., true),
                ],
                [16, 16, 16],
            )
            .unwrap();
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
        self.instance
            .submit_commands(&mut [recorder])
            .unwrap()
            .wait()
            .unwrap();
        let access = download_buffer
            .access(
                0,
                self.config.width as u64 * self.config.height as u64 * 4,
                false,
            )
            .unwrap();
        let readable = access.readable::<u32>().unwrap();
        // Count is always zero :(
        let mut count = 0;
        for &val in readable {
            if val != 0 {
                count += 1;
            }
        }
        println!("Count: {count}");
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
            max_pass.set_bind_group(1, Some(&self.buffer_bind_group), &[]);
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
            render_pass.set_bind_group(1, Some(&self.buffer_bind_group), &[]);
            render_pass.draw(0..6, 0..1);
        }

        self.queue.submit([encoder.finish()]);
        output.present();
        true
    }
}

#[derive(Default)]
pub struct App {
    state: Option<AppState>,
}
impl ApplicationHandler<AppState> for App {
    fn resumed(&mut self, event_loop: &winit::event_loop::ActiveEventLoop) {
        let window_attributes = Window::default_attributes();

        let window = Arc::new(event_loop.create_window(window_attributes).unwrap());

        self.state = Some(pollster::block_on(AppState::new(window)));
    }
    fn user_event(&mut self, _event_loop: &ActiveEventLoop, event: AppState) {
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
    env_logger::init();
    let event_loop = EventLoop::with_user_event().build().unwrap();
    event_loop.set_control_flow(winit::event_loop::ControlFlow::Poll);
    let mut app = App::default();
    event_loop.run_app(&mut app).unwrap();
}
