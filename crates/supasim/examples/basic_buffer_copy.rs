use std::fmt::Write;
use supasim::{BufferDescriptor, BufferSlice, SupaSimInstance};
use tracing_subscriber::{
    layer::{Context, SubscriberExt},
    util::SubscriberInitExt,
};

struct EnterSpanPrinter;

impl<S> tracing_subscriber::Layer<S> for EnterSpanPrinter
where
    S: tracing::Subscriber,
    S: for<'a> tracing_subscriber::registry::LookupSpan<'a>,
{
    fn on_enter(&self, id: &tracing::Id, ctx: Context<'_, S>) {
        if let Some(span_ref) = ctx.span(id) {
            let name = span_ref.name();

            // Get all field values as a string
            let mut fields = String::new();
            if let Some(ext) = span_ref
                .extensions()
                .get::<tracing_subscriber::fmt::FormattedFields<
                    tracing_subscriber::fmt::format::DefaultFields,
                >>()
            {
                write!(fields, "{}", ext).ok();
            }

            println!("\t{} [{}]", name, fields);
        }
    }
}

pub fn main_test<Backend: supasim::hal::Backend>(hal: Backend::Instance) {
    println!("Hello, world!");
    if false {
        tracing_subscriber::registry()
            .with(EnterSpanPrinter)
            .with(tracing_subscriber::fmt::layer())
            .init();
    }
    let instance = SupaSimInstance::<Backend>::from_hal(hal);
    let buffer1 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: supasim::BufferType::Upload,
            contents_align: 4,
            priority: 0.0,
        })
        .unwrap();
    let buffer2 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: supasim::BufferType::Storage,
            contents_align: 4,
            priority: 0.0,
        })
        .unwrap();
    let buffer3 = instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            buffer_type: supasim::BufferType::Download,
            contents_align: 4,
            priority: 0.0,
        })
        .unwrap();
    let slices = [
        &BufferSlice::entire_buffer(&buffer1, true).unwrap(),
        &BufferSlice::entire_buffer(&buffer3, false).unwrap(),
    ];
    instance
        .access_buffers(
            Box::new(|buffers| {
                println!("{:?}", buffers[0].readable::<u32>().unwrap());
                println!("{:?}", buffers[1].readable::<u32>().unwrap());
                buffers[0]
                    .writeable::<u32>()
                    .unwrap()
                    .clone_from_slice(&[1, 2, 3, 4]);
                Ok(())
            }),
            &slices[..],
        )
        .unwrap();
    let slices = [
        &BufferSlice::entire_buffer(&buffer1, false).unwrap(),
        &BufferSlice::entire_buffer(&buffer3, false).unwrap(),
    ];
    instance
        .access_buffers(
            Box::new(|buffers| {
                println!("{:?}", buffers[0].readable::<u32>().unwrap());
                println!("{:?}", buffers[1].readable::<u32>().unwrap());
                Ok(())
            }),
            &slices[..],
        )
        .unwrap();
    let recorder = instance.create_recorder().unwrap();
    recorder
        .copy_buffer(buffer1.clone(), buffer2.clone(), 0, 0, 16)
        .unwrap();
    recorder
        .copy_buffer(buffer2.clone(), buffer3.clone(), 0, 0, 16)
        .unwrap();
    instance.submit_commands(&mut [recorder]).unwrap();
    instance.wait_for_idle(1.0).unwrap();
    instance
        .access_buffers(
            Box::new(|buffers| {
                println!("{:?}", buffers[0].readable::<u32>().unwrap());
                println!("{:?}", buffers[1].readable::<u32>().unwrap());
                Ok(())
            }),
            &slices[..],
        )
        .unwrap();
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
