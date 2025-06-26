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
pub use paste;
use std::fmt::Write;
use tracing_subscriber::{
    layer::{Context, SubscriberExt},
    util::SubscriberInitExt,
};

pub struct EnterSpanPrinter;

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
                write!(fields, "{ext}").ok();
            }

            println!("\t{name} [{fields}]");
        }
    }
}
pub fn setup_trace_printer() {
    let _ = tracing_subscriber::registry()
        .with(EnterSpanPrinter)
        .with(tracing_subscriber::fmt::layer())
        .try_init();
}
pub fn setup_trace_printer_if_env() {
    if let Ok(a) = std::env::var("SUPASIM_LOG_FULL_TRACE") {
        if &a != "0" {
            setup_trace_printer();
        }
    }
}
#[macro_export]
macro_rules! all_backend_tests_inner {
    ($func_name:ident, $backend_name:literal, $instance_create:block, $test_name:ident, $hal_backend: ident) => {
        #[test]
        pub fn $func_name() {
            if std::env::var(concat!("SUPASIM_SKIP_BACKEND_", $backend_name))
                .is_ok_and(|a| &a != "0" && &a != "false" && !a.is_empty())
            {
                return;
            }
            log::info!("{} test", $backend_name);
            let instance = $instance_create;
            let instance = instance.expect(&format!("Failed to create {} instance", $backend_name));
            log::info!("Created {} instance", $backend_name);
            $test_name::<hal::$hal_backend>(instance).unwrap();
        }
    };
}
#[macro_export]
macro_rules! all_backend_tests {
    ($test_name:ident) => {
        $crate::paste::paste! {
            $crate::all_backend_tests_inner!([<$test_name _dummy>], "DUMMY", {
                hal::Dummy::create_instance()
            }, $test_name, Dummy);

            #[cfg(feature = "vulkan")]
            $crate::all_backend_tests_inner!([<$test_name _vulkan>], "VULKAN", {
                hal::Vulkan::create_instance(true)
            }, $test_name, Vulkan);

            #[cfg(feature = "wgpu")]
            $crate::all_backend_tests_inner!([<$test_name _wgpu_vulkan>], "WGPU_VULKAN", {
                hal::wgpu::Wgpu::create_instance(true, hal::wgpu::Backends::VULKAN, None)
            }, $test_name, Wgpu);

            #[cfg(all(feature = "wgpu", target_vendor = "apple"))]
            $crate::all_backend_tests_inner!([<$test_name _wgpu_metal>], "WGPU_METAL", {
                hal::wgpu::Wgpu::create_instance(true, hal::wgpu::Backends::METAL, None)
            }, $test_name, Wgpu);

            #[cfg(all(feature = "wgpu", target_os = "windows"))]
            $crate::all_backend_tests_inner!([<$test_name _wgpu_dx12>], "WGPU_DX12", {
                hal::wgpu::Wgpu::create_instance(true, hal::wgpu::Backends::DX12, None)
            }, $test_name, Wgpu);
        }
    };
}
