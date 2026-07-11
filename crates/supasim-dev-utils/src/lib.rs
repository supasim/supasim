/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

pub mod testing;
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
    if let Ok(a) = std::env::var("SUPASIM_LOG_FULL_TRACE")
        && &a != "0"
    {
        setup_trace_printer();
    }
    let _ = env_logger::builder()
        .filter_level(log::LevelFilter::Info)
        .parse_default_env()
        .try_init();
}
