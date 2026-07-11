/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! Shared test bodies for integration test binaries.
//!
//! Each function is generic over `B: hal::Backend` and takes an already-created
//! `hal::InstanceDescriptor<B>`.  The per-backend binaries monomorphise them and
//! hand the resulting closures to `dev_utils::testing::run_backend_tests`.
//!
//! `all_tests` is the central registry — add new tests here.

#![allow(unused_imports, dead_code)]

use supasim::{
    Buffer, BufferDescriptor, BufferSlice, Instance,
    hal::{self, Backend, InstanceDescriptor},
    kernels,
};
use dev_utils::testing::TestCase;

// ── helpers ──────────────────────────────────────────────────────────────────

fn make_instance<B: Backend>(desc: InstanceDescriptor<B>) -> Instance<B> {
    Instance::from_hal(desc)
}

fn mk_buf<B: Backend>(instance: &Instance<B>) -> Buffer<B> {
    instance
        .create_buffer(&BufferDescriptor {
            size: 16,
            contents_align: 4,
            ..Default::default()
        })
        .unwrap()
}

// ── test bodies ───────────────────────────────────────────────────────────────

fn add_numbers<B: Backend>(desc: InstanceDescriptor<B>) -> Result<(), String> {
    dev_utils::setup_trace_printer_if_env();
    let instance = make_instance(desc);

    let download_buffer = mk_buf(&instance);
    let buffer1 = mk_buf(&instance);
    let buffer2 = mk_buf(&instance);
    let buffer3 = mk_buf(&instance);

    let global_state = kernels::GlobalState::new_from_env().unwrap();
    let mut spirv = Vec::new();
    let reflection_info = global_state
        .compile_kernel(kernels::KernelCompileOptions {
            target: instance.properties().unwrap().kernel_lang,
            source: kernels::KernelSource::Memory(include_bytes!(
                "../../../../kernels/test_add.slang"
            )),
            dest: kernels::KernelDest::Memory(&mut spirv),
            entry: "add",
            include: None,
            fp_mode: kernels::KernelFpMode::Precise,
            opt_level: kernels::OptimizationLevel::Maximal,
            stability: kernels::StabilityGuarantee::ExtraValidation,
            minify: false,
        })
        .map_err(|e| e.to_string())?;
    assert!(reflection_info.buffers == vec![false, false, true]);
    let kernel = instance
        .compile_raw_kernel(&spirv, reflection_info)
        .map_err(|e| e.to_string())?;

    let recorder = instance.create_recorder().map_err(|e| e.to_string())?;
    recorder.write_buffer::<u32>(&buffer1, 0, &[1, 2, 3, 4]).map_err(|e| e.to_string())?;
    recorder.write_buffer::<u32>(&buffer2, 0, &[5, 6, 7, 8]).map_err(|e| e.to_string())?;
    recorder.write_buffer::<u32>(&buffer3, 0, &[1, 1, 1, 1]).map_err(|e| e.to_string())?;
    recorder
        .dispatch_kernel(
            &kernel,
            &[
                &BufferSlice::entire_buffer(&buffer1, false).map_err(|e| e.to_string())?,
                &BufferSlice::entire_buffer(&buffer2, false).map_err(|e| e.to_string())?,
                &BufferSlice::entire_buffer(&buffer3, true).map_err(|e| e.to_string())?,
            ],
            [4, 1, 1],
        )
        .map_err(|e| e.to_string())?;
    recorder
        .copy_buffer(&buffer3, &download_buffer, 0, 0, 16)
        .map_err(|e| e.to_string())?;
    instance.submit_commands(&[recorder]).map_err(|e| e.to_string())?;

    assert_eq!(
        download_buffer
            .access(0, 16, false)
            .map_err(|e| e.to_string())?
            .readable::<u32>()
            .map_err(|e| e.to_string())?,
        [6, 8, 10, 12]
    );
    Ok(())
}

fn basic_buffer_copy<B: Backend>(desc: InstanceDescriptor<B>) -> Result<(), String> {
    dev_utils::setup_trace_printer_if_env();
    let instance = make_instance(desc);

    let upload_buffer = mk_buf(&instance);
    let gpu_buffer = mk_buf(&instance);
    let download_buffer = mk_buf(&instance);

    upload_buffer.write::<u32>(0, &[1, 2, 3, 4]).map_err(|e| e.to_string())?;
    let recorder = instance.create_recorder().map_err(|e| e.to_string())?;
    recorder
        .copy_buffer(&upload_buffer, &gpu_buffer, 0, 0, 16)
        .map_err(|e| e.to_string())?;
    recorder
        .copy_buffer(&gpu_buffer, &download_buffer, 0, 0, 16)
        .map_err(|e| e.to_string())?;
    instance.submit_commands(&[recorder]).map_err(|e| e.to_string())?;
    instance.wait_for_idle(1.0).map_err(|e| e.to_string())?;

    assert_eq!(
        download_buffer
            .access(0, 16, false)
            .map_err(|e| e.to_string())?
            .readable::<u32>()
            .map_err(|e| e.to_string())?,
        [1, 2, 3, 4]
    );
    Ok(())
}

/// Regression test for PR #93 review finding #1: a live CPU read mapping held
/// open across a conflicting GPU write must not panic on drop.
fn concurrent_map_gpu_use<B: Backend>(desc: InstanceDescriptor<B>) -> Result<(), String> {
    dev_utils::setup_trace_printer_if_env();
    let instance = make_instance(desc);

    let src = mk_buf(&instance);
    let dst = mk_buf(&instance);
    src.write::<u32>(0, &[1, 2, 3, 4]).map_err(|e| e.to_string())?;

    let mapping = dst.access(0, 16, false).map_err(|e| e.to_string())?;

    let r = instance.create_recorder().map_err(|e| e.to_string())?;
    r.copy_buffer(&src, &dst, 0, 0, 16).map_err(|e| e.to_string())?;
    let w = instance.submit_commands(&[r]).map_err(|e| e.to_string())?;
    w.wait().map_err(|e| e.to_string())?;

    drop(mapping);
    instance.wait_for_idle(1.0).map_err(|e| e.to_string())?;

    assert_eq!(
        dst.access(0, 16, false)
            .map_err(|e| e.to_string())?
            .readable::<u32>()
            .map_err(|e| e.to_string())?,
        [1, 2, 3, 4]
    );
    Ok(())
}

/// Regression test B2: two submissions touching the same buffer/range must
/// not hang and must produce correct data.
fn double_submit_same_buffer<B: Backend>(desc: InstanceDescriptor<B>) -> Result<(), String> {
    dev_utils::setup_trace_printer_if_env();
    let instance = make_instance(desc);

    let a1 = mk_buf(&instance);
    let b1 = mk_buf(&instance);
    let a2 = mk_buf(&instance);
    let b2 = mk_buf(&instance);
    let scratch = mk_buf(&instance);
    let download = mk_buf(&instance);

    let global_state = kernels::GlobalState::new_from_env().unwrap();
    let mut spirv = Vec::new();
    let reflection_info = global_state
        .compile_kernel(kernels::KernelCompileOptions {
            target: instance.properties().unwrap().kernel_lang,
            source: kernels::KernelSource::Memory(include_bytes!(
                "../../../../kernels/test_add.slang"
            )),
            dest: kernels::KernelDest::Memory(&mut spirv),
            entry: "add",
            include: None,
            fp_mode: kernels::KernelFpMode::Precise,
            opt_level: kernels::OptimizationLevel::Maximal,
            stability: kernels::StabilityGuarantee::ExtraValidation,
            minify: false,
        })
        .map_err(|e| e.to_string())?;
    let kernel = instance
        .compile_raw_kernel(&spirv, reflection_info)
        .map_err(|e| e.to_string())?;

    a1.write::<u32>(0, &[1, 2, 3, 4]).map_err(|e| e.to_string())?;
    b1.write::<u32>(0, &[10, 20, 30, 40]).map_err(|e| e.to_string())?;
    a2.write::<u32>(0, &[100, 200, 300, 400]).map_err(|e| e.to_string())?;
    b2.write::<u32>(0, &[1000, 2000, 3000, 4000]).map_err(|e| e.to_string())?;

    let r1 = instance.create_recorder().map_err(|e| e.to_string())?;
    r1.dispatch_kernel(
        &kernel,
        &[
            &BufferSlice::entire_buffer(&a1, false).map_err(|e| e.to_string())?,
            &BufferSlice::entire_buffer(&b1, false).map_err(|e| e.to_string())?,
            &BufferSlice::entire_buffer(&scratch, true).map_err(|e| e.to_string())?,
        ],
        [4, 1, 1],
    )
    .map_err(|e| e.to_string())?;
    r1.copy_buffer(&scratch, &download, 0, 0, 16).map_err(|e| e.to_string())?;
    let w1 = instance.submit_commands(&[r1]).map_err(|e| e.to_string())?;
    w1.wait().map_err(|e| e.to_string())?;

    let r2 = instance.create_recorder().map_err(|e| e.to_string())?;
    r2.dispatch_kernel(
        &kernel,
        &[
            &BufferSlice::entire_buffer(&a2, false).map_err(|e| e.to_string())?,
            &BufferSlice::entire_buffer(&b2, false).map_err(|e| e.to_string())?,
            &BufferSlice::entire_buffer(&scratch, true).map_err(|e| e.to_string())?,
        ],
        [4, 1, 1],
    )
    .map_err(|e| e.to_string())?;
    r2.copy_buffer(&scratch, &download, 0, 0, 16).map_err(|e| e.to_string())?;
    let w2 = instance.submit_commands(&[r2]).map_err(|e| e.to_string())?;
    w2.wait().map_err(|e| e.to_string())?;
    instance.wait_for_idle(1.0).map_err(|e| e.to_string())?;

    assert_eq!(
        download
            .access(0, 16, false)
            .map_err(|e| e.to_string())?
            .readable::<u32>()
            .map_err(|e| e.to_string())?,
        [1100, 2200, 3300, 4400]
    );
    Ok(())
}

/// Regression test B3/B4: reading through a mutable CPU mapping must see the
/// GPU's last write.
fn mut_map_readback<B: Backend>(desc: InstanceDescriptor<B>) -> Result<(), String> {
    dev_utils::setup_trace_printer_if_env();
    let instance = make_instance(desc);

    let a = mk_buf(&instance);
    let b = mk_buf(&instance);
    let out = mk_buf(&instance);

    let global_state = kernels::GlobalState::new_from_env().unwrap();
    let mut spirv = Vec::new();
    let reflection_info = global_state
        .compile_kernel(kernels::KernelCompileOptions {
            target: instance.properties().unwrap().kernel_lang,
            source: kernels::KernelSource::Memory(include_bytes!(
                "../../../../kernels/test_add.slang"
            )),
            dest: kernels::KernelDest::Memory(&mut spirv),
            entry: "add",
            include: None,
            fp_mode: kernels::KernelFpMode::Precise,
            opt_level: kernels::OptimizationLevel::Maximal,
            stability: kernels::StabilityGuarantee::ExtraValidation,
            minify: false,
        })
        .map_err(|e| e.to_string())?;
    let kernel = instance
        .compile_raw_kernel(&spirv, reflection_info)
        .map_err(|e| e.to_string())?;

    a.write::<u32>(0, &[1, 2, 3, 4]).map_err(|e| e.to_string())?;
    b.write::<u32>(0, &[10, 20, 30, 40]).map_err(|e| e.to_string())?;

    let r = instance.create_recorder().map_err(|e| e.to_string())?;
    r.dispatch_kernel(
        &kernel,
        &[
            &BufferSlice::entire_buffer(&a, false).map_err(|e| e.to_string())?,
            &BufferSlice::entire_buffer(&b, false).map_err(|e| e.to_string())?,
            &BufferSlice::entire_buffer(&out, true).map_err(|e| e.to_string())?,
        ],
        [4, 1, 1],
    )
    .map_err(|e| e.to_string())?;
    let w = instance.submit_commands(&[r]).map_err(|e| e.to_string())?;
    w.wait().map_err(|e| e.to_string())?;
    instance.wait_for_idle(1.0).map_err(|e| e.to_string())?;

    {
        let mapping = out.access(0, 16, true).map_err(|e| e.to_string())?;
        assert_eq!(mapping.readable::<u32>().map_err(|e| e.to_string())?, [11, 22, 33, 44]);
    }
    {
        let mut mapping = out.access(0, 16, true).map_err(|e| e.to_string())?;
        for v in mapping.writable::<u32>().map_err(|e| e.to_string())? {
            *v *= 2;
        }
    }
    let mut got = [0u32; 4];
    out.read(0, &mut got).map_err(|e| e.to_string())?;
    assert_eq!(got, [22, 44, 66, 88]);
    Ok(())
}

// ── registry ──────────────────────────────────────────────────────────────────

/// Canonical test names — used to register skip-only stubs when a backend
/// feature is not compiled so tests remain visible in nextest output.
pub const TEST_NAMES: &[&str] = &[
    "add_numbers",
    "basic_buffer_copy",
    "concurrent_map_gpu_use",
    "double_submit_same_buffer",
    "mut_map_readback",
];

/// Build a fully-monomorphised test list for backend `B`.
///
/// `create` is called fresh for every trial (nextest gives each trial its own
/// process, so creating a new instance per test is correct and expected).
pub fn all_tests<B: Backend + 'static>(
    create: impl Fn() -> InstanceDescriptor<B> + Clone + Send + 'static,
) -> Vec<TestCase> {
    macro_rules! case {
        ($name:ident) => {{
            let c = create.clone();
            TestCase::new(stringify!($name), move || $name::<B>(c()))
        }};
    }
    vec![
        case!(add_numbers),
        case!(basic_buffer_copy),
        case!(concurrent_map_gpu_use),
        case!(double_submit_same_buffer),
        case!(mut_map_readback),
    ]
}

/// Build skip-only stubs for every test, used when the backend feature is not
/// compiled.  The tests are still listed by nextest and shown as ignored.
pub fn skipped_tests(reason: &'static str) -> Vec<TestCase> {
    TEST_NAMES
        .iter()
        .map(|&name| TestCase::new(name, move || Err(reason.to_string())))
        .collect()
}
