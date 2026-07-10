/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! Regression test for major finding #3/#4 (PR #93 review): reading through a
//! **mutable** CPU mapping (`access(.., true)` then `readable()`) must observe the
//! buffer's current data, i.e. a read-modify-write via a mapping must see what the GPU
//! last wrote.
//!
//! Repro path:
//!  * A dispatch writes `out = a + b` on the *device* buffer. `out`'s host copy is left
//!    stale.
//!  * `out.access(0, 16, true)` calls `get_cpu_access(is_mut = true)`, which
//!    INVALIDATES every device tracker and marks the HOST tracker CURRENT *without
//!    copying device -> host* (a mutable access is assumed to overwrite). So
//!    `ensure_host_current` then skips the readback copy, and the mapping exposes the
//!    stale host buffer.
//!  * `MappedBuffer::new` additionally maps in the access direction
//!    (`mutable = needs_mut`), so on non-unified memory (e.g. wgpu over llvmpipe) the
//!    `MapMode::Write` mapping returns stale bytes even if a copy had happened
//!    (`buffer/access.rs`; `MappedBuffer::readable` has no `has_mut` guard).
//!
//! Result: `readable::<u32>()` on the mutable mapping returns zeroes/old data instead
//! of the GPU result `[11, 22, 33, 44]`.
//!
//! Expected fix: either fetch device -> host for the read half of a mutable mapping
//! (and map `MapMode::Read` for the fill), or make `readable()` on a non-unified
//! mutable mapping an error. If the maintainer decides mutable mappings are
//! write-only by contract, this test should be changed to assert `readable()` returns
//! `Err`, not silently-stale data. Against the current tree the `assert_eq!` fails.

use crate::*;

#[allow(unused)]
pub fn mut_map_readback<Backend: hal::Backend>(
    hal: hal::InstanceDescriptor<Backend>,
) -> Result<(), ()> {
    dev_utils::setup_trace_printer_if_env();

    let instance = Instance::<Backend>::from_hal(hal);

    let mk = || {
        instance
            .create_buffer(&BufferDescriptor {
                size: 16,
                contents_align: 4,
                ..Default::default()
            })
            .unwrap()
    };
    let a = mk();
    let b = mk();
    let out = mk();

    // Compile test_add: result = buffer0 + buffer1 (result is the mutable binding 2).
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
        .unwrap();
    let kernel = instance
        .compile_raw_kernel(&spirv, reflection_info)
        .unwrap();

    a.write::<u32>(0, &[1, 2, 3, 4]).unwrap();
    b.write::<u32>(0, &[10, 20, 30, 40]).unwrap();

    // GPU: out = a + b, written to `out`'s device buffer. Host copy of `out` is stale.
    let r = instance.create_recorder().unwrap();
    r.dispatch_kernel(
        &kernel,
        &[
            &BufferSlice::entire_buffer(&a, false).unwrap(),
            &BufferSlice::entire_buffer(&b, false).unwrap(),
            &BufferSlice::entire_buffer(&out, true).unwrap(),
        ],
        [4, 1, 1],
    )
    .unwrap();
    let w = instance.submit_commands(&[r]).unwrap();
    w.wait().unwrap();
    instance.wait_for_idle(1.0).unwrap();

    // Read the GPU result through a MUTABLE mapping (the read half of a read-modify-write).
    // BUG #3/#4: returns stale data because the device->host readback was skipped for
    // the mutable access.
    let mapping = out.access(0, 16, true).unwrap();
    assert_eq!(mapping.readable::<u32>().unwrap(), [11, 22, 33, 44]);
    Ok(())
}

dev_utils::all_backend_tests!(mut_map_readback);
