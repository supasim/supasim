/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! B2 regression test: two separate `submit_commands` calls that touch the SAME
//! buffer/range must not hang, and must produce correct data.
//!
//! The B2 fix has two parts, both exercised here:
//!  1. `OutOfDateTracker::get_needed_waits` no longer manufactures an `extra_copy`
//!     placeholder carrying a fresh GPU timeline semaphore that NOTHING ever signals
//!     (a per-submission HAL semaphore leak, and — if that stale placeholder were
//!     ever waited — a deadlock; that literal wait path turned out to be unreachable
//!     on this branch, but the leak and the tracker poisoning below were real).
//!  2. `BufferResidency::add_gpu_use` now marks the accessed device's own OOD tracker
//!     CURRENT after a use (`update_range_immediate`) instead of INVALIDATING it. The
//!     old `invalidate_range` marked the just-written device copy stale on the very
//!     device that wrote it; that was only ever masked by the placeholder from (1),
//!     whose `update_range_delayed` re-marked the range current. With the placeholder
//!     removed, the stale marking became visible: the device->host readback saw
//!     `device_current == false`, skipped the copy, and returned stale/zeroed data.
//!
//! Observed before/after on this branch (lavapipe): with BOTH parts of the B2 fix
//! reverted, this test FAILS deterministically — after the second submission the
//! readback of `download` is wrong (it stays at the first submission's result, or is
//! zero on a fresh readback). After the fix it PASSES deterministically across
//! repeated runs. (A panic probe on the placeholder-sentinel semaphore confirmed the
//! literal deadlock path is never actually `wait()`-ed on this branch; the reachable,
//! and equally blocking-for-correctness, manifestation is the stale/zeroed readback
//! exercised here.)
//!
//! `download` (the readback buffer) is written by TWO separate submissions over the
//! same [0,16) range: each submission runs a `test_add` dispatch into `scratch` then
//! copies `scratch -> download`. No buffer is both read and written within one
//! dispatch (which wgpu's validator rejects). This drives the full residency path:
//! device buffer creation, host->device input uploads, the same-range write on the
//! second submission, and the device->host readback whose sync the bug used to skip.

use crate::*;

#[allow(unused)]
pub fn double_submit_same_buffer<Backend: hal::Backend>(
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
    // Inputs for the two submissions, a scratch dispatch target, plus the shared
    // `download` buffer that BOTH submissions copy their result into over the same
    // [0,16) range (the readback path proven by `add_numbers`).
    let a1 = mk();
    let b1 = mk();
    let a2 = mk();
    let b2 = mk();
    let scratch = mk();
    let download = mk();

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

    // Seed inputs.
    a1.write::<u32>(0, &[1, 2, 3, 4]).unwrap();
    b1.write::<u32>(0, &[10, 20, 30, 40]).unwrap();
    a2.write::<u32>(0, &[100, 200, 300, 400]).unwrap();
    b2.write::<u32>(0, &[1000, 2000, 3000, 4000]).unwrap();

    // FIRST submission: scratch = a1 + b1, then copy scratch -> download [0,16).
    let r1 = instance.create_recorder().unwrap();
    r1.dispatch_kernel(
        &kernel,
        &[
            &BufferSlice::entire_buffer(&a1, false).unwrap(),
            &BufferSlice::entire_buffer(&b1, false).unwrap(),
            &BufferSlice::entire_buffer(&scratch, true).unwrap(),
        ],
        [4, 1, 1],
    )
    .unwrap();
    r1.copy_buffer(&scratch, &download, 0, 0, 16).unwrap();
    let w1 = instance.submit_commands(&[r1]).unwrap();
    w1.wait().unwrap();

    // SECOND submission touching the SAME buffer/range `download` [0,16) (and reusing
    // `scratch`). Before the fix, the placeholder left in these buffers' device OOD
    // `current_copies` by the first submission poisoned this submission's residency
    // sync, intermittently yielding stale `download` data on readback (the first
    // submission's [11,22,33,44]).
    let r2 = instance.create_recorder().unwrap();
    r2.dispatch_kernel(
        &kernel,
        &[
            &BufferSlice::entire_buffer(&a2, false).unwrap(),
            &BufferSlice::entire_buffer(&b2, false).unwrap(),
            &BufferSlice::entire_buffer(&scratch, true).unwrap(),
        ],
        [4, 1, 1],
    )
    .unwrap();
    r2.copy_buffer(&scratch, &download, 0, 0, 16).unwrap();
    let w2 = instance.submit_commands(&[r2]).unwrap();
    w2.wait().unwrap();

    instance.wait_for_idle(1.0).unwrap();

    // download must hold the SECOND submission's result: [100,200,300,400]+[1000,..].
    assert_eq!(
        download
            .access(0, 16, false)
            .unwrap()
            .readable::<u32>()
            .unwrap(),
        [1100, 2200, 3300, 4400]
    );
    Ok(())
}

dev_utils::all_backend_tests!(double_submit_same_buffer);
