/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! Regression test for major finding #1 (PR #93 review): a GPU submission that
//! conflicts with a still-live CPU buffer access must not panic when that access is
//! released.
//!
//! Repro path:
//!  1. A live CPU **read** mapping of `dst[0,16)` is held open. `get_cpu_access`
//!     records its `BufferAccessFinish` in `read_accesses` with
//!     `device_semaphore == None`.
//!  2. A GPU submission **writes** the same `dst[0,16)` range (`copy_buffer src->dst`).
//!     `BufferResidency::add_gpu_use(is_mut = true)` scans `read_accesses`, finds the
//!     live read finish, and — because it has no semaphore yet — manufactures one with
//!     `device_stream_submission: Some((u16::MAX, u16::MAX, u64::MAX))`
//!     (`buffer/residency.rs` / `buffer/ood.rs`).
//!  3. Dropping the mapping calls `release_cpu_access`, which does `sem.signal()` on
//!     that manufactured semaphore. `Semaphore::signal` asserts
//!     `device_stream_submission.is_none()` (`sync/mod.rs:107`) — but it is `Some(..)`,
//!     so the assert fires and the process panics.
//!
//! A host-signalled semaphore must carry `device_stream_submission: None`. The fix is
//! to construct the manufactured placeholders with `None` (or to not attach a HAL
//! semaphore to a CPU access at all). Against the current tree this test panics inside
//! `Drop for MappedBuffer`; after the fix it completes and the post-drop readback of
//! `dst` observes the GPU copy's result.
//!
//! Backend-agnostic: the GPU only ever touches `dst`'s *device* buffer while the CPU
//! holds the *host* buffer mapped, and `ensure_device_current(dst)` performs no
//! host->device copy here (dst's host copy is never marked current), so there is no
//! "buffer mapped during submission" hazard on wgpu.

use crate::*;

#[allow(unused)]
pub fn concurrent_map_gpu_use<Backend: hal::Backend>(
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
    let src = mk();
    let dst = mk();
    src.write::<u32>(0, &[1, 2, 3, 4]).unwrap();

    // Hold a live CPU READ mapping of `dst` open across the GPU submission below.
    let mapping = dst.access(0, 16, false).unwrap();

    // GPU submission that WRITES the same range of `dst`, conflicting with `mapping`.
    let r = instance.create_recorder().unwrap();
    r.copy_buffer(&src, &dst, 0, 0, 16).unwrap();
    let w = instance.submit_commands(&[r]).unwrap();
    w.wait().unwrap();

    // BUG #1: this drop panics in `release_cpu_access` (`Semaphore::signal` assert).
    // After the fix it returns cleanly.
    drop(mapping);

    instance.wait_for_idle(1.0).unwrap();

    // Post-fix sanity: the completed GPU copy is visible on readback.
    assert_eq!(
        dst.access(0, 16, false)
            .unwrap()
            .readable::<u32>()
            .unwrap(),
        [1, 2, 3, 4]
    );
    Ok(())
}

dev_utils::all_backend_tests!(concurrent_map_gpu_use);
