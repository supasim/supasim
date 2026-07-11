/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! Integration test binary: Vulkan backend.
//!
//! All tests are always registered.  When the backend is unavailable at
//! runtime (or not compiled) every trial is reported as *ignored* rather than
//! silently absent.

mod common;

fn main() {
    use dev_utils::testing::run_backend_tests;

    #[cfg(feature = "vulkan")]
    {
        use supasim::hal;
        let availability = hal::Vulkan::create_instance(true)
            .map(|_| ())
            .map_err(|e| e.to_string());
        let cases = common::all_tests(|| hal::Vulkan::create_instance(true).unwrap());
        run_backend_tests("vulkan", availability, cases);
    }

    #[cfg(not(feature = "vulkan"))]
    run_backend_tests(
        "vulkan",
        Err("vulkan feature not compiled".to_string()),
        common::skipped_tests("vulkan feature not compiled"),
    );
}
