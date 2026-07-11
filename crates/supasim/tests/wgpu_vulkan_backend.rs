/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! Integration test binary: wgpu + Vulkan backend.

mod common;

fn main() {
    use dev_utils::testing::run_backend_tests;

    #[cfg(feature = "wgpu")]
    {
        use supasim::hal;
        let availability = hal::wgpu::Wgpu::create_instance(true, hal::wgpu::Backends::VULKAN, None)
            .map(|_| ())
            .map_err(|e| e.to_string());
        let cases = common::all_tests(|| {
            hal::wgpu::Wgpu::create_instance(true, hal::wgpu::Backends::VULKAN, None).unwrap()
        });
        run_backend_tests("wgpu_vulkan", availability, cases);
    }

    #[cfg(not(feature = "wgpu"))]
    run_backend_tests(
        "wgpu_vulkan",
        Err("wgpu feature not compiled".to_string()),
        common::skipped_tests("wgpu feature not compiled"),
    );
}
