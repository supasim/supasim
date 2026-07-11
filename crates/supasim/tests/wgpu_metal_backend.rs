/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! Integration test binary: wgpu + Metal backend.

mod common;

fn main() {
    use dev_utils::testing::run_backend_tests;

    #[cfg(all(feature = "wgpu", target_vendor = "apple"))]
    {
        use supasim::hal;
        let availability = hal::wgpu::Wgpu::create_instance(true, hal::wgpu::Backends::METAL, None)
            .map(|_| ())
            .map_err(|e| e.to_string());
        let cases = common::all_tests(|| {
            hal::wgpu::Wgpu::create_instance(true, hal::wgpu::Backends::METAL, None).unwrap()
        });
        run_backend_tests("wgpu_metal", availability, cases);
    }

    #[cfg(not(all(feature = "wgpu", target_vendor = "apple")))]
    run_backend_tests(
        "wgpu_metal",
        Err("wgpu+Metal requires the wgpu feature on Apple platforms".to_string()),
        common::skipped_tests("wgpu+Metal requires the wgpu feature on Apple platforms"),
    );
}
