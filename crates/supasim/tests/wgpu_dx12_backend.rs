/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! Integration test binary: wgpu + DX12 backend.

mod common;

fn main() {
    use dev_utils::testing::run_backend_tests;

    #[cfg(all(feature = "wgpu", target_os = "windows"))]
    {
        use supasim::hal;
        let availability = hal::wgpu::Wgpu::create_instance(true, hal::wgpu::Backends::DX12, None)
            .map(|_| ())
            .map_err(|e| e.to_string());
        let cases = common::all_tests(|| {
            hal::wgpu::Wgpu::create_instance(true, hal::wgpu::Backends::DX12, None).unwrap()
        });
        run_backend_tests("wgpu_dx12", availability, cases);
    }

    #[cfg(not(all(feature = "wgpu", target_os = "windows")))]
    run_backend_tests(
        "wgpu_dx12",
        Err("wgpu+DX12 requires the wgpu feature on Windows".to_string()),
        common::skipped_tests("wgpu+DX12 requires the wgpu feature on Windows"),
    );
}
