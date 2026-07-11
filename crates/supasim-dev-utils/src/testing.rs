/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

//! Test harness utilities for `harness = false` integration test binaries.
//!
//! Backends are probed at runtime; tests are marked ignored rather than
//! silently absent when a backend is unavailable or its feature flag is not
//! compiled in.
//!
//! **Do not use `skip()` from `harness = true` (`#[test]`) functions** — exit
//! code 51 is only recognised by nextest for `libtest_mimic` binaries.  In
//! `#[test]` functions, use `#[cfg_attr(not(feature = "…"), ignore)]` for
//! compile-time skips and an early `return` for runtime unavailability.

use libtest_mimic::{Arguments, Failed, Trial};

/// A single named test case whose body has been fully monomorphised.
///
/// Build one per test function via [`TestCase::new`], collect them in a
/// `Vec`, and hand the whole collection to [`run_backend_tests`].
pub struct TestCase {
    name: &'static str,
    run: Box<dyn Fn() -> Result<(), String> + Send>,
}

impl TestCase {
    pub fn new<F>(name: &'static str, run: F) -> Self
    where
        F: Fn() -> Result<(), String> + Send + 'static,
    {
        Self {
            name,
            run: Box::new(run),
        }
    }
}

/// Signal to nextest that this test should be skipped.
///
/// Exits with code 51, which nextest treats as an ignored/skipped outcome.
/// Use when a backend is unavailable or its feature flag is not compiled.
pub fn skip(reason: &str) -> ! {
    eprintln!("SKIP: {reason}");
    std::process::exit(51)
}

/// Drive a set of test cases against a single backend using `libtest_mimic`.
///
/// `backend_name` — human-readable label used only in skip messages.
/// `availability` — `Ok(())` if the backend is usable; `Err(reason)` to mark
///   every trial ignored with the given reason.
/// `cases` — the test cases to run; each carries its own backend-creating
///   closure so nextest can give each trial its own process.
///
/// This function never returns — call it as the last thing in `main()`.
pub fn run_backend_tests(
    backend_name: &'static str,
    availability: Result<(), String>,
    cases: Vec<TestCase>,
) -> ! {
    let args = Arguments::from_args();

    let trials: Vec<Trial> = cases
        .into_iter()
        .map(|case| match &availability {
            Err(reason) => {
                let msg = format!("backend '{backend_name}' unavailable: {reason}");
                Trial::test(case.name, move || Err(Failed::from(msg))).with_ignored_flag(true)
            }
            Ok(()) => Trial::test(case.name, move || (case.run)().map_err(Failed::from)),
        })
        .collect();

    libtest_mimic::run(&args, trials).exit()
}
