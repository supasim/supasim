/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use std::{path::PathBuf, str::FromStr};

use anyhow::Result;
use supasim_kernels::*;

pub fn main() -> Result<()> {
    let ctx = GlobalState::new_from_env()?;

    let target = {
        //KernelTarget::Glsl
        /*KernelTarget::Dxil {
            shader_model: ShaderModel::Sm6_1,
        }*/
        KernelTarget::Dxil {
            shader_model: ShaderModel::Sm6_7,
        }
    };
    ctx.compile_kernel(KernelCompileOptions {
        target,
        source: KernelSource::Memory(include_bytes!("../../../kernels/test_add.slang")),
        dest: KernelDest::File(&PathBuf::from_str("test.dxil").unwrap()),
        entry: "add",
        include: None,
        fp_mode: Default::default(),
        opt_level: OptimizationLevel::Standard,
        stability: StabilityGuarantee::Experimental,
        minify: false,
    })?;
    Ok(())
}
