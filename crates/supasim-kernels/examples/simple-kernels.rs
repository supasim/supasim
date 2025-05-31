/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 SupaMaggie70 (Magnus Larsson)


  SupaSim is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 3
  of the License, or (at your option) any later version.

  SupaSim is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
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
