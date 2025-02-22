use std::{path::PathBuf, str::FromStr};

use anyhow::Result;
use supasim_shaders::*;

pub fn main() -> Result<()> {
    let ctx = supasim_shaders::GlobalState::new_from_env()?;
    let target = if true {
        ShaderTarget::Dxil {
            shader_model: ShaderModel::Sm6_1,
        }
    } else {
        ShaderTarget::Spirv {
            version: SpirvVersion::V1_2,
        }
    };
    ctx.compile_shader(ShaderCompileOptions {
        target,
        source: ShaderSource::Memory(include_bytes!("test.slang")),
        dest: ShaderDest::File(&PathBuf::from_str("test.ptx").unwrap()),
        entry: "computeMain",
        include: None,
        fp_mode: Default::default(),
        opt_level: OptimizationLevel::Maximal,
        stability: StabilityGuarantee::Experimental,
        minify: true,
    })?;
    Ok(())
}
