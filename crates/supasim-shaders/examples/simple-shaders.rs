use std::{path::PathBuf, str::FromStr};

use anyhow::Result;
use supasim_shaders::*;

pub fn main() -> Result<()> {
    let ctx = supasim_shaders::GlobalState::new_from_env()?;

    let target = {
        //ShaderTarget::Glsl
        /*ShaderTarget::Dxil {
            shader_model: ShaderModel::Sm6_1,
        }*/
        ShaderTarget::Dxil {
            shader_model: ShaderModel::Sm6_7,
        }
    };
    ctx.compile_shader(ShaderCompileOptions {
        target,
        source: ShaderSource::Memory(include_bytes!("../test.slang")),
        dest: ShaderDest::File(&PathBuf::from_str("test.dxil").unwrap()),
        entry: "computeMain",
        include: None,
        fp_mode: Default::default(),
        opt_level: OptimizationLevel::Standard,
        stability: StabilityGuarantee::Experimental,
        minify: false,
    })?;
    Ok(())
}
