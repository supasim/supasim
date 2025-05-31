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
use std::path::PathBuf;

pub fn should_skip(shader_target: &str) -> bool {
    std::env::var(format!("SUPASIM_SKIP_SHADERS_{shader_target}"))
        .is_ok_and(|a| &a != "0" && &a != "false" && !a.is_empty())
}

macro_rules! shader_test {
    ($func_name:ident, $target_name:literal, $filename:literal, $target:expr) => {
        #[test]
        pub fn $func_name() {
            if should_skip($target_name) {
                return;
            }
            let code = include_bytes!("../../../kernels/test_add.slang");
            let ctx = crate::GlobalState::new_from_env().unwrap();
            std::fs::create_dir_all("shader-tests").unwrap();
            let mut dest_file = PathBuf::new();
            dest_file.push("shader-tests");
            dest_file.push($filename);
            let reflect = ctx
                .compile_shader(crate::ShaderCompileOptions {
                    target: $target,
                    source: crate::ShaderSource::Memory(code),
                    dest: crate::ShaderDest::File(&dest_file),
                    entry: "add",
                    include: None,
                    fp_mode: crate::ShaderFpMode::Precise,
                    opt_level: crate::OptimizationLevel::Maximal,
                    stability: crate::StabilityGuarantee::ExtraValidation,
                    minify: true,
                })
                .unwrap();
            dest_file.set_file_name(format!("{}.reflect.json", $filename));
            std::fs::write(dest_file, serde_json::to_string(&reflect).unwrap()).unwrap();
        }
    };
}
#[cfg(feature = "opt-valid")]
shader_test!(add_glsl, "GLSL", "test.glsl", types::ShaderTarget::Glsl);
#[cfg(feature = "opt-valid")]
shader_test!(
    add_spirv_1_0,
    "SPIRV_1_0",
    "test.spv.1_0",
    types::ShaderTarget::Spirv {
        version: types::SpirvVersion::V1_0
    }
);
#[cfg(feature = "opt-valid")]
shader_test!(
    add_spirv_1_4,
    "SPIRV_1_4",
    "test.spv.1_4",
    types::ShaderTarget::Spirv {
        version: types::SpirvVersion::V1_4
    }
);
#[cfg(feature = "opt-valid")]
shader_test!(
    add_msl,
    "MSL",
    "test.metal",
    types::ShaderTarget::Msl {
        version: types::MetalVersion::V2_3
    }
);
#[cfg(all(target_os = "macos", feature = "opt-valid"))]
shader_test!(
    add_metallib,
    "METALLIB",
    "test.metallib",
    types::ShaderTarget::MetalLib {
        version: types::MetalVersion::V2_3
    }
);
#[cfg(all(feature = "wgsl-out", feature = "opt-valid"))]
shader_test!(add_wgsl, "WGSL", "test.wgsl", types::ShaderTarget::Wgsl);
shader_test!(add_cuda, "CUDA", "test.cu", types::ShaderTarget::CudaCpp);
// This test must be skipped manually if it fails
shader_test!(add_ptx_test, "PTX", "test.ptx", types::ShaderTarget::Ptx);
#[cfg(feature = "opt-valid")]
shader_test!(add_hlsl, "HLSL", "test.hlsl", types::ShaderTarget::Hlsl);
#[cfg(all(feature = "dxil-out", feature = "opt-valid"))]
shader_test!(
    add_dxil,
    "DXIL",
    "test.dxil",
    types::ShaderTarget::Dxil {
        shader_model: types::ShaderModel::Sm6_7
    }
);
