use std::path::PathBuf;

use types::{MetalVersion, ShaderTarget};

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
            let code = include_bytes!("../test.slang");
            let ctx = crate::GlobalState::new_from_env().unwrap();
            std::fs::create_dir_all("shader-tests").unwrap();
            if std::fs::exists("shader-tests/{target_name}").unwrap() {
                std::fs::remove_file("shader-tests/{target_name}").unwrap();
            }
            let mut dest_file = PathBuf::new();
            dest_file.push("shader-tests");
            dest_file.push($filename);
            ctx.compile_shader(crate::ShaderCompileOptions {
                target: $target,
                source: crate::ShaderSource::Memory(code),
                dest: crate::ShaderDest::File(&dest_file),
                entry: "computeMain",
                include: None,
                fp_mode: crate::ShaderFpMode::Precise,
                opt_level: crate::OptimizationLevel::Maximal,
                stability: crate::StabilityGuarantee::ExtraValidation,
                minify: true,
            })
            .unwrap();
        }
    };
}
shader_test!(glsl_test, "GLSL", "test.glsl", ShaderTarget::Glsl);
shader_test!(
    spirv_1_0_test,
    "SPIRV_1_0",
    "test.spv.1_0",
    ShaderTarget::Spirv {
        version: types::SpirvVersion::V1_0
    }
);
shader_test!(
    spirv_1_4_test,
    "SPIRV_1_4",
    "test.spv.1_4",
    ShaderTarget::Spirv {
        version: types::SpirvVersion::V1_4
    }
);
shader_test!(
    msl_test,
    "MSL",
    "test.metal",
    ShaderTarget::Msl {
        version: MetalVersion::V2_3
    }
);
#[cfg(target_os = "macos")]
shader_test!(
    metallib_test,
    "METALLIB",
    "test.metallib",
    ShaderTarget::MetalLib {
        version: MetalVersion::V2_3
    }
);
shader_test!(wgsl_test, "WGSL", "test.wgsl", ShaderTarget::Wgsl);
shader_test!(cuda_test, "CUDA", "test.cu", ShaderTarget::CudaCpp);
shader_test!(ptx_test, "PTX", "test.ptx", ShaderTarget::Ptx);
shader_test!(hlsl_test, "HLSL", "test.hlsl", ShaderTarget::Hlsl);
shader_test!(
    dxil_test,
    "DXIL",
    "test.dxil",
    ShaderTarget::Dxil {
        shader_model: types::ShaderModel::Sm6_7
    }
);

/*#[test]
pub fn shader_tests_main() {
    let code = include_bytes!("../test.slang");
    let ctx = crate::GlobalState::new_from_env().unwrap();
    if std::fs::exists("shader-tests").unwrap() {
        std::fs::remove_dir_all("shader-tests").unwrap();
    }
    std::fs::create_dir("shader-tests").unwrap();

    let mut targets = vec![
        ShaderTarget::Spirv {
            version: types::SpirvVersion::V1_0,
        },
        ShaderTarget::Spirv {
            version: types::SpirvVersion::V1_4,
        },
        ShaderTarget::Msl,
        ShaderTarget::Wgsl,
        ShaderTarget::CudaCpp,
        ShaderTarget::Hlsl,
    ];
    if !std::env::var("SUPASIM_SKIP_SHADERS_DXIL")
        .is_ok_and(|a| &a != "0" && &a != "false" && !a.is_empty())
    {
        targets.push(ShaderTarget::Dxil {
            shader_model: types::ShaderModel::Sm6_7,
        });
    }
    if !std::env::var("SUPASIM_SKIP_SHADERS_PTX")
        .is_ok_and(|a| &a != "0" && &a != "false" && !a.is_empty())
    {
        targets.push(ShaderTarget::Ptx);
    }
    for target in targets {
        println!("Target: {target:?}");
        let mut dest_file = PathBuf::new();
        dest_file.push("shader-tests");
        dest_file.push(format!("test.{}", target.file_extension()));
        ctx.compile_shader(crate::ShaderCompileOptions {
            target,
            source: crate::ShaderSource::Memory(code),
            dest: crate::ShaderDest::File(&dest_file),
            entry: "computeMain",
            include: None,
            fp_mode: crate::ShaderFpMode::Precise,
            opt_level: crate::OptimizationLevel::Maximal,
            stability: crate::StabilityGuarantee::ExtraValidation,
            minify: true,
        })
        .unwrap();
    }
}
*/
