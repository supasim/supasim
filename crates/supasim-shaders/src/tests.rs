use std::path::PathBuf;

use types::ShaderTarget;

#[test]
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
        ShaderTarget::Dxil {
            shader_model: types::ShaderModel::Sm6_7,
        },
        ShaderTarget::Wgsl,
    ];
    if unsafe { libloading::Library::new(libloading::library_filename("nvrtc")).is_ok() } {
        targets.push(ShaderTarget::Ptx);
    } else {
        targets.push(ShaderTarget::CudaCpp);
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
