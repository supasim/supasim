use anyhow::{anyhow, Result};
use rand::Rng;
use slang::Downcast;
#[cfg(feature = "spirv_cross")]
use spirv_cross::spirv;
use std::{ffi::CString, io::Write, path::Path, str::FromStr};
use tempfile::tempdir;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum SpirvVersion {
    V1_0,
    V1_1,
    V1_2,
    V1_3,
    #[default]
    V1_4,
    V1_5,
    V1_6,
    Cl1_2,
    Cl2_0,
    Cl2_1,
    Cl2_2,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShaderModel {
    #[default]
    Sm6_0,
    Sm6_1,
    Sm6_2,
    Sm6_3,
    Sm6_4,
    Sm6_5,
    Sm6_6,
    Sm6_7,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderTarget {
    CudaCpp,
    Ptx {},
    Spirv { version: SpirvVersion },
    Msl,
    Hlsl,
    Glsl,
    Wgsl,
    Dxil { shader_model: ShaderModel },
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum StabilityGuarantee {
    ExtraValidation,
    #[default]
    Stable,
    Experimental,
}
pub enum ShaderSource<'a> {
    File(&'a Path),
    Memory(&'a [u8]),
}
pub enum ShaderDest<'a> {
    File(&'a Path),
    Memory(&'a mut Vec<u8>),
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum ShaderFpMode {
    Fast,
    #[default]
    Precise,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum OptimizationLevel {
    None,
    #[default]
    Standard,
    Maximal,
}
pub struct ShaderCompileOptions<'a> {
    pub target: ShaderTarget,
    pub source: ShaderSource<'a>,
    pub dest: ShaderDest<'a>,
    pub entry: &'a str,
    pub include: Option<&'a str>,
    pub fp_mode: ShaderFpMode,
    pub opt_level: OptimizationLevel,
    pub stability: StabilityGuarantee,
    pub minify: bool,
}
pub struct GlobalState {
    slang_session: slang::GlobalSession,
}
impl GlobalState {
    #[cfg(feature = "opt-valid")]
    pub fn env_from_version(s: SpirvVersion) -> u32 {
        match s {
            SpirvVersion::Cl1_2 => spirv_tools_sys::spv_target_env_SPV_ENV_OPENCL_1_2,
            SpirvVersion::Cl2_0 => spirv_tools_sys::spv_target_env_SPV_ENV_OPENCL_2_0,
            SpirvVersion::Cl2_1 => spirv_tools_sys::spv_target_env_SPV_ENV_OPENCL_2_1,
            SpirvVersion::Cl2_2 => spirv_tools_sys::spv_target_env_SPV_ENV_OPENCL_2_2,
            SpirvVersion::V1_0 => spirv_tools_sys::spv_target_env_SPV_ENV_UNIVERSAL_1_0,
            SpirvVersion::V1_1 => spirv_tools_sys::spv_target_env_SPV_ENV_UNIVERSAL_1_1,
            SpirvVersion::V1_2 => spirv_tools_sys::spv_target_env_SPV_ENV_UNIVERSAL_1_2,
            SpirvVersion::V1_3 => spirv_tools_sys::spv_target_env_SPV_ENV_UNIVERSAL_1_3,
            SpirvVersion::V1_4 => spirv_tools_sys::spv_target_env_SPV_ENV_UNIVERSAL_1_4,
            SpirvVersion::V1_5 => spirv_tools_sys::spv_target_env_SPV_ENV_UNIVERSAL_1_5,
            SpirvVersion::V1_6 => spirv_tools_sys::spv_target_env_SPV_ENV_UNIVERSAL_1_6,
        }
    }
    #[cfg(feature = "opt-valid")]
    pub fn validate_spv(module: &[u32], s: SpirvVersion) -> Result<()> {
        use std::{ffi::CStr, ptr::null_mut};

        unsafe {
            let env = Self::env_from_version(s);
            let options = spirv_tools_sys::spvValidatorOptionsCreate();
            let ctx = spirv_tools_sys::spvContextCreate(env);
            let mut binary = spirv_tools_sys::spv_const_binary_t {
                code: module.as_ptr() as *mut u32,
                wordCount: module.len(),
            };
            let mut diagnostic = null_mut();
            let res =
                spirv_tools_sys::spvValidateWithOptions(ctx, options, &mut binary, &mut diagnostic);
            let mut out = Ok(());
            if res < 0 {
                let diagnostic = &*diagnostic;
                out = Err(anyhow!(
                    "Validation error: {}",
                    CStr::from_ptr(diagnostic.error as *const i8).to_str()?
                ));
            }
            if diagnostic.is_null() {
                spirv_tools_sys::spvDiagnosticDestroy(diagnostic);
            }
            spirv_tools_sys::spvContextDestroy(ctx);
            spirv_tools_sys::spvValidatorOptionsDestroy(options);
            out
        }
    }
    // TODO: needs to know whether to minify or do other optimizations
    #[cfg(feature = "opt-valid")]
    pub fn optimize_spv(module: &[u32], s: SpirvVersion) -> Result<Vec<u8>> {
        use std::{ffi::c_void, ptr::null_mut};

        unsafe {
            let env = Self::env_from_version(s);
            let optim = spirv_tools_sys::spvOptimizerCreate(env);
            let options = spirv_tools_sys::spvOptimizerOptionsCreate();
            // TODO: Set options here
            let mut optimized = null_mut();
            let res = spirv_tools_sys::spvOptimizerRun(
                optim,
                module.as_ptr(),
                module.len(),
                &mut optimized,
                options,
            );
            if res < 0 {
                return Err(anyhow!("Error in optimization"));
            }
            let v = {
                let optimized = &mut *optimized;
                Vec::from_raw_parts(
                    optimized.code as *mut u8,
                    optimized.wordCount * 4,
                    optimized.wordCount * 4,
                )
            };
            libc::free(optimized as *mut c_void);
            spirv_tools_sys::spvOptimizerDestroy(optim);
            spirv_tools_sys::spvOptimizerOptionsDestroy(options);
            Ok(v)
        }
    }
    pub fn new_from_env() -> Result<Self> {
        let global_session = slang::GlobalSession::new().unwrap();
        Ok(Self {
            slang_session: global_session,
        })
    }
    pub fn compile_shader(&self, options: ShaderCompileOptions) -> Result<()> {
        let extra_optim = options.opt_level == OptimizationLevel::Maximal || options.minify;
        let extra_valid = options.stability == StabilityGuarantee::ExtraValidation;
        let (target, needs_further_transpile) = match options.target {
            ShaderTarget::Ptx {} => (slang::CompileTarget::Ptx, false),
            ShaderTarget::CudaCpp => (slang::CompileTarget::CudaSource, false),
            ShaderTarget::Msl => {
                if extra_optim || options.stability != StabilityGuarantee::Experimental {
                    (slang::CompileTarget::Spirv, true)
                } else {
                    (slang::CompileTarget::Metal, false)
                }
            }
            ShaderTarget::Spirv { .. } => (slang::CompileTarget::Spirv, false),
            ShaderTarget::Wgsl => (slang::CompileTarget::Spirv, true),
            ShaderTarget::Glsl => {
                if extra_optim || extra_valid {
                    (slang::CompileTarget::Spirv, true)
                } else {
                    (slang::CompileTarget::Glsl, false)
                }
            }
            ShaderTarget::Hlsl => (slang::CompileTarget::Hlsl, false),
            ShaderTarget::Dxil { .. } => (slang::CompileTarget::Dxil, false),
        };
        if options.target == ShaderTarget::Msl && needs_further_transpile {
            #[cfg(not(feature = "msl-stable-out"))]
            {
                return Err(anyhow!("Shader compiler was not compiled with MSL support"));
            }
        } else if options.target == ShaderTarget::Wgsl {
            #[cfg(not(feature = "wgsl-out"))]
            {
                return Err(anyhow!(
                    "Shader compiler was not compiled with WGSL support"
                ));
            }
        }
        if needs_further_transpile && (extra_optim || extra_valid) {
            #[cfg(not(feature = "opt-valid"))]
            {
                return Err(anyhow!(
                    "Shader compiler was not compiled with support for extra validation or optimization"
                ));
            }
        }
        if needs_further_transpile {
            #[cfg(not(feature = "spirv_cross"))]
            {
                return Err(anyhow!(
                    "Shader compiler was not compiled with advanced cross compilation support"
                ));
            }
        }
        let _tempdir;
        let (source_file, _source, search_dir) = match options.source {
            ShaderSource::File(f) => (
                f.to_owned(),
                std::fs::File::open(f)?,
                CString::from_str(f.parent().unwrap().to_str().unwrap())?,
            ),
            ShaderSource::Memory(m) => {
                _tempdir = tempdir()?;
                let mut path = _tempdir.path().to_owned();
                path.push(format!("{}.tmp.slang", rand::rng().random::<u64>()));
                let mut f = std::fs::File::create_new(&path)?;

                f.write_all(m)?;
                f.flush()?;
                (
                    path.clone(),
                    f,
                    CString::from_str(path.parent().unwrap().to_str().unwrap())?,
                )
            }
        };
        let optim = match options.opt_level {
            OptimizationLevel::None => slang::OptimizationLevel::None,
            OptimizationLevel::Standard => slang::OptimizationLevel::Default,
            OptimizationLevel::Maximal => slang::OptimizationLevel::Maximal,
        };
        let mut opt = slang::CompilerOptions::default()
            .language(slang::SourceLanguage::Slang)
            .optimization(optim)
            .target(target);
        if let Some(include) = options.include {
            opt = opt.include(include);
        }
        if let ShaderTarget::Spirv { version } = options.target {
            opt = opt.profile(self.slang_session.find_profile(match version {
                SpirvVersion::V1_0 => "spirv_1_0",
                SpirvVersion::V1_1 => "spirv_1_1",
                SpirvVersion::V1_2 => "spirv_1_2",
                SpirvVersion::V1_3 => "spirv_1_3",
                SpirvVersion::V1_4 => "spirv_1_4",
                SpirvVersion::V1_5 => "spirv_1_5",
                SpirvVersion::V1_6 => "spirv_1_6",
                // Pulled from chatgpt, aka my ass
                SpirvVersion::Cl1_2 | SpirvVersion::Cl2_0 => unimplemented!(),
                SpirvVersion::Cl2_1 => "spirv_1_0",
                SpirvVersion::Cl2_2 => "spirv_1_2",
            }));
        }
        if let ShaderTarget::Dxil { shader_model } = options.target {
            use ShaderModel::*;
            let profile = self.slang_session.find_profile(match shader_model {
                Sm6_0 => "cs_6_0",
                Sm6_1 => "cs_6_1",
                Sm6_2 => "cs_6_2",
                Sm6_3 => "cs_6_3",
                Sm6_4 => "cs_6_4",
                Sm6_5 => "cs_6_5",
                Sm6_6 => "cs_6_6",
                Sm6_7 => "cs_6_7",
            });
            opt = opt.profile(profile);
        }
        let session = self
            .slang_session
            .create_session(
                &slang::SessionDesc::default()
                    .targets(&[slang::TargetDesc::default().format(target)])
                    .options(&opt)
                    .search_paths(&[search_dir.as_ptr()]),
            )
            .ok_or(anyhow!("Failed to create slang session"))?;
        let module = session.load_module(
            source_file
                .file_name()
                .unwrap()
                .to_str()
                .ok_or(anyhow!("Module path is non-utf8"))?,
        )?;
        let ep = module
            .find_entry_point_by_name(options.entry)
            .ok_or(anyhow!("Entry point not found"))?;
        let program = session
            .create_composite_component_type(&[module.downcast().clone(), ep.downcast().clone()])?;
        let linked_program = program.link()?;
        let bytecode = linked_program.entry_point_code(0, 0)?;
        let mut _stringcode = String::new();
        let mut _other_blob = Vec::<u8>::new();

        #[allow(unused_mut)]
        let mut data = bytecode.as_slice();

        #[cfg(feature = "opt-valid")]
        if extra_valid && needs_further_transpile {
            Self::validate_spv(bytemuck::cast_slice(data), SpirvVersion::V1_0)?;
        }
        #[cfg(feature = "opt-valid")]
        if extra_optim && needs_further_transpile {
            _other_blob = Self::optimize_spv(bytemuck::cast_slice(data), SpirvVersion::V1_0)?;
            data = &_other_blob;
        }

        if options.target == ShaderTarget::Msl {
            #[cfg(feature = "msl-stable-out")]
            {
                let vec = bytecode.as_slice().to_owned();
                let module = spirv::Module::from_words(bytemuck::cast_slice(&vec));
                _stringcode = spirv::Ast::<spirv_cross::msl::Target>::parse(&module)?.compile()?;
                data = _stringcode.as_bytes();
            }
        } else if options.target == ShaderTarget::Wgsl {
            #[cfg(feature = "wgsl-out")]
            {
                let module = naga::front::spv::parse_u8_slice(
                    bytecode.as_slice(),
                    &naga::front::spv::Options {
                        adjust_coordinate_space: true,
                        strict_capabilities: false,
                        block_ctx_dump_prefix: None,
                    },
                )?;
                module.to_ctx();
                let mut valid = naga::valid::Validator::new(
                    naga::valid::ValidationFlags::all(),
                    naga::valid::Capabilities::all(),
                );
                let info = valid.validate(&module)?;
                _stringcode = naga::back::wgsl::write_string(
                    &module,
                    &info,
                    naga::back::wgsl::WriterFlags::empty(),
                )?;
                data = _stringcode.as_bytes();
            }
        } else if options.target == ShaderTarget::Glsl {
            #[cfg(feature = "spirv_cross")]
            {
                let vec = bytecode.as_slice().to_owned();
                let module = spirv::Module::from_words(bytemuck::cast_slice(&vec));
                _stringcode = spirv::Ast::<spirv_cross::glsl::Target>::parse(&module)?.compile()?;
                data = _stringcode.as_bytes();
            }
        }

        match options.dest {
            ShaderDest::File(out) => {
                std::fs::write(out, data)?;
            }
            ShaderDest::Memory(out) => {
                out.write_all(data)?;
            }
        }
        Ok(())
    }
}
