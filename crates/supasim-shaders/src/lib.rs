/* BEGIN LICENSE
  SupaSim, a GPUGPU and simulation toolkit.
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
#[cfg(test)]
mod tests;

use anyhow::{Result, anyhow};
use rand::Rng;
use slang::Downcast;
#[allow(unused_imports)]
use std::ptr::null_mut;
use std::{ffi::CString, io::Write, path::Path, str::FromStr};
use tempfile::tempdir;
use types::ShaderReflectionInfo;
pub use types::{ShaderModel, ShaderTarget, SpirvVersion};

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
pub type ShaderCompileError = anyhow::Error;
pub struct GlobalState {
    slang_session: slang::GlobalSession,
}
impl GlobalState {
    #[cfg(feature = "opt-valid")]
    fn env_from_version(s: SpirvVersion) -> spirv_tools_sys::spv_target_env {
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
    fn validate_spv(module: &[u32], s: SpirvVersion) -> Result<()> {
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
    fn optimize_spv(module: &[u32], s: SpirvVersion) -> Result<Vec<u8>> {
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
    #[cfg(feature = "spirv-cross")]
    fn transpile_spirv(module: &[u32], target: ShaderTarget) -> Result<Vec<u8>> {
        use std::ptr::null;

        use spirv_cross_sys as spvc;
        unsafe {
            let mut context = null_mut();
            spvc::spvc_context_create(&mut context);
            let mut ir = null_mut();
            spvc::spvc_context_parse_spirv(context, module.as_ptr(), module.len(), &mut ir);
            let mut compiler = null_mut();
            spvc::spvc_context_create_compiler(
                context,
                match target {
                    ShaderTarget::Glsl => spvc::spvc_backend_SPVC_BACKEND_GLSL,
                    ShaderTarget::Msl { .. } | ShaderTarget::MetalLib { .. } => {
                        spvc::spvc_backend_SPVC_BACKEND_MSL
                    }
                    ShaderTarget::Hlsl => spvc::spvc_backend_SPVC_BACKEND_HLSL,
                    _ => panic!("Transpile spirv called on invalid target"),
                },
                ir,
                spvc::spvc_capture_mode_SPVC_CAPTURE_MODE_TAKE_OWNERSHIP,
                &mut compiler,
            );
            let mut options = null_mut();
            spvc::spvc_compiler_create_compiler_options(compiler, &mut options);
            spvc::spvc_compiler_install_compiler_options(compiler, options);
            let mut result = null();
            spvc::spvc_compiler_compile(compiler, &mut result);
            let result =
                String::from(std::ffi::CStr::from_ptr(result).to_str().unwrap()).into_bytes();
            spvc::spvc_context_destroy(context);
            Ok(result)
        }
    }
    #[cfg(all(target_os = "macos", feature = "msl-stable-out"))]
    fn compile_metallib(module: &[u8], temp_dir: &Path) -> Result<Vec<u8>> {
        use std::process::Command;
        let inter_path = format!("{}/inter.ir", temp_dir.to_str().unwrap());
        let out_path = format!("{}/out.metallib", temp_dir.to_str().unwrap());
        let mut metal = Command::new("xcrun")
            .args([
                "-sdk",
                "macosx",
                "metal",
                "-x", // The input doesn't have extension, so specify it is MSL
                "metal",
                "-c", // Preprocess, compile, assemble
                "-",  // Stdin input
                "-o",
                &inter_path,
            ])
            .stdin(std::process::Stdio::piped())
            .spawn()?;
        metal.stdin.as_mut().unwrap().write_all(module)?;
        metal.stdin.as_mut().unwrap().flush()?;
        metal.stdin = None; // Close stdin to signal end of input
        let output = metal.wait_with_output()?;
        if !output.status.success() {
            return Err(anyhow!(
                "Metal compilation failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        let output = Command::new("xcrun")
            .args(["-sdk", "macosx", "metallib", &inter_path, "-o", &out_path])
            .output()?;
        if !output.status.success() {
            return Err(anyhow!(
                "Metal library linking failed: {}",
                String::from_utf8_lossy(&output.stderr)
            ));
        }
        Ok(std::fs::read(&out_path)?)
    }
    pub fn new_from_env() -> Result<Self> {
        let global_session = slang::GlobalSession::new().unwrap();
        Ok(Self {
            slang_session: global_session,
        })
    }
    pub fn compile_shader(&self, options: ShaderCompileOptions) -> Result<ShaderReflectionInfo> {
        let extra_optim = options.opt_level == OptimizationLevel::Maximal || options.minify;
        let extra_valid = options.stability == StabilityGuarantee::ExtraValidation;
        let (target, needs_spirv_transpile) = match options.target {
            ShaderTarget::Ptx => (slang::CompileTarget::Ptx, false),
            ShaderTarget::CudaCpp => (slang::CompileTarget::CudaSource, false),
            ShaderTarget::Msl { .. } | ShaderTarget::MetalLib { .. } => {
                if extra_optim || options.stability != StabilityGuarantee::Experimental {
                    (slang::CompileTarget::Spirv, true)
                } else {
                    (
                        if let ShaderTarget::MetalLib { .. } = options.target {
                            slang::CompileTarget::MetalLib
                        } else {
                            slang::CompileTarget::Metal
                        },
                        false,
                    )
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
            ShaderTarget::Dxil { .. } => (slang::CompileTarget::Hlsl, false),
        };
        if options.target.metal_version().is_some() && needs_spirv_transpile {
            #[cfg(not(feature = "msl-stable-out"))]
            {
                return Err(anyhow!(
                    "Shader compiler was not compiled with MSL or MetalLib support"
                ));
            }
        } else if options.target == ShaderTarget::Wgsl {
            #[cfg(not(feature = "wgsl-out"))]
            {
                return Err(anyhow!(
                    "Shader compiler was not compiled with WGSL support"
                ));
            }
        } else if matches!(options.target, ShaderTarget::Dxil { .. }) {
            #[cfg(not(feature = "dxil-out"))]
            {
                return Err(anyhow!(
                    "Shader compiler was not compiled with DXIL support"
                ));
            }
        }
        if needs_spirv_transpile && (extra_optim || extra_valid) {
            #[cfg(not(feature = "opt-valid"))]
            {
                return Err(anyhow!(
                    "Shader compiler was not compiled with support for extra validation or optimization"
                ));
            }
        }
        if needs_spirv_transpile {
            #[cfg(not(feature = "spirv-cross"))]
            {
                return Err(anyhow!(
                    "Shader compiler was not compiled with advanced cross compilation support"
                ));
            }
        }
        if matches!(options.target, ShaderTarget::MetalLib { .. }) {
            #[cfg(not(target_os = "macos"))]
            return Err(anyhow!("MetalLib compilation is only supported on macOS"));
        }
        let mut _tempdir = None;
        let (source_file, _source, search_dir) = match options.source {
            ShaderSource::File(f) => (
                f.to_owned(),
                std::fs::File::open(f)?,
                CString::from_str(f.parent().unwrap().to_str().unwrap())?,
            ),
            ShaderSource::Memory(m) => {
                let tempdir = tempdir()?;
                let mut path = tempdir.path().to_owned();
                _tempdir = Some(tempdir);
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
        let mut profile = slang::ProfileID::UNKNOWN;
        let mut opt = slang::CompilerOptions::default()
            .language(slang::SourceLanguage::Slang)
            .optimization(optim)
            .target(target)
            .stage(slang::Stage::Compute);
        if options.minify {
            opt = opt
                .debug_information(slang::DebugInfoLevel::None)
                .no_mangle(false)
                .obfuscate(true);
        }
        if let Some(include) = options.include {
            opt = opt.include(include);
        }
        if let ShaderTarget::Spirv { version } = options.target {
            profile = self.slang_session.find_profile(match version {
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
            });
        } else if target == slang::CompileTarget::Spirv {
            profile = self.slang_session.find_profile("spirv_1_0");
        } else if target == slang::CompileTarget::Dxil {
            if let ShaderTarget::Dxil { shader_model } = options.target {
                profile = self.slang_session.find_profile(shader_model.to_str());
            } else {
                unreachable!();
            }
        } else if target == slang::CompileTarget::Metal || target == slang::CompileTarget::MetalLib
        {
            opt = opt.capability(
                self.slang_session
                    .find_capability(options.target.metal_version().unwrap().to_str()),
            )
        }
        if !profile.is_unknown() {
            opt = opt.profile(profile);
        }
        let session = self
            .slang_session
            .create_session(
                &slang::SessionDesc::default()
                    .targets(&[slang::TargetDesc::default().format(target).profile(profile)])
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
        let mut workgroup_size = [0; 3];
        let linked_program = program.link()?;
        {
            let layout = program.layout(0)?;
            //layout.global_params_type_layout();
            assert!(layout.entry_point_count() == 1);
            let ep = layout.entry_point_by_index(0).unwrap();
            let wgs = ep.compute_thread_group_size();
            for i in 0..3 {
                workgroup_size[i] = wgs[i] as u32;
            }
            /*for p in ep.parameters() {
                println!(
                    "Parameter: bind-index {} bind-space {} sem-name {:?} category {:?} var-name {:?}",
                    p.binding_index(),
                    p.binding_space(),
                    p.semantic_name(),
                    p.category(),
                    p.variable().name()
                )
            }
            for p in layout.global_params_type_layout().fields() {
                println!(
                    "Parameter: bind-index {} bind-space {} sem-name {:?} category {:?} var-name {:?}",
                    p.binding_index(),
                    p.binding_space(),
                    p.semantic_name(),
                    p.category(),
                    p.variable().name()
                )
            }*/
            /*
            As it stands:
            We need to find all global params, and filter out the ones unused for a given entry point
            We need to find all entry point params
            We need to traverse the tree of structures to find all the actual buffers/whatever
            We need to figure out which of these have bindings, then collect those
            */
        }
        let bytecode = linked_program.entry_point_code(0, 0)?;
        let mut _other_blob = Vec::<u8>::new();

        #[allow(unused_mut)] // In case no features, so this doesn't get flagged
        let mut data = bytecode.as_slice();

        #[allow(unused_variables)]
        let spirv_version = match options.target {
            ShaderTarget::Spirv { version } => version,
            _ => SpirvVersion::V1_0,
        };

        #[cfg(feature = "opt-valid")]
        if extra_valid && needs_spirv_transpile {
            let vec = bytecode.as_slice().to_owned();
            Self::validate_spv(bytemuck::cast_slice(&vec), spirv_version)?;
        }
        #[cfg(feature = "opt-valid")]
        if extra_optim && needs_spirv_transpile {
            let vec = bytecode.as_slice().to_owned();
            _other_blob = Self::optimize_spv(bytemuck::cast_slice(&vec), spirv_version)?;
            data = &_other_blob;
        }

        #[cfg(feature = "msl-stable-out")]
        if needs_spirv_transpile {
            if let Some(version) = options.target.metal_version() {
                let vec = bytecode.as_slice().to_owned();
                let res = Self::transpile_spirv(
                    bytemuck::cast_slice(&vec),
                    ShaderTarget::Msl { version },
                )?;
                _other_blob = res;
                data = &_other_blob;
            }
            #[cfg(target_os = "macos")]
            if let ShaderTarget::MetalLib { .. } = options.target {
                if _tempdir.is_none() {
                    _tempdir = Some(tempdir()?);
                }
                let res = Self::compile_metallib(data, _tempdir.unwrap().path())?;
                _other_blob = res;
                data = &_other_blob;
            }
        }
        #[cfg(feature = "wgsl-out")]
        if options.target == ShaderTarget::Wgsl && needs_spirv_transpile {
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
            _other_blob = naga::back::wgsl::write_string(
                &module,
                &info,
                naga::back::wgsl::WriterFlags::empty(),
            )?
            .into_bytes();
            data = &_other_blob;
        }

        #[cfg(feature = "spirv-cross")]
        if options.target == ShaderTarget::Glsl && needs_spirv_transpile {
            let vec = bytecode.as_slice().to_owned();
            let res = Self::transpile_spirv(bytemuck::cast_slice(&vec), ShaderTarget::Glsl)?;
            _other_blob = res;
            data = &_other_blob;
        }
        #[cfg(feature = "dxil-out")]
        if let ShaderTarget::Dxil { shader_model } = options.target {
            let dxil = hassle_rs::compile_hlsl(
                "intermediate.hlsl",
                unsafe { std::str::from_utf8_unchecked(data) },
                options.entry,
                shader_model.to_str(),
                &[],
                &[],
            )?;
            let dxil = hassle_rs::validate_dxil(&dxil)?;
            _other_blob = dxil;
            data = &_other_blob;
        }

        match options.dest {
            ShaderDest::File(out) => {
                std::fs::write(out, data)?;
            }
            ShaderDest::Memory(out) => {
                out.write_all(data)?;
            }
        }
        Ok(ShaderReflectionInfo {
            entry_name: options.entry.to_owned(),
            workgroup_size,
            resources: Vec::new(),
            push_constant_len: 0,
        })
    }
}
