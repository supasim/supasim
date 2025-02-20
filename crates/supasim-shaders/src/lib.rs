use anyhow::{anyhow, Result};
use rand::Rng;
use slang::Downcast;
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
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct VulkanCapabilities {
    version: SpirvVersion,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum ShaderTarget {
    Ptx {},
    VulkanSpv { capabilities: VulkanCapabilities },
    OpenClSpv,
    Hip,
    Msl,
}
pub enum ShaderSource<'a> {
    File(&'a Path),
    Memory(&'a [u8]),
}
pub enum ShaderDest<'a> {
    File(&'a Path),
    Memory(&'a mut Vec<u8>),
}
#[derive(Clone, Copy, Debug, Default)]
pub enum ShaderFpMode {
    Fast,
    #[default]
    Precise,
}
pub enum OptimizationLevel {
    None,
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
}
pub struct GlobalState {
    slang_session: slang::GlobalSession,
}
impl GlobalState {
    pub fn new_from_env() -> Result<Self> {
        let global_session = slang::GlobalSession::new().unwrap();
        Ok(Self {
            slang_session: global_session,
        })
    }
    pub fn compile_shader(&self, options: ShaderCompileOptions) -> Result<()> {
        if options.target == ShaderTarget::Msl {
            #[cfg(not(feature = "msl-out"))]
            {
                return Err(anyhow!("Shader compiler was not compiled with MSL support"));
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
        let target = match options.target {
            ShaderTarget::Ptx {} | ShaderTarget::Hip => slang::CompileTarget::Ptx,
            ShaderTarget::Msl | ShaderTarget::OpenClSpv | ShaderTarget::VulkanSpv { .. } => {
                slang::CompileTarget::Spirv
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
        if let ShaderTarget::VulkanSpv {
            capabilities: VulkanCapabilities { version },
        } = options.target
        {
            opt = opt.capability(self.slang_session.find_capability(match version {
                SpirvVersion::V1_0 => "spirv_1_0",
                SpirvVersion::V1_1 => "spirv_1_1",
                SpirvVersion::V1_2 => "spirv_1_2",
                SpirvVersion::V1_3 => "spirv_1_3",
                SpirvVersion::V1_4 => "spirv_1_4",
                SpirvVersion::V1_5 => "spirv_1_5",
                SpirvVersion::V1_6 => "spirv_1_6",
            }));
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

        let mut data = bytecode.as_slice();

        if options.target == ShaderTarget::Msl {
            #[cfg(feature = "msl-out")]
            {
                let vec = bytecode.as_slice().to_owned();
                let module = spirv::Module::from_words(bytemuck::cast_slice(&vec));
                _stringcode = spirv::Ast::<spirv_cross::msl::Target>::parse(&module)?.compile()?;
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
