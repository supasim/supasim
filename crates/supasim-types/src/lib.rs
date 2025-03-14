use serde::{Deserialize, Serialize};

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub enum MemoryType {
    /// Driver decides
    #[default]
    Any,
    /// Only used on GPU
    GpuOnly,
    /// Best for uploading data to GPU
    Upload,
    /// Best for downloading data from GPU
    Download,
    /// Both upload and download supported, optimized for download
    UploadDownload,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct BufferDescriptor {
    pub size: u64,
    pub memory_type: MemoryType,
    pub mapped_at_creation: bool,
    pub visible_to_renderer: bool,
    pub indirect_capable: bool,
    pub transfer_src: bool,
    pub transfer_dst: bool,
    pub uniform: bool,
    pub needs_flush: bool,
}

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
impl ShaderModel {
    pub fn to_str(&self) -> &str {
        use ShaderModel::*;
        match self {
            Sm6_0 => "cs_6_0",
            Sm6_1 => "cs_6_1",
            Sm6_2 => "cs_6_2",
            Sm6_3 => "cs_6_3",
            Sm6_4 => "cs_6_4",
            Sm6_5 => "cs_6_5",
            Sm6_6 => "cs_6_6",
            Sm6_7 => "cs_6_7",
        }
    }
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
#[derive(Clone, Copy, Debug)]
pub struct InstanceProperties {
    pub needs_explicit_sync: bool,
    pub indirect: bool,
    pub pipeline_cache: bool,
    pub shader_type: ShaderTarget,
}
/// # Safety
/// This is undefined behavior lol
pub unsafe fn to_static_lifetime<T>(r: &T) -> &'static T
where
    T: ?Sized,
{
    unsafe {
        let r = r as *const T;
        &*r
    }
}
/// # Safety
/// This is undefined behavior lol
pub unsafe fn to_static_lifetime_mut<T>(r: &mut T) -> &'static mut T {
    unsafe {
        let r = r as *mut T;
        &mut *r
    }
}

#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub enum ShaderResourceType {
    #[default]
    Unknown,
    Buffer,
    UniformBuffer,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct ShaderReflectionInfo {
    pub workgroup_size: [u32; 3],
    pub entry_name: String,
    pub resources: Vec<ShaderResourceType>,
    pub push_constant_len: u32,
}
