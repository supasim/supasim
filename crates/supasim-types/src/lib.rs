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
use serde::{Deserialize, Serialize};

pub use daggy::Walker;
pub use daggy::petgraph::algo::toposort;
pub use daggy::petgraph::graph::NodeIndex;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum HalBufferType {
    /// Driver decides
    #[default]
    Any,
    /// Used for kernel access. Memory type on GPU that can be copied around on GPU but is optimized for local access.
    Storage,
    /// Best for uploading data to GPU. Memory type on GPU that can be written to from CPU and copied from on GPU.
    Upload,
    /// Best for downloading data from GPU. Memory type on GPU that can be copied to from GPU and read from CPU.
    Download,
    /// Can be copied to from GPU and used in other use cases such as uniform buffers.
    Other,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HalBufferDescriptor {
    pub size: u64,
    pub memory_type: HalBufferType,
    pub visible_to_renderer: bool,
    pub indirect_capable: bool,
    pub uniform: bool,
    pub needs_flush: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
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
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub enum MetalVersion {
    #[default]
    V2_3,
    V3_1,
}
impl MetalVersion {
    pub fn to_str(&self) -> &str {
        use MetalVersion::*;
        match self {
            V2_3 => "metallib_2_3",
            V3_1 => "metallib_3_1",
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum KernelTarget {
    CudaCpp,
    Ptx,
    Wgsl,
    Glsl,
    Spirv { version: SpirvVersion },
    Hlsl,
    Dxil { shader_model: ShaderModel },
    Msl { version: MetalVersion },
    MetalLib { version: MetalVersion },
}
impl KernelTarget {
    pub fn file_extension(&self) -> &str {
        use KernelTarget::*;
        match self {
            CudaCpp => "cu",
            Ptx => "ptx",
            Spirv { .. } => "spv",
            Msl { .. } => "metal",
            Hlsl => "hlsl",
            Glsl => "glsl",
            Wgsl => "wgsl",
            Dxil { .. } => "dxil",
            MetalLib { .. } => "metallib",
        }
    }
    pub fn metal_version(&self) -> Option<MetalVersion> {
        match self {
            Self::MetalLib { version } | Self::Msl { version } => Some(*version),
            _ => None,
        }
    }
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncMode {
    VulkanStyle,
    Dag,
    Automatic,
}
#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize)]
pub enum SyncOperations {
    ComputeDispatch,
    Transfer,
    Both,
}
#[derive(Clone, Copy, Debug, Serialize, Deserialize)]
pub struct HalInstanceProperties {
    /// What synchronization requirements the backend has
    pub sync_mode: SyncMode,
    /// Whether the backend supports pipeline caches
    pub pipeline_cache: bool,
    /// What kernel langauge the backend takes
    pub kernel_lang: KernelTarget,
    /// Whether the backend supports bind groups that are updated while commands are already recorded. This makes bind groups far cheaper to use
    pub easily_update_bind_groups: bool,
    /// Whether the backend supports CPU->GPU communication using semaphore signalling.
    pub semaphore_signal: bool,
    /// Whether the system has unified memory, which provides opportunities for optimization, particularly on apple, mobile, or other devices with integrated GPUs
    pub is_unified_memory: bool,
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
pub enum KernelResourceType {
    #[default]
    Unknown,
    Buffer,
    UniformBuffer,
}
#[derive(Clone, Debug, Default, Serialize, Deserialize)]
pub struct KernelReflectionInfo {
    pub workgroup_size: [u32; 3],
    pub resources: Vec<KernelResourceType>,
    pub push_constant_len: u32,
}

pub type Dag<T> = daggy::Dag<T, ()>;

#[derive(Clone, Debug)]
pub enum BackendOptions {
    Wgpu { backends: wgpu::Backends },
    Vulkan,
}
pub struct InstanceDescriptor {
    pub backend_options: BackendOptions,
    pub max_host_memory: Option<u64>,
    pub max_device_memory: Option<u64>,
    pub force_embedded: Option<bool>,
    pub full_debug: bool,
}
