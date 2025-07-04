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

#[derive(Clone, Copy, Debug, PartialEq, Eq, Serialize, Deserialize, Default)]
pub enum HalBufferType {
    /// Used for kernel access. Memory type on GPU that can be copied around on GPU but is optimized for local access.
    #[default]
    Storage,
    /// Best for uploading data to GPU. Memory type on GPU that can be written to from CPU and copied from on GPU.
    Upload,
    /// Best for downloading data from GPU. Memory type on GPU that can be copied to from GPU and read from CPU.
    Download,
    UploadDownload,
}
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct HalBufferDescriptor {
    pub size: u64,
    pub memory_type: HalBufferType,
    pub min_alignment: usize,
    pub can_export: bool,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Serialize, Deserialize, PartialOrd, Ord)]
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
    None,
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
    /// Whether the backend supports CPU->GPU synchronization using CPU-side semaphore signalling.
    pub semaphore_signal: bool,
    /// Whether the backend supports directly mapping host memory on the CPU instead of just reads/writes.
    /// This is likely false only on some wgpu backends such as WebGPU itself. Currently it is true everywhere.
    pub map_buffers: bool,
    /// Whether the system has unified memory, which provides opportunities for optimization, particularly on apple, mobile, or other devices with integrated GPUs
    pub is_unified_memory: bool,
    /// Whether you can map buffers while they are in use on the GPU(even if the slices don't overlap)
    pub map_buffer_while_gpu_use: bool,
    /// Whether it supports dual upload-download buffers
    pub upload_download_buffers: bool,
    /// Whether it supports exporting buffers
    pub export_memory: bool,
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
pub struct KernelReflectionInfo {
    pub workgroup_size: [u32; 3],
    /// bool is for whether it is writeable
    pub buffers: Vec<bool>,
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
