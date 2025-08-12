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
pub mod dummy;
#[cfg(feature = "external_wgpu")]
pub mod external_wgpu;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub mod metal;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(test)]
mod tests;

use std::any::Any;

pub use dummy::Dummy;
#[cfg(feature = "external_wgpu")]
pub use external_wgpu::{WgpuDeviceExportInfo, wgpu_adapter_supports_external};
#[cfg(feature = "wgpu")]
pub use metal::Metal;
#[cfg(feature = "vulkan")]
pub use vulkan::Vulkan;
#[cfg(feature = "wgpu")]
pub use wgpu::Wgpu;

#[cfg(any(feature = "external_wgpu", feature = "wgpu"))]
pub use ::wgpu as wgpu_dep;

use types::*;

/// Backend traits should not have their own destructors, as higher level operations may replace them with uninitialized memory by destructor time.
/// # Safety (general)
/// * All types are assumed to be safely send/sync
pub trait Backend: Sized + std::fmt::Debug + Clone + Send + Sync + 'static {
    type Instance: BackendInstance<Self>;
    type Kernel: Kernel<Self>;
    type Buffer: Buffer<Self>;
    type CommandRecorder: CommandRecorder<Self>;
    type BindGroup: BindGroup<Self>;
    type KernelCache: KernelCache<Self>;
    type Semaphore: Semaphore<Self>;

    type Error: Error<Self>;
}

pub trait BackendInstance<B: Backend<Instance = Self>>: Send {
    fn get_properties(&mut self) -> HalInstanceProperties;
    /// Get whether or not memory can be shared to a certain device. Usually, this device would be a wgpu device. Note that using this
    /// with wgpu devices will require the wgpu feature, even if the backend isn't used.
    ///
    /// # Safety
    /// * Unknown safety requirements lol
    unsafe fn can_share_memory_to_device(&mut self, device: &dyn Any) -> Result<bool, B::Error>;
    /// # Safety
    /// * The kernel code must be valid
    /// * The reflection info must match exactly with the shader
    /// * The cache must be valid
    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::KernelReflectionInfo,
        cache: Option<&mut B::KernelCache>,
    ) -> Result<B::Kernel, B::Error>;
    /// # Safety
    /// * Initial data must be either empty or valid
    ///   * The data must be from the same backend
    unsafe fn create_kernel_cache(
        &mut self,
        initial_data: &[u8],
    ) -> Result<B::KernelCache, B::Error>;
    /// # Safety
    /// * All kernels in the cache must've been destroyed
    unsafe fn destroy_kernel_cache(&mut self, cache: B::KernelCache) -> Result<(), B::Error>;
    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn get_kernel_cache_data(
        &mut self,
        cache: &mut B::KernelCache,
    ) -> Result<Vec<u8>, B::Error>;
    /// # Safety
    /// * All command recorders using this kernel must've been cleared
    /// * All bind groups using this kernel must've been destroyed
    unsafe fn destroy_kernel(&mut self, kernel: B::Kernel) -> Result<(), B::Error>;
    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn wait_for_idle(&mut self) -> Result<(), B::Error>;
    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn create_recorder(&mut self) -> Result<B::CommandRecorder, B::Error>;
    /// # Safety
    /// * For each recorder submitted, it must have had __exactly__ one record command called on it since being cleared or created
    /// * For each wait semaphore, the semaphore must be signalled at some point on the CPU side only
    /// * For each signal semaphore, the semaphore must be used only by CPU side wait commands
    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [RecorderSubmitInfo<B>],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * All submissions using this recorder must have completed
    unsafe fn destroy_recorder(&mut self, recorder: B::CommandRecorder) -> Result<(), B::Error>;
    /// # Safety
    /// * If the device doesn't support `upload_download_buffers`, `memory_type` cannot be `UploadDownload`
    unsafe fn create_buffer(
        &mut self,
        alloc_info: &HalBufferDescriptor,
    ) -> Result<B::Buffer, B::Error>;
    /// # Safety
    /// * All bind groups using this buffer must have been updated or destroyed
    /// * The buffer must not be mapped
    unsafe fn destroy_buffer(&mut self, buffer: B::Buffer) -> Result<(), B::Error>;
    /// # Safety
    /// * All submitted command recorders using this buffer must have completed
    /// * The buffer must be of type `Upload`
    /// * No concurrent reads/writes are allowed, hence why it requires a mutable reference to buffer
    ///   * This is a limitation of wgpu and may be dealt with in the future
    /// * No mapped pointers may be used to access the same data ranges concurrently
    unsafe fn write_buffer(
        &mut self,
        buffer: &mut B::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * All submitted command recorders using this buffer mutably must have completed
    /// * The buffer must be of type `Download`
    /// * No concurrent reads/writes are allowed, hence why it requires a mutable reference to buffer
    ///   * This is a limitation of wgpu and may be dealt with in the future
    unsafe fn read_buffer(
        &mut self,
        buffer: &mut B::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * The device must support `map_buffers`
    /// * Map buffer may be called multiple times to obtain multiple pointers
    /// * Unmap buffer invalidates all such pointers
    /// * Writing to a mapped upload buffer is illegal(except on unified memory systems)
    /// * If the device doesn't support `map_buffer_while_gpu_use`, the buffer must be unmapped before
    ///   any GPU work using the buffer is submitted
    unsafe fn map_buffer(&mut self, buffer: &mut B::Buffer) -> Result<*mut u8, B::Error>;
    /// # Safety
    /// * All mapped pointers are invalidated
    unsafe fn unmap_buffer(&mut self, buffer: &mut B::Buffer) -> Result<(), B::Error>;
    /// # Safety
    /// * The resources must correspond with the kernel and its layout
    unsafe fn create_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        buffers: &[HalBufferSlice<B>],
    ) -> Result<B::BindGroup, B::Error>;
    /// # Safety
    /// * If the backend doesn't support easily updatable bind groups, all command recorders using this bind group must've been cleared
    unsafe fn update_bind_group(
        &mut self,
        bg: &mut B::BindGroup,
        kernel: &mut B::Kernel,
        buffers: &[HalBufferSlice<B>],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * All command recorders using this bind group must've been cleared
    unsafe fn destroy_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        bind_group: B::BindGroup,
    ) -> Result<(), B::Error>;

    /// # Safety
    /// * This function is marked unsafe for semver reasons
    unsafe fn create_semaphore(&mut self) -> Result<B::Semaphore, B::Error>;
    /// # Safety
    /// * All command recorders using this semaphore must've been cleared
    unsafe fn destroy_semaphore(&mut self, semaphore: B::Semaphore) -> Result<(), B::Error>;

    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), B::Error>;

    /// # Safety
    /// * All associated resources must be destroyed
    /// * All device work must be completed
    unsafe fn destroy(self) -> Result<(), B::Error>;
}

/// # Safety
/// * Wherever this is accepted in the API, it is assumed to be a valid slice within the buffer
///   unless explicitly stated otherwise
#[derive(Debug)]
pub struct HalBufferSlice<'a, B: Backend> {
    pub buffer: &'a B::Buffer,
    pub offset: u64,
    pub len: u64,
}

pub struct CommandSynchronization<'a, B: Backend> {
    pub buffers_needing_sync: &'a mut [&'a mut HalBufferSlice<'a, B>],
    pub out_semaphore: Option<(&'a mut B::Semaphore, u64)>,
}

pub trait CommandRecorder<B: Backend<CommandRecorder = Self>>: Send {
    /// # Safety
    /// * Must only be called on instances with `SyncMode::Dag`
    /// * The recorder must not have had any record command since being created or cleared
    /// * All commands must follow the corresponding safety section in BufferCommand
    unsafe fn record_dag(
        &mut self,
        instance: &mut B::Instance,
        dag: &mut Dag<BufferCommand<B>>,
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * Must only be called on instances with `SyncMode` of `Automatic` or `VulkanStyle`
    /// * The recorder must not have had any record command since being created or cleared
    /// * All commands must follow the corresponding safety section in BufferCommand
    unsafe fn record_commands(
        &mut self,
        instance: &mut B::Instance,
        commands: &mut [BufferCommand<B>],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * The recorder must not be queued and incomplete on the GPU
    unsafe fn clear(&mut self, instance: &mut B::Instance) -> Result<(), B::Error>;
}

pub trait Kernel<B: Backend<Kernel = Self>>: Send {}

pub trait Buffer<B: Backend<Buffer = Self>>: Send {
    /// # Safety
    /// * Synchronization must be managed by the user
    /// * The buffer must be of type `Storage`
    unsafe fn export(
        &mut self,
        instance: &mut B::Instance,
    ) -> Result<ExternalMemoryObject, B::Error>;
    /// # Safety
    /// * The instance must have had can_share_memory_to_device called on the same external device, and it must have returned true
    /// * Synchronization must be managed by the user
    /// * The buffer must be of type `Storage`
    unsafe fn share_to_device(
        &mut self,
        instance: &mut B::Instance,
        external_device: &dyn Any,
    ) -> Result<Box<dyn Any>, B::Error>;
}

pub trait BindGroup<B: Backend<BindGroup = Self>>: Send {}

pub trait KernelCache<B: Backend<KernelCache = Self>>: Send {}

pub trait Semaphore<B: Backend<Semaphore = Self>>: Send {
    /// # Safety
    /// * The semaphore must be signalled by some already submitted command recorder
    unsafe fn wait(&self) -> Result<(), B::Error>;
    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn is_signalled(&self) -> Result<bool, B::Error>;
    /// # Safety
    /// * The semaphore must not be waited on by any CPU side wait command
    unsafe fn signal(&self) -> Result<(), B::Error>;
    /// Note that in most implementations this won't actually result in any underlying API changes.
    /// # Safety
    /// * The semaphore must not be waited on by any CPU or GPU side wait command
    /// * The semaphore must not be signalled by any CPU or GPU side signal command
    unsafe fn reset(&self) -> Result<(), B::Error>;
}

#[derive(Debug)]
pub struct RecorderSubmitInfo<'a, B: Backend> {
    pub command_recorder: &'a mut B::CommandRecorder,
    pub wait_semaphore: Option<&'a B::Semaphore>,
    pub signal_semaphore: Option<&'a B::Semaphore>,
}

#[must_use]
pub trait Error<B: Backend<Error = Self>>: std::error::Error + Send {
    fn is_out_of_device_memory(&self) -> bool;
    fn is_out_of_host_memory(&self) -> bool;
    fn is_timeout(&self) -> bool;
}

#[derive(Debug)]
pub enum BufferCommand<'a, B: Backend> {
    /// # Safety
    /// * The length and offsets must be form valid slices in the buffers
    CopyBuffer {
        src_buffer: &'a B::Buffer,
        dst_buffer: &'a B::Buffer,
        src_offset: u64,
        dst_offset: u64,
        len: u64,
    },
    /// # Safety
    /// * The bindgroup must be created for the kernel
    DispatchKernel {
        kernel: &'a B::Kernel,
        bind_group: &'a B::BindGroup,
        push_constants: &'a [u8],
        workgroup_dims: [u32; 3],
    },
    /// Only for vulkan like synchronization
    /// # Safety
    /// * this is always safe
    PipelineBarrier {
        before: SyncOperations,
        after: SyncOperations,
    },
    /// Only for vulkan like synchronization. Consecutive pipeline and memory barriers will combine. Memory barriers without such a pipeline barrier are undefined behavior.
    /// # Safety
    /// * This is always safe
    MemoryBarrier { buffer: HalBufferSlice<'a, B> },
    /// # Safety
    /// * The instance must support the form of import/export used
    MemoryTransfer {
        buffer: HalBufferSlice<'a, B>,
        import: bool,
    },
    /// # Safety
    /// * Only valid on instances with the `easily_update_bind_groups` property.
    UpdateBindGroup {
        bg: &'a B::BindGroup,
        kernel: &'a B::Kernel,
        buffers: &'a [HalBufferSlice<'a, B>],
    },
    /// # Safety
    /// * Offset and length must be multiples of 4
    ZeroMemory { buffer: HalBufferSlice<'a, B> },
    /// # Safety
    /// * This is always safe
    Dummy,
}

#[derive(Debug)]
pub struct ExternalMemoryObject {
    pub handle: isize,
    pub offset: u64,
}
