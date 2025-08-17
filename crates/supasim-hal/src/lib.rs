/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

pub mod dummy;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub mod metal;
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(test)]
mod tests;

pub use dummy::Dummy;
#[cfg(all(feature = "metal", target_vendor = "apple"))]
pub use metal::Metal;
#[cfg(feature = "vulkan")]
pub use vulkan::Vulkan;
#[cfg(feature = "wgpu")]
pub use wgpu::Wgpu;

#[cfg(feature = "wgpu")]
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
    type Semaphore: Semaphore<Self>;
    type Device: Device<Self>;
    type Stream: Stream<Self>;

    type Error: Error<Self>;
}

pub trait BackendInstance<B: Backend<Instance = Self>>: Send {
    fn get_properties(&self) -> HalInstanceProperties;
    /// # Safety
    /// * The kernel code must be valid
    /// * The reflection info must match exactly with the shader
    /// * The cache must be valid
    unsafe fn compile_kernel(&self, descriptor: KernelDescriptor) -> Result<B::Kernel, B::Error>;

    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn cleanup_cached_resources(&self) -> Result<(), B::Error>;

    /// # Safety
    /// * All associated resources must be destroyed
    /// * All device work must be completed
    /// * All devices and streams must be dropped
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
    /// * Must only be called on instances with `SyncMode` of `Automatic` or `VulkanStyle`
    /// * The recorder must not have had any record command since being created or cleared
    /// * All commands must follow the corresponding safety section in BufferCommand
    unsafe fn record_commands(
        &mut self,
        instance: &B::Instance,
        commands: &[BufferCommand<B>],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * The recorder must not be queued and incomplete on the GPU
    unsafe fn clear(&mut self, instance: &B::Instance) -> Result<(), B::Error>;
    /// # Safety
    /// * All submissions using this recorder must have completed
    unsafe fn destroy(self, instance: &B::Instance) -> Result<(), B::Error>;
}

pub trait Kernel<B: Backend<Kernel = Self>>: Send {
    /// # Safety
    /// * All command recorders using this kernel must've been cleared
    /// * All bind groups using this kernel must've been destroyed
    unsafe fn destroy(self, instance: &B::Instance) -> Result<(), B::Error>;
}

pub trait Buffer<B: Backend<Buffer = Self>>: Send {
    /// # Safety
    /// * All submitted command recorders using this buffer must have completed
    /// * The buffer must be of type `Upload`
    /// * No concurrent reads/writes are allowed, hence why it requires a mutable reference to buffer
    ///   * This is a limitation of wgpu and may be dealt with in the future
    /// * No mapped pointers may be used to access the same data ranges concurrently
    unsafe fn write(
        &self,
        instance: &B::Instance,
        offset: u64,
        data: &[u8],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * All submitted command recorders using this buffer mutably must have completed
    /// * The buffer must be of type `Download`
    /// * No concurrent reads/writes are allowed, hence why it requires a mutable reference to buffer
    ///   * This is a limitation of wgpu and may be dealt with in the future
    unsafe fn read(
        &self,
        buffer: &B::Instance,
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
    unsafe fn map(&mut self, instance: &B::Instance) -> Result<*mut u8, B::Error>;
    /// # Safety
    /// * All mapped pointers are invalidated
    unsafe fn unmap(&mut self, instance: &B::Instance) -> Result<(), B::Error>;
    /// # Safety
    /// * All bind groups using this buffer must have been updated or destroyed
    /// * The buffer must not be mapped
    unsafe fn destroy(self, instance: &B::Instance) -> Result<(), B::Error>;
}

pub trait BindGroup<B: Backend<BindGroup = Self>>: Send {
    /// # Safety
    /// * If the backend doesn't support easily updatable bind groups, all command recorders using this bind group must've been cleared
    unsafe fn update(
        &mut self,
        instance: &B::Instance,
        kernel: &B::Kernel,
        buffers: &[HalBufferSlice<B>],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * All command recorders using this bind group must've been cleared
    unsafe fn destroy(self, instance: &B::Instance, kernel: &mut B::Kernel)
    -> Result<(), B::Error>;
}

pub trait Semaphore<B: Backend<Semaphore = Self>>: Send {
    /// # Safety
    /// * The semaphore must be signalled by some already submitted command recorder
    unsafe fn wait(&self, instance: &B::Instance) -> Result<(), B::Error>;
    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn is_signalled(&self, instance: &B::Instance) -> Result<bool, B::Error>;
    /// # Safety
    /// * The semaphore must not be waited on by any CPU side wait command
    unsafe fn signal(&mut self, instance: &B::Instance) -> Result<(), B::Error>;
    /// Note that in most implementations this won't actually result in any underlying API changes.
    /// # Safety
    /// * The semaphore must not be waited on by any CPU or GPU side wait command
    /// * The semaphore must not be signalled by any CPU or GPU side signal command
    unsafe fn reset(&mut self, instance: &B::Instance) -> Result<(), B::Error>;
    /// # Safety
    /// * All command recorders using this semaphore must've been cleared
    unsafe fn destroy(self, instance: &B::Instance) -> Result<(), B::Error>;
}

pub trait Device<B: Backend<Device = Self>> {
    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn cleanup_cached_resources(&self) -> Result<(), B::Error>;
    /// # Safety
    /// * If the device doesn't support `upload_download_buffers`, `memory_type` cannot be `UploadDownload`
    unsafe fn create_buffer(
        &self,
        instance: &B::Instance,
        alloc_info: &HalBufferDescriptor,
    ) -> Result<B::Buffer, B::Error>;
    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn create_semaphore(&self) -> Result<B::Semaphore, B::Error>;
}
pub trait Stream<B: Backend<Stream = Self>> {
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
    /// * The resources must correspond with the kernel and its layout
    unsafe fn create_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        buffers: &[HalBufferSlice<B>],
    ) -> Result<B::BindGroup, B::Error>;
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

#[derive(Debug, Clone)]
pub struct KernelDescriptor<'a> {
    binary: &'a [u8],
    reflection: types::KernelReflectionInfo,
}
