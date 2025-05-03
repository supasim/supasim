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
#[cfg(feature = "vulkan")]
pub mod vulkan;
#[cfg(feature = "wgpu")]
pub mod wgpu;

#[cfg(test)]
mod tests;

pub use dummy::Dummy;
#[cfg(feature = "vulkan")]
pub use vulkan::Vulkan;
#[cfg(feature = "wgpu")]
pub use wgpu::Wgpu;

use types::*;

/// Backend traits should not have their own destructors, as higher level operations may replace them with uninitialized memory by destructor time.
pub trait Backend: Sized + std::fmt::Debug + Clone {
    type Instance: BackendInstance<Self>;
    type Kernel: Kernel<Self>;
    type Buffer: Buffer<Self>;
    type CommandRecorder: CommandRecorder<Self>;
    type BindGroup: BindGroup<Self>;
    type KernelCache: KernelCache<Self>;
    type Semaphore: Semaphore<Self>;
    type Event: Event<Self>;

    type Error: Error<Self>;
}
pub trait BackendInstance<B: Backend<Instance = Self>> {
    fn get_properties(&mut self) -> InstanceProperties;
    /// # Safety
    /// * The shader code must be valid
    /// * The reflection info must match exactly with the shader
    /// * The cache must be valid
    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::ShaderReflectionInfo,
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
    /// * All submissions using any of these recorders must have completed
    unsafe fn clear_recorders(
        &mut self,
        buffers: &mut [&mut B::CommandRecorder],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * Indirect must only be set if the instance supports it
    unsafe fn create_buffer(
        &mut self,
        alloc_info: &BufferDescriptor,
    ) -> Result<B::Buffer, B::Error>;
    /// # Safety
    /// * All bind groups using this buffer must have been updated or destroyed
    unsafe fn destroy_buffer(&mut self, buffer: B::Buffer) -> Result<(), B::Error>;
    /// # Safety
    /// * All submitted command recorders using this buffer must have completed
    /// * The buffer must be of type `Upload`
    unsafe fn write_buffer(
        &mut self,
        buffer: &B::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * All submitted command recorders using this buffer mutably must have completed
    /// * The buffer must be of type `Download`
    unsafe fn read_buffer(
        &mut self,
        buffer: &B::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * The resources must correspond with the kernel and its layout
    unsafe fn create_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        resources: &[GpuResource<B>],
    ) -> Result<B::BindGroup, B::Error>;
    /// # Safety
    /// * If the backend doesn't support easily updatable bind groups, all command recorders using this bind group must've been cleared
    unsafe fn update_bind_group(
        &mut self,
        bg: &mut B::BindGroup,
        kernel: &mut B::Kernel,
        resources: &[GpuResource<B>],
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * All command recorders using this bind group must've been cleared
    unsafe fn destroy_bind_group(
        &mut self,
        kernel: &mut B::Kernel,
        bind_group: B::BindGroup,
    ) -> Result<(), B::Error>;

    /// # Safety
    /// * All command recorders using this event must've been cleared
    unsafe fn create_semaphore(&mut self) -> Result<B::Semaphore, B::Error>;
    /// # Safety
    /// * All command recorders using this semaphore must've been cleared
    unsafe fn destroy_semaphore(&mut self, semaphore: B::Semaphore) -> Result<(), B::Error>;

    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn create_event(&mut self) -> Result<B::Event, B::Error>;

    /// # Safety
    /// * All command recorders using this event must've completed
    unsafe fn destroy_event(&mut self, event: B::Event) -> Result<(), B::Error>;

    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), B::Error>;

    /// # Safety
    /// * All associated resources must be destroyed
    /// * All device work must be completed
    unsafe fn destroy(self) -> Result<(), B::Error>;
}
pub enum GpuResource<'a, B: Backend> {
    Buffer {
        buffer: &'a B::Buffer,
        offset: u64,
        size: u64,
    },
}
impl<'a, B: Backend> GpuResource<'a, B> {
    pub fn buffer(buffer: &'a B::Buffer, offset: u64, size: u64) -> Self {
        Self::Buffer {
            buffer,
            offset,
            size,
        }
    }
}
pub struct CommandSynchronization<'a, B: Backend> {
    pub resources_needing_sync: &'a mut [&'a mut GpuResource<'a, B>],
    pub out_semaphore: Option<(&'a mut B::Semaphore, u64)>,
}
pub trait CommandRecorder<B: Backend<CommandRecorder = Self>> {
    /// # Safety
    /// * Must only be called on instances with `SyncMode::Dag`
    /// * The recorder must not have had any record command since being created or cleared
    unsafe fn record_dag(
        &mut self,
        instance: &mut B::Instance,
        resources: &[&GpuResource<B>],
        dag: &mut Dag<BufferCommand<B>>,
    ) -> Result<(), B::Error>;
    /// # Safety
    /// * Must only be called on instances with `SyncMode` of `Automatic` or `VulkanStyle`
    /// * The recorder must not have had any record command since being created or cleared
    unsafe fn record_commands(
        &mut self,
        instance: &mut B::Instance,
        commands: &mut [BufferCommand<B>],
    ) -> Result<(), B::Error>;
}
pub trait Kernel<B: Backend<Kernel = Self>> {}
pub trait Buffer<B: Backend<Buffer = Self>> {}
pub trait BindGroup<B: Backend<BindGroup = Self>> {}
pub trait KernelCache<B: Backend<KernelCache = Self>> {}
pub trait Semaphore<B: Backend<Semaphore = Self>> {
    /// # Safety
    /// * The semaphore must be signalled by some already submitted command recorder
    unsafe fn wait(&mut self, instance: &mut B::Instance) -> Result<(), B::Error>;
    /// # Safety
    /// Currently no safety requirements. This is subject to change
    unsafe fn is_signalled(&mut self, instance: &mut B::Instance) -> Result<bool, B::Error>;
    /// # Safety
    /// * The semaphore must not be waited on by any CPU side wait command
    unsafe fn signal(&mut self, instance: &mut B::Instance) -> Result<(), B::Error>;
}
pub trait Event<B: Backend<Event = Self>> {}
pub struct RecorderSubmitInfo<'a, B: Backend> {
    pub command_recorder: &'a mut B::CommandRecorder,
    pub wait_semaphore: Option<&'a B::Semaphore>,
    pub signal_semaphore: Option<&'a B::Semaphore>,
}
#[must_use]
pub trait Error<B: Backend<Error = Self>>: std::error::Error {
    fn is_out_of_device_memory(&self) -> bool;
    fn is_out_of_host_memory(&self) -> bool;
    fn is_timeout(&self) -> bool;
}

pub enum BufferCommand<'a, B: Backend> {
    CopyBuffer {
        src_buffer: &'a B::Buffer,
        dst_buffer: &'a B::Buffer,
        src_offset: u64,
        dst_offset: u64,
        len: u64,
    },
    DispatchKernel {
        kernel: &'a B::Kernel,
        bind_group: &'a B::BindGroup,
        push_constants: &'a [u8],
        workgroup_dims: [u32; 3],
    },
    DispatchKernelIndirect {
        kernel: &'a B::Kernel,
        bind_group: &'a B::BindGroup,
        push_constants: &'a [u8],
        indirect_buffer: &'a B::Buffer,
        buffer_offset: u64,
        validate: bool,
    },
    /// Only for vulkan like synchronization
    SetEvent {
        event: &'a B::Event,
        wait: SyncOperations,
    },
    /// Only for vulkan like synchronization
    WaitEvent {
        event: &'a B::Event,
        signal: SyncOperations,
    },
    /// Only for vulkan like synchronization
    PipelineBarrier {
        before: SyncOperations,
        after: SyncOperations,
    },
    /// Only for vulkan like synchronization. Will hitch a ride with the previous PipelineBarrier or WaitEvent
    MemoryBarrier { resource: GpuResource<'a, B> },
    UpdateBindGroup {
        bg: &'a B::BindGroup,
        kernel: &'a B::Kernel,
        resources: &'a [GpuResource<'a, B>],
    },
}
