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
#![allow(unused_variables)]

use crate::*;

/// # Overview
/// Testing backend to allow certain tests to run even without GPU support on a system
#[derive(Clone, Debug)]
pub struct Dummy;
impl Backend for Dummy {
    type Instance = DummyResource;
    type Kernel = DummyResource;
    type Buffer = DummyResource;
    type CommandRecorder = DummyResource;
    type KernelCache = DummyResource;
    type BindGroup = DummyResource;
    type Semaphore = DummyResource;
    type Error = DummyResource;
}
impl Dummy {
    pub fn create_instance() -> Result<DummyResource, DummyResource> {
        Ok(DummyResource)
    }
}
#[derive(Clone, Debug)]
pub struct DummyResource;
impl BackendInstance<Dummy> for DummyResource {
    fn get_properties(&mut self) -> types::HalInstanceProperties {
        HalInstanceProperties {
            sync_mode: SyncMode::Automatic,
            pipeline_cache: true,
            kernel_lang: KernelTarget::Spirv {
                version: SpirvVersion::V1_0,
            },
            easily_update_bind_groups: true,
            semaphore_signal: true,
            map_buffers: true,
            is_unified_memory: false,
            map_buffer_while_gpu_use: true,
            upload_download_buffers: true,
        }
    }

    unsafe fn compile_kernel(
        &mut self,
        binary: &[u8],
        reflection: &types::KernelReflectionInfo,
        cache: Option<&mut <Dummy as Backend>::KernelCache>,
    ) -> Result<<Dummy as Backend>::Kernel, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn create_kernel_cache(
        &mut self,
        initial_data: &[u8],
    ) -> Result<<Dummy as Backend>::KernelCache, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn destroy_kernel_cache(
        &mut self,
        cache: <Dummy as Backend>::KernelCache,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn get_kernel_cache_data(
        &mut self,
        cache: &mut <Dummy as Backend>::KernelCache,
    ) -> Result<Vec<u8>, <Dummy as Backend>::Error> {
        Ok(Vec::new())
    }

    unsafe fn destroy_kernel(
        &mut self,
        kernel: <Dummy as Backend>::Kernel,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn wait_for_idle(&mut self) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn create_recorder(
        &mut self,
    ) -> Result<<Dummy as Backend>::CommandRecorder, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn submit_recorders(
        &mut self,
        infos: &mut [crate::RecorderSubmitInfo<Dummy>],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn destroy_recorder(
        &mut self,
        recorder: <Dummy as Backend>::CommandRecorder,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn create_buffer(
        &mut self,
        alloc_info: &types::HalBufferDescriptor,
    ) -> Result<<Dummy as Backend>::Buffer, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn destroy_buffer(
        &mut self,
        buffer: <Dummy as Backend>::Buffer,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn write_buffer(
        &mut self,
        buffer: &mut <Dummy as Backend>::Buffer,
        offset: u64,
        data: &[u8],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn read_buffer(
        &mut self,
        buffer: &mut <Dummy as Backend>::Buffer,
        offset: u64,
        data: &mut [u8],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn map_buffer(
        &mut self,
        buffer: &mut <Dummy as Backend>::Buffer,
    ) -> Result<*mut u8, <Dummy as Backend>::Error> {
        Ok(std::ptr::null_mut())
    }

    unsafe fn unmap_buffer(
        &mut self,
        buffer: &mut <Dummy as Backend>::Buffer,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn create_bind_group(
        &mut self,
        kernel: &mut <Dummy as Backend>::Kernel,
        buffers: &[crate::HalBufferSlice<Dummy>],
    ) -> Result<<Dummy as Backend>::BindGroup, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn update_bind_group(
        &mut self,
        bg: &mut <Dummy as Backend>::BindGroup,
        kernel: &mut <Dummy as Backend>::Kernel,
        buffers: &[crate::HalBufferSlice<Dummy>],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn destroy_bind_group(
        &mut self,
        kernel: &mut <Dummy as Backend>::Kernel,
        bind_group: <Dummy as Backend>::BindGroup,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn create_semaphore(
        &mut self,
    ) -> Result<<Dummy as Backend>::Semaphore, <Dummy as Backend>::Error> {
        Ok(DummyResource)
    }

    unsafe fn destroy_semaphore(
        &mut self,
        semaphore: <Dummy as Backend>::Semaphore,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }

    unsafe fn cleanup_cached_resources(&mut self) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
    unsafe fn destroy(self) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
}
impl Kernel<Dummy> for DummyResource {}
impl Buffer<Dummy> for DummyResource {}
impl CommandRecorder<Dummy> for DummyResource {
    unsafe fn record_commands(
        &mut self,
        instance: &mut <Dummy as Backend>::Instance,
        commands: &mut [crate::BufferCommand<Dummy>],
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
    unsafe fn record_dag(
        &mut self,
        instance: &mut <Dummy as Backend>::Instance,
        dag: &mut types::Dag<crate::BufferCommand<Dummy>>,
    ) -> Result<(), <Dummy as Backend>::Error> {
        unreachable!()
    }
    unsafe fn clear(
        &mut self,
        instance: &mut <Dummy as Backend>::Instance,
    ) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
}
impl KernelCache<Dummy> for DummyResource {}
impl BindGroup<Dummy> for DummyResource {}
impl Semaphore<Dummy> for DummyResource {
    unsafe fn wait(&mut self) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
    unsafe fn is_signalled(&mut self) -> Result<bool, <Dummy as Backend>::Error> {
        Ok(false)
    }
    unsafe fn signal(&mut self) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
    unsafe fn reset(&mut self) -> Result<(), <Dummy as Backend>::Error> {
        Ok(())
    }
}
impl Error<Dummy> for DummyResource {
    // Error will never be constructed
    fn is_out_of_device_memory(&self) -> bool {
        unreachable!()
    }
    fn is_out_of_host_memory(&self) -> bool {
        unreachable!()
    }
    fn is_timeout(&self) -> bool {
        unreachable!()
    }
}
impl std::fmt::Display for DummyResource {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        unreachable!()
    }
}
impl std::error::Error for DummyResource {}
