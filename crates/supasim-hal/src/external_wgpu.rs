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
//! External memory overview
//! * If the device and backend are the same, life is simple
//! * Otherwise, due to wgpu limitations, we must manually check the hal device
//!   for feature support. On Linux this means `external_memory_fd` support, and
//!   on windows it means `external_memory_win32` support. This is because wgpu
//!   automatically enables these extensions whether requested or not.
//!   * On apple, this is more complex. I haven't worked that out yet.
//! * Currently, only Vulkan support is being used.

use types::HalBufferType;
#[cfg(not(target_vendor = "apple"))]
const EXTERNAL_MEMORY_VULKAN_EXTENSION: &std::ffi::CStr = const {
    if cfg!(windows) {
        c"VK_KHR_external_memory_win32"
    } else if cfg!(unix) {
        c"VK_KHR_external_memory_fd"
    } else {
        c"ALWAYS_UNSUPPORTED"
    }
};

#[derive(Clone, Debug)]
pub struct WgpuDeviceExportInfo {
    pub device: wgpu::Device,
    pub features: wgpu::Features,
    pub backend: wgpu::Backend,
    pub usages: wgpu::BufferUsages,
}
impl WgpuDeviceExportInfo {
    pub fn supports_external_memory(&self) -> bool {
        match self.backend {
            #[cfg(not(target_vendor = "apple"))]
            wgpu::Backend::Vulkan => unsafe {
                self.device.as_hal::<wgpu::hal::vulkan::Api, _, _>(|dev| {
                    dev.unwrap()
                        .enabled_device_extensions()
                        .contains(&EXTERNAL_MEMORY_VULKAN_EXTENSION)
                })
            },
            // TODO: add support for dx12, metal external memory
            wgpu::Backend::Dx12 | wgpu::Backend::Metal => false,
            _ => false,
        }
    }
    /// # Safety
    /// * No current requirements are specified. Don't do anything stupid :)
    #[allow(unused_variables)]
    pub unsafe fn import_external_memory(
        &self,
        handle: crate::ExternalMemoryObject,
        alloc_info: types::HalBufferDescriptor,
    ) -> Option<wgpu::Buffer> {
        // Wgpu has weird requirements for buffer imports, and host memory seems useless to copy
        assert!(alloc_info.memory_type == HalBufferType::Storage);
        // There's lots of nesting here, deal with it
        match self.backend {
            #[cfg(not(target_vendor = "apple"))]
            wgpu::Backend::Vulkan => {
                /*self.device.as_hal::<wgpu::hal::vulkan::Api, _, _>(|dev| {
                    use ash::vk;
                    let hal_dev = dev.unwrap();
                    let dev = hal_dev.raw_device();

                    let buffer = dev
                        .create_buffer(
                            &vk::BufferCreateInfo::default()
                                .size(alloc_info.size)
                                .sharing_mode(vk::SharingMode::EXCLUSIVE)
                                .queue_family_indices(std::slice::from_ref(
                                    &hal_dev.queue_family_index(),
                                ))
                                .usage({
                                    use vk::BufferUsageFlags as F;
                                    match alloc_info.memory_type {
                                        HalBufferType::Storage => {
                                            F::TRANSFER_SRC
                                                | F::TRANSFER_DST
                                                | F::STORAGE_BUFFER
                                        }
                                        HalBufferType::Upload => F::TRANSFER_SRC,
                                        HalBufferType::Download => F::TRANSFER_DST,
                                        HalBufferType::UploadDownload => {
                                            F::TRANSFER_SRC | F::TRANSFER_DST
                                        }
                                    }
                                }),
                            None,
                        )
                        .unwrap();
                    // TODO: currently this doesn't check for dedicated allocation requirements.
                    // That's because I can't guarantee wgpu has enabled the right extensions
                    #[cfg(unix)]
                    let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                        .fd(handle.handle as std::ffi::c_int)
                        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD_KHR);
                    #[cfg(windows)]
                    let mut import_info = vk::ImportMemoryWin32HandleInfoKHR::default()
                        .handle(handle.handle)
                        .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KHR);
                    let allocate_info =
                        vk::MemoryAllocateInfo::default().push_next(&mut import_info);
                    let device_memory = dev.allocate_memory(&allocate_info, None).unwrap();
                    dev.bind_buffer_memory(buffer, device_memory, handle.offset)
                        .unwrap();
                    // TODO: revise this, as wgpu requires we keep track of the buffer memory ourselves
                    // Currently, we simply forget about the device memory backing the buffer.
                    let hal_buffer = wgpu::hal::vulkan::Device::buffer_from_raw(buffer);
                    let buffer = self
                        .device
                        .create_buffer_from_hal::<wgpu::hal::vulkan::Api>(
                            hal_buffer,
                            &wgpu::BufferDescriptor {
                                label: None,
                                size: alloc_info.size,
                                usage: wgpu::BufferUsages::STORAGE
                                    | wgpu::BufferUsages::COPY_SRC
                                    | wgpu::BufferUsages::COPY_DST,
                                mapped_at_creation: false,
                            },
                        );
                    Some(buffer)
                })*/
                if self
                    .features
                    .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_FD)
                {
                    Some(unsafe {
                        self.device.create_buffer_external_memory_fd(
                            handle.handle as i32,
                            handle.offset,
                            &wgpu::BufferDescriptor {
                                label: None,
                                size: alloc_info.size,
                                usage: self.usages,
                                mapped_at_creation: false,
                            },
                        )
                    })
                } else {
                    None
                }
            }
            _ => None,
        }
    }
}
