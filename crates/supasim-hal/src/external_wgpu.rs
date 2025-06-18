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
pub struct WgpuDeviceInfo {
    pub device: wgpu::Device,
    pub features: wgpu::Features,
    pub backend: wgpu::Backend,
}
impl WgpuDeviceInfo {
    pub fn supports_external_memory(&self) -> bool {
        unsafe {
            match self.backend {
                #[cfg(not(target_vendor = "apple"))]
                wgpu::Backend::Vulkan => {
                    self.device.as_hal::<wgpu::hal::vulkan::Api, _, _>(|dev| {
                        dev.unwrap()
                            .enabled_device_extensions()
                            .contains(&EXTERNAL_MEMORY_VULKAN_EXTENSION)
                    })
                }
                // TODO: add support for dx12, metal external memory
                wgpu::Backend::Dx12 | wgpu::Backend::Metal => false,
                _ => false,
            }
        }
    }
    /// # Safety
    /// * No current requirements are specified. Don't do anything stupid :)
    pub unsafe fn import_external_memory(
        &self,
        handle: crate::ExternalMemoryObject,
        alloc_info: types::HalBufferDescriptor,
    ) -> Option<wgpu::Buffer> {
        // There's lots of nesting here, deal with it
        unsafe {
            match self.backend {
                #[cfg(not(target_vendor = "apple"))]
                wgpu::Backend::Vulkan => {
                    self.device.as_hal::<wgpu::hal::vulkan::Api, _, _>(|dev| {
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

                        let mut dedicated_reqs = vk::MemoryDedicatedRequirementsKHR::default();
                        let mut requirements =
                            vk::MemoryRequirements2::default().push_next(&mut dedicated_reqs);
                        dev.get_buffer_memory_requirements2(
                            &vk::BufferMemoryRequirementsInfo2::default().buffer(buffer),
                            &mut requirements,
                        );

                        // TODO: investigate further whether dedicated allocation info is required. I know it is for
                        // D3D12 resource handles and for many DMA buffer things
                        let dedicated_alloc =
                            vk::MemoryDedicatedAllocateInfoKHR::default().buffer(buffer);
                        #[cfg(unix)]
                        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                            .fd(handle.handle as std::ffi::c_int)
                            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD_KHR);
                        #[cfg(windows)]
                        let mut import_info = vk::ImportMemoryWin32HandleInfoKHR::default()
                            .handle(handle.handle)
                            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KHR);
                        if dedicated_reqs.requires_dedicated_allocation == vk::TRUE {
                            import_info.p_next =
                                &dedicated_alloc as *const _ as *const std::ffi::c_void;
                        }
                        let allocate_info =
                            vk::MemoryAllocateInfo::default().push_next(&mut import_info);
                        let device_memory = dev.allocate_memory(&allocate_info, None).unwrap();
                        dev.bind_buffer_memory(buffer, device_memory, handle.offset)
                            .unwrap();
                        // Then convert this into a wgpu hal buffer
                        todo!()
                    })
                }
                _ => None,
            }
        }
    }
}
