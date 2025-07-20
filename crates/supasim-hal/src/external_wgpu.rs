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

pub fn wgpu_adapter_supports_external(_adapter: wgpu::Adapter, backend: wgpu::Backend) -> bool {
    match backend {
        #[cfg(all(not(target_vendor = "apple"), feature = "vulkan"))]
        wgpu::Backend::Vulkan => unsafe {
            let a = _adapter.as_hal::<wgpu::hal::vulkan::Api>().unwrap();
            let exts = a.required_device_extensions(wgpu::Features::empty());
            exts.contains(&EXTERNAL_MEMORY_VULKAN_EXTENSION)
        },
        // TODO: add support for dx12, metal external memory
        _ => false,
    }
}
#[derive(Clone, Debug)]
pub struct WgpuDeviceExportInfo {
    pub device: wgpu::Device,
    pub features: wgpu::Features,
    pub backend: wgpu::Backend,
    pub usages: wgpu::BufferUsages,
    pub adapter: wgpu::Adapter,
}
impl WgpuDeviceExportInfo {
    pub fn supports_external_memory(&self) -> bool {
        wgpu_adapter_supports_external(self.adapter.clone(), self.backend)
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
                /*if self
                    .features
                    .contains(wgpu::Features::VULKAN_EXTERNAL_MEMORY_FD)
                {
                    let buffer_usage = wgpu::hal::vulkan::conv::map_buffer_usage(desc.usage);

                    let handle_type = if is_win32 {
                        vk::ExternalMemoryHandleTypeFlagsKHR::OPAQUE_WIN32_KHR
                    } else {
                        vk::ExternalMemoryHandleTypeFlagsKHR::OPAQUE_FD_KHR
                    };

                    let mut external_properties = vk::ExternalBufferProperties::default();
                    unsafe {
                        let caps = self
                            .shared_instance()
                            .external_memory_capabilities
                            .as_ref()
                            .unwrap();
                        let info = vk::PhysicalDeviceExternalBufferInfoKHR::default()
                            .usage(buffer_usage)
                            .handle_type(handle_type);
                        (caps.fp().get_physical_device_external_buffer_properties_khr)(
                            self.raw_physical_device(),
                            &info,
                            &mut external_properties,
                        );
                    }
                    if !external_properties
                        .external_memory_properties
                        .external_memory_features
                        .contains(vk::ExternalMemoryFeatureFlags::IMPORTABLE_KHR)
                    {
                        return Err(crate::DeviceError::Unexpected);
                    }
                    let needs_dedicated = external_properties
                        .external_memory_properties
                        .external_memory_features
                        .contains(vk::ExternalMemoryFeatureFlags::DEDICATED_ONLY_KHR);

                    let mut external_vk_info =
                        vk::ExternalMemoryBufferCreateInfoKHR::default().handle_types(handle_type);
                    let vk_info = vk::BufferCreateInfo::default()
                        .size(desc.size)
                        .usage(buffer_usage)
                        .sharing_mode(vk::SharingMode::EXCLUSIVE)
                        .push_next(&mut external_vk_info);

                    let raw = unsafe {
                        self.shared
                            .raw
                            .create_buffer(&vk_info, None)
                            .map_err(super::map_host_device_oom_and_ioca_err)?
                    };

                    let mut alloc_usage = if desc
                        .usage
                        .intersects(wgt::BufferUses::MAP_READ | wgt::BufferUses::MAP_WRITE)
                    {
                        let mut flags = gpu_alloc::UsageFlags::HOST_ACCESS;
                        //TODO: find a way to use `crate::MemoryFlags::PREFER_COHERENT`
                        flags.set(
                            gpu_alloc::UsageFlags::DOWNLOAD,
                            desc.usage.contains(wgt::BufferUses::MAP_READ),
                        );
                        flags.set(
                            gpu_alloc::UsageFlags::UPLOAD,
                            desc.usage.contains(wgt::BufferUses::MAP_WRITE),
                        );
                        flags
                    } else {
                        gpu_alloc::UsageFlags::FAST_DEVICE_ACCESS
                    };
                    alloc_usage.set(
                        gpu_alloc::UsageFlags::TRANSIENT,
                        desc.memory_flags.contains(crate::MemoryFlags::TRANSIENT),
                    );
                    let reqs = unsafe { self.raw_device().get_buffer_memory_requirements(raw) };

                    let mut dedicated_alloc_info =
                        vk::MemoryDedicatedAllocateInfoKHR::default().buffer(raw);
                    let memory = if is_win32 {
                        let mut import_info = vk::ImportMemoryWin32HandleInfoKHR::default()
                            .handle(handle as isize)
                            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_WIN32_KHR);
                        let mut allocate_info = vk::MemoryAllocateInfo::default()
                            .allocation_size(reqs.size)
                            .memory_type_index(reqs.memory_type_bits.trailing_zeros())
                            .push_next(&mut import_info);
                        if needs_dedicated {
                            allocate_info = allocate_info.push_next(&mut dedicated_alloc_info);
                        }
                        unsafe { self.raw_device().allocate_memory(&allocate_info, None) }
                    } else {
                        let mut import_info = vk::ImportMemoryFdInfoKHR::default()
                            .fd(handle as i32)
                            .handle_type(vk::ExternalMemoryHandleTypeFlags::OPAQUE_FD_KHR);
                        let mut allocate_info = vk::MemoryAllocateInfo::default()
                            .allocation_size(reqs.size)
                            .memory_type_index(reqs.memory_type_bits.trailing_zeros())
                            .push_next(&mut import_info);
                        if needs_dedicated {
                            allocate_info = allocate_info.push_next(&mut dedicated_alloc_info);
                        }
                        unsafe { self.raw_device().allocate_memory(&allocate_info, None) }
                    }
                    .map_err(|_| crate::DeviceError::Unexpected)
                    .inspect_err(|_| unsafe { self.raw_device().destroy_buffer(raw, None) })?;

                    unsafe { self.shared.raw.bind_buffer_memory(raw, memory, offset) }
                        .map_err(super::map_host_device_oom_and_ioca_err)
                        .inspect_err(|_| {
                            unsafe { self.shared.raw.destroy_buffer(raw, None) };
                        })?;

                    if let Some(label) = desc.label {
                        unsafe { self.shared.set_object_name(raw, label) };
                    }

                    self.counters.buffer_memory.add(desc.size as isize);
                    self.counters.buffers.add(1);

                    Ok(super::Buffer {
                        raw,
                        block: Some(Mutex::new(super::BufferMemoryBacking::VulkanMemory {
                            memory,
                            offset,
                            size: reqs.size,
                        })),
                    })
                } else {
                    None
                }*/
                None
            }
            _ => None,
        }
    }
}
