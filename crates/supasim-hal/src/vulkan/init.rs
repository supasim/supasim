use crate::vulkan::{
    DeviceFunctions, SharedDeviceInfo, Vulkan, VulkanDevice, VulkanError, VulkanInstance,
    VulkanStream,
};
use crate::{DeviceDescriptor, InstanceDescriptor, StreamDescriptor};
use ash::{
    Entry, khr,
    vk::{self},
};
use core::ffi;
use log::{Level, warn};
use std::sync::Arc;
use std::{borrow::Cow, cell::Cell, ffi::CStr, sync::Mutex};
use vk_mem::{Allocator, AllocatorCreateInfo};

use scopeguard::defer;

impl Vulkan {
    unsafe extern "system" fn vulkan_debug_callback(
        message_severity: vk::DebugUtilsMessageSeverityFlagsEXT,
        message_type: vk::DebugUtilsMessageTypeFlagsEXT,
        p_callback_data: *const vk::DebugUtilsMessengerCallbackDataEXT<'_>,
        _user_data: *mut std::os::raw::c_void,
    ) -> vk::Bool32 {
        let callback_data = unsafe { *p_callback_data };
        let message_id_number = callback_data.message_id_number;

        let message_id_name = if callback_data.p_message_id_name.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message_id_name).to_string_lossy() }
        };

        let message = if callback_data.p_message.is_null() {
            Cow::from("")
        } else {
            unsafe { ffi::CStr::from_ptr(callback_data.p_message).to_string_lossy() }
        };
        let level = match message_severity {
            vk::DebugUtilsMessageSeverityFlagsEXT::ERROR => Level::Error,
            vk::DebugUtilsMessageSeverityFlagsEXT::WARNING => Level::Warn,
            vk::DebugUtilsMessageSeverityFlagsEXT::INFO => Level::Info,
            vk::DebugUtilsMessageSeverityFlagsEXT::VERBOSE => Level::Trace,
            _ => Level::Error,
        };

        log::log!(
            level,
            "{message_severity:?}: {message_type:?} [{message_id_name} ({message_id_number})] : {message}\n",
        );
        #[cfg(test)]
        if message_severity == vk::DebugUtilsMessageSeverityFlagsEXT::ERROR {
            panic!(
                "Vulkan validation error in test. Users of SupaSim should report this as a bug. Error message:\n\t{message_type:?} [{message_id_name} ({message_id_number})] : {message}",
            )
        }

        vk::FALSE
    }

    pub fn create_instance(debug: bool) -> Result<InstanceDescriptor<Vulkan>, VulkanError> {
        unsafe {
            let err = Cell::new(true);
            let entry = Entry::load()?;
            let debug = if debug
                && entry
                    .enumerate_instance_extension_properties(None)
                    .unwrap()
                    .iter()
                    .any(|e| e.extension_name_as_c_str().unwrap() == c"VK_EXT_debug_utils")
                && entry
                    .enumerate_instance_layer_properties()
                    .unwrap()
                    .iter()
                    .any(|l| l.layer_name_as_c_str().unwrap() == c"VK_LAYER_KHRONOS_validation")
            {
                true
            } else if debug {
                warn!("Debug support was requested but is not available on the current system!");
                false
            } else {
                false
            };
            let validation_layers = if debug {
                vec![c"VK_LAYER_KHRONOS_validation".as_ptr()]
            } else {
                Vec::new()
            };
            let extension_names = if debug {
                vec![ash::ext::debug_utils::NAME.as_ptr()]
            } else {
                Vec::new()
            };
            // Check instance extensions
            {
                for &ext in &extension_names {
                    let ext_name = CStr::from_ptr(ext);
                    if !entry
                        .enumerate_instance_extension_properties(None)
                        .unwrap()
                        .iter()
                        .map(|a| a.extension_name_as_c_str().unwrap())
                        .any(|a| a == ext_name)
                    {
                        return Err(VulkanError::VulkanInstanceExtensionNotSupported(ext_name));
                    }
                }
            }
            // Check for validation layers
            {
                for &layer in &validation_layers {
                    let layer_name = CStr::from_ptr(layer);
                    if !entry
                        .enumerate_instance_layer_properties()
                        .unwrap()
                        .iter()
                        .map(|a| a.layer_name_as_c_str().unwrap())
                        .any(|a| a == layer_name)
                    {
                        return Err(VulkanError::VulkanLayerNotSupported(layer_name));
                    }
                }
            }
            // Check vulkan version
            let instance_api_version = match entry.try_enumerate_instance_version().unwrap() {
                Some(v) => v,
                None => vk::API_VERSION_1_0,
            };
            let app_info = vk::ApplicationInfo::default().api_version(instance_api_version);

            let instance = entry.create_instance(
                &vk::InstanceCreateInfo::default()
                    .application_info(&app_info)
                    .enabled_layer_names(&validation_layers)
                    .enabled_extension_names(&extension_names),
                None,
            )?;
            defer! {
                if err.get() {
                    instance.destroy_instance(None);
                }
            }
            let debug_callback = if debug {
                let debug_info = vk::DebugUtilsMessengerCreateInfoEXT::default()
                    .message_severity(
                        vk::DebugUtilsMessageSeverityFlagsEXT::ERROR
                            | vk::DebugUtilsMessageSeverityFlagsEXT::WARNING
                            | vk::DebugUtilsMessageSeverityFlagsEXT::INFO,
                    )
                    .message_type(
                        vk::DebugUtilsMessageTypeFlagsEXT::GENERAL
                            | vk::DebugUtilsMessageTypeFlagsEXT::VALIDATION
                            | vk::DebugUtilsMessageTypeFlagsEXT::PERFORMANCE,
                    )
                    .pfn_user_callback(Some(Self::vulkan_debug_callback));
                let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
                let debug_callback =
                    debug_utils_loader.create_debug_utils_messenger(&debug_info, None)?;
                Some(debug_callback)
            } else {
                None
            };
            defer! {
                if err.get()
                    && let Some(debug_callback) = debug_callback {
                        let debug_utils_loader = ash::ext::debug_utils::Instance::new(&entry, &instance);
                        debug_utils_loader.destroy_debug_utils_messenger(debug_callback, None);
                    }
            }
            let ext = [
                (khr::synchronization2::NAME, vk::API_VERSION_1_3),
                (khr::timeline_semaphore::NAME, vk::API_VERSION_1_2),
                (
                    khr::get_physical_device_properties2::NAME,
                    vk::API_VERSION_1_1,
                ),
                (khr::shader_atomic_int64::NAME, vk::API_VERSION_1_2),
            ];
            let (phyd, queue_family_idx, api_version, extension_spirv_version) = {
                let mut best_score = 0;
                let mut pair = (vk::PhysicalDevice::null(), 0, 0, types::SpirvVersion::V1_0);
                'outer: for phyd in instance.enumerate_physical_devices()? {
                    let properties = instance.get_physical_device_properties(phyd);
                    let api_version = properties.api_version.min(instance_api_version);
                    let extensions = instance.enumerate_device_extension_properties(phyd)?;
                    let extensions: Vec<&CStr> = extensions
                        .iter()
                        .map(|a| a.extension_name_as_c_str().unwrap())
                        .collect();
                    let mut extension_spirv_version = types::SpirvVersion::V1_0;
                    let caps = instance.get_physical_device_features(phyd);
                    if caps.shader_int64 == 0 {
                        continue;
                    }
                    for extension in ext {
                        if extension.1 > api_version && !extensions.contains(&extension.0) {
                            continue 'outer;
                        }
                        if extension.0 == khr::spirv_1_4::NAME && api_version >= vk::API_VERSION_1_1
                        {
                            extension_spirv_version = types::SpirvVersion::V1_4;
                        }
                    }
                    let queue_families = instance.get_physical_device_queue_family_properties(phyd);
                    let mut best_queue = (0, 0);
                    for (i, queue) in queue_families.into_iter().enumerate() {
                        let flags = if queue.queue_flags.contains(vk::QueueFlags::TRANSFER) {
                            queue.queue_flags ^ vk::QueueFlags::TRANSFER
                        } else {
                            queue.queue_flags
                        };
                        if !queue.queue_flags.contains(vk::QueueFlags::COMPUTE) {
                            continue;
                        }
                        let score;
                        if flags == vk::QueueFlags::COMPUTE {
                            score = 3;
                        } else if !flags.contains(vk::QueueFlags::GRAPHICS) {
                            score = 2;
                        } else {
                            score = 1;
                        }
                        if score > best_queue.0 {
                            best_queue = (score, i);
                        }
                    }
                    if best_queue.0 == 0 {
                        continue;
                    }
                    // In order of priority:
                    // * Prefer discrete gpus
                    // * Then prefer one with a more specific compute queue
                    // * Then prefer the higher API version
                    let score = match properties.device_type {
                        vk::PhysicalDeviceType::DISCRETE_GPU => 3,
                        vk::PhysicalDeviceType::VIRTUAL_GPU => 2,
                        vk::PhysicalDeviceType::INTEGRATED_GPU => 1,
                        vk::PhysicalDeviceType::CPU => 0,
                        _ => continue,
                    } * 16
                        + best_queue.0 * 4
                        + vk::api_version_minor(api_version).min(3);
                    if score > best_score {
                        best_score = score;
                        pair = (
                            phyd,
                            best_queue.1 as u32,
                            api_version,
                            extension_spirv_version,
                        );
                    }
                }
                if best_score > 0 {
                    pair
                } else {
                    return Err(VulkanError::NoSupportedDevice);
                }
            };
            let api_supported_spirv_version = match api_version {
                vk::API_VERSION_1_0 => types::SpirvVersion::V1_0,
                vk::API_VERSION_1_1 => types::SpirvVersion::V1_3,
                vk::API_VERSION_1_2 => types::SpirvVersion::V1_5,
                vk::API_VERSION_1_3 => types::SpirvVersion::V1_6,
                v => {
                    if v > vk::API_VERSION_1_3 {
                        types::SpirvVersion::V1_6
                    } else {
                        panic!("Unrecognized phyd api version!");
                    }
                }
            };
            let mut ext: Vec<_> = ext
                .iter()
                .filter_map(|(ext, api)| {
                    if *api > api_version {
                        Some(ext.as_ptr())
                    } else {
                        None
                    }
                })
                .collect();
            let spirv_version = if api_supported_spirv_version >= extension_spirv_version {
                api_supported_spirv_version
            } else {
                match extension_spirv_version {
                    types::SpirvVersion::V1_4 => {
                        ext.push(khr::shader_float_controls::NAME.as_ptr());
                        ext.push(khr::spirv_1_4::NAME.as_ptr());
                    }
                    _ => unreachable!(),
                };
                extension_spirv_version
            };
            let mut timeline_semaphore =
                vk::PhysicalDeviceTimelineSemaphoreFeatures::default().timeline_semaphore(true);
            let mut sync2 =
                vk::PhysicalDeviceSynchronization2Features::default().synchronization2(true);
            let mut atomic_int64 = vk::PhysicalDeviceShaderAtomicInt64Features::default()
                .shader_shared_int64_atomics(true)
                .shader_buffer_int64_atomics(true);
            let phyd_features = vk::PhysicalDeviceFeatures::default().shader_int64(true);
            // TODO: investigate multiple queues. currently we only use a general queue, but this could potentially be optimized by using special compute queues and special transfer queues
            let queue_priority = 1.0;
            let queue_create_info = vk::DeviceQueueCreateInfo::default()
                .queue_priorities(std::slice::from_ref(&queue_priority))
                .queue_family_index(queue_family_idx);
            let dev_create_info = vk::DeviceCreateInfo::default()
                .queue_create_infos(std::slice::from_ref(&queue_create_info))
                .enabled_extension_names(&ext)
                .enabled_features(&phyd_features)
                .push_next(&mut timeline_semaphore)
                .push_next(&mut sync2)
                .push_next(&mut atomic_int64);
            let device = instance.create_device(phyd, &dev_create_info, None)?;
            defer! {
                if err.get() {
                    device.destroy_device(None);
                }
            }
            let queue = device.get_device_queue(queue_family_idx, 0);
            let s = Self::from_existing(
                entry.clone(),
                instance.clone(),
                device.clone(),
                phyd,
                queue,
                queue_family_idx,
                debug_callback,
                spirv_version,
                api_version,
                None,
            )?;
            err.set(false);
            Ok(s)
        }
    }

    /// # Safety
    /// * Queue family must support `COMPUTE`
    /// * Queue must be of the given queue family, and belong to the given device
    /// * Phyd must be from the given vulkan instance
    /// * Device must be from the given physical device, and must support timeline semaphores and synchronization 2
    /// * The queue must not be used outside of this hal instance
    /// * All resources belonging to the vulkan instance must be destroyed before the hal instance
    /// * The instance and all resources will be destroyed when the hal instance is destroyed
    /// * Instance must be created with API version at least 1.1
    /// * The device must have a high enough API version or support these features:
    ///   * Timeline semaphores
    ///   * Synchronization2
    #[allow(clippy::too_many_arguments)]
    pub unsafe fn from_existing(
        entry: ash::Entry,
        instance: ash::Instance,
        device: ash::Device,
        phyd: vk::PhysicalDevice,
        queue: vk::Queue,
        queue_family_idx: u32,
        debug_callback: Option<vk::DebugUtilsMessengerEXT>,
        spirv_version: types::SpirvVersion,
        api_version: u32,
        force_is_unified_memory: Option<bool>,
    ) -> Result<InstanceDescriptor<Vulkan>, VulkanError> {
        unsafe {
            let create_info = AllocatorCreateInfo::new(&instance, &device, phyd);
            let alloc = Allocator::new(create_info).unwrap();
            let command_pool = device.create_command_pool(
                &vk::CommandPoolCreateInfo::default()
                    .queue_family_index(queue_family_idx)
                    .flags(vk::CommandPoolCreateFlags::RESET_COMMAND_BUFFER),
                None,
            )?;
            let sync2_device = if api_version < vk::API_VERSION_1_3 {
                Some(khr::synchronization2::Device::new(&instance, &device))
            } else {
                None
            };
            let timeline_device = if api_version < vk::API_VERSION_1_2 {
                Some(khr::timeline_semaphore::Device::new(&instance, &device))
            } else {
                None
            };
            let is_unified_memory = if let Some(u) = force_is_unified_memory {
                u
            } else {
                let memory_props = instance.get_physical_device_memory_properties(phyd);
                // The idea here is that implementations are required to provide a host visible and coherent memory type.
                // Heaps and types are different, but implementations generally provide (pinned) system memory as a heap.
                // If this heap isn't separate from the device heap, we know any system memory type is also device local.
                // The spec recommends this for UMA systems: https://registry.khronos.org/vulkan/specs/latest/html/vkspec.html#memory-device
                memory_props.memory_heaps.len() == 1
                    && memory_props.memory_heaps_as_slice()[0]
                        .flags
                        .contains(vk::MemoryHeapFlags::DEVICE_LOCAL)
            };
            let instance = VulkanInstance {
                entry,
                instance,
                _phyd: phyd,
                debug: debug_callback,
                shared: Arc::new(SharedDeviceInfo {
                    functions: DeviceFunctions {
                        device,
                        sync2_device,
                        timeline_device,
                    },
                    queue_family_indices: vec![queue_family_idx, vk::QUEUE_FAMILY_EXTERNAL],
                }),
                spirv_version,
                _api_version: api_version,
                atomic_int64: true,
            };
            let device = VulkanDevice {
                shared: instance.shared.clone(),
                alloc,
                is_unified_memory,
            };
            let stream = VulkanStream {
                shared: instance.shared.clone(),
                queue,
                queue_family_idx,
                command_pool,
                unused_command_buffers: Mutex::new(Vec::new()),
            };
            Ok(InstanceDescriptor {
                instance,
                devices: vec![DeviceDescriptor {
                    device,
                    streams: vec![StreamDescriptor {
                        stream,
                        stream_type: crate::StreamType::ComputeAndTransfer,
                    }],
                    group_idx: None,
                }],
            })
        }
    }
}
