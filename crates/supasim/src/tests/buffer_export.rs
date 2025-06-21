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
use std::any::TypeId;

use crate::SupaSimInstance;
use crate::wgpu;

pub fn buffer_export<Backend: hal::Backend>(hal: Backend::Instance) -> Result<(), ()> {
    // Test specific stuff
    if TypeId::of::<Backend>() == TypeId::of::<hal::Dummy>() {
        // We don't want to "test" on the dummy backend, where nothing happens so the result will be wrong
        return Ok(());
    }
    println!("Hello, world!");
    dev_utils::setup_trace_printer_if_env();

    // Create the instance
    let instance: SupaSimInstance<Backend> = SupaSimInstance::from_hal(hal);
    if !instance.properties().unwrap().export_buffers {
        println!("Skipping test as instance doesn't have export memory property");
        return Ok(());
    }
    let wgpu_instance = wgpu::Instance::new(&wgpu::InstanceDescriptor {
        backends: wgpu::Backends::VULKAN,
        flags: wgpu::InstanceFlags::all(),
        ..Default::default()
    });
    let required_features = if cfg!(windows) {
        wgpu::Features::VULKAN_EXTERNAL_MEMORY_WIN32
    } else {
        wgpu::Features::VULKAN_EXTERNAL_MEMORY_FD
    };
    let adapter = match wgpu_instance
        .enumerate_adapters(wgpu::Backends::VULKAN)
        .into_iter()
        .find(|a| a.features().contains(required_features))
    {
        Some(adapter) => adapter,
        None => {
            println!("Skipping test as wgpu device doesn't have export memory property");
            return Ok(());
        }
    };
    let (device, queue) = pollster::block_on(adapter.request_device(&wgpu::DeviceDescriptor {
        required_features,
        ..Default::default()
    }))
    .unwrap();
    let device_info = crate::WgpuDeviceInfo {
        device: device.clone(),
        features: required_features,
        backend: adapter.get_info().backend,
    };

    let supasim_buffer = instance
        .create_buffer(&crate::BufferDescriptor {
            size: 16,
            buffer_type: crate::BufferType::Gpu,
            contents_align: 4,
            priority: 1.0,
            can_export: true,
        })
        .unwrap();
    {
        let recorder = instance.create_recorder().unwrap();
        recorder
            .write_buffer(&supasim_buffer, 0, &[1u32, 2, 3, 4])
            .unwrap();
        instance
            .submit_commands(&mut [recorder])
            .unwrap()
            .wait()
            .unwrap();
    }
    let buffer = unsafe { supasim_buffer.export_to_wgpu(device_info).unwrap() };
    let wgpu_download_buffer = device.create_buffer(&wgpu::BufferDescriptor {
        label: None,
        size: 16,
        usage: wgpu::BufferUsages::MAP_READ | wgpu::BufferUsages::COPY_DST,
        mapped_at_creation: false,
    });
    {
        let mut recorder =
            device.create_command_encoder(&wgpu::CommandEncoderDescriptor { label: None });
        recorder.copy_buffer_to_buffer(&buffer, 0, &wgpu_download_buffer, 0, 16);
        queue.submit([recorder.finish()]);
        wgpu_download_buffer.map_async(wgpu::MapMode::Read, 0..16, |_| ());
        device.poll(wgpu::PollType::Wait).unwrap();
    }
    let mapping = &*wgpu_download_buffer.get_mapped_range(0..16);
    let slice = bytemuck::cast_slice::<u8, u32>(mapping);
    assert_eq!(slice, [1, 2, 3, 4]);
    println!("Buffer export completed");

    Ok(())
}

dev_utils::all_backend_tests!(buffer_export);
