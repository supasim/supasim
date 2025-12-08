/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 Magnus Larsson
  SPDX-License-Identifier: MIT OR Apache-2.0
END LICENSE */

use crate::{
    Backend, BufferCommand, CommandRecorder, HalBufferSlice, Vulkan,
    vulkan::{VulkanBuffer, VulkanError, VulkanStream},
};
use ash::vk;
use std::fmt::Debug;
use types::SyncOperations;

#[derive(Debug)]
pub struct VulkanCommandRecorder {
    pub inner: vk::CommandBuffer,
}

impl VulkanCommandRecorder {
    fn begin(
        &mut self,
        stream: &<Vulkan as Backend>::Stream,
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            stream.shared.functions.begin_command_buffer(
                cb,
                &vk::CommandBufferBeginInfo::default()
                    .flags(vk::CommandBufferUsageFlags::ONE_TIME_SUBMIT),
            )?;
            Ok(())
        }
    }

    fn end(
        &mut self,
        stream: &<Vulkan as Backend>::Stream,
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            stream.shared.functions.end_command_buffer(cb)?;
            Ok(())
        }
    }

    #[allow(clippy::too_many_arguments)]
    fn copy_buffer(
        &mut self,
        stream: &<Vulkan as Backend>::Stream,
        src_buffer: &<Vulkan as Backend>::Buffer,
        dst_buffer: &<Vulkan as Backend>::Buffer,
        src_offset: u64,
        dst_offset: u64,
        size: u64,
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            stream.shared.functions.cmd_copy_buffer(
                cb,
                src_buffer.buffer,
                dst_buffer.buffer,
                &[vk::BufferCopy::default()
                    .src_offset(src_offset)
                    .dst_offset(dst_offset)
                    .size(size)],
            );
        }
        Ok(())
    }

    fn dispatch_kernel(
        &mut self,
        stream: &<Vulkan as Backend>::Stream,
        kernel: &<Vulkan as Backend>::Kernel,
        descriptor_set: &<Vulkan as Backend>::BindGroup,
        push_constants: &[u8],
        workgroup_dims: [u32; 3],
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            stream.shared.functions.cmd_bind_pipeline(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline,
            );
            stream.shared.functions.cmd_bind_descriptor_sets(
                cb,
                vk::PipelineBindPoint::COMPUTE,
                kernel.pipeline_layout,
                0,
                &[descriptor_set.inner],
                &[],
            );
            if !push_constants.is_empty() {
                stream.shared.functions.cmd_push_constants(
                    cb,
                    kernel.pipeline_layout,
                    vk::ShaderStageFlags::COMPUTE,
                    0,
                    push_constants,
                );
            }
            stream.shared.functions.cmd_dispatch(
                cb,
                workgroup_dims[0],
                workgroup_dims[1],
                workgroup_dims[2],
            );
        }
        Ok(())
    }

    fn zero_memory(
        &mut self,
        stream: &<Vulkan as Backend>::Stream,
        buffer: &VulkanBuffer,
        offset: u64,
        size: u64,
        cb: vk::CommandBuffer,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            stream
                .shared
                .functions
                .cmd_fill_buffer(cb, buffer.buffer, offset, size, 0);
        }
        Ok(())
    }

    fn stage_mask_khr(sync_ops: SyncOperations) -> vk::PipelineStageFlags2KHR {
        match sync_ops {
            SyncOperations::None => vk::PipelineStageFlags2KHR::empty(),
            SyncOperations::Transfer => vk::PipelineStageFlags2KHR::TRANSFER,
            SyncOperations::ComputeDispatch => vk::PipelineStageFlags2KHR::COMPUTE_SHADER,
            SyncOperations::Both => vk::PipelineStageFlags2KHR::ALL_COMMANDS,
        }
    }

    /// First command must be a pipeline barrier. Following commands must be memory barriers
    fn sync_command<'a>(
        &mut self,
        stream: &<Vulkan as Backend>::Stream,
        cb: vk::CommandBuffer,
        commands: impl IntoIterator<Item = &'a BufferCommand<'a, Vulkan>>,
    ) -> Result<(), VulkanError> {
        let mut barriers = Vec::new();
        let mut pre_flags = vk::PipelineStageFlags2KHR::empty();
        let mut post_flags = vk::PipelineStageFlags2KHR::empty();
        for command in commands {
            match command {
                BufferCommand::MemoryBarrier {
                    buffer:
                        HalBufferSlice {
                            buffer,
                            offset,
                            length: len,
                        },
                } => barriers.push(
                    vk::BufferMemoryBarrier2KHR::default()
                        .buffer(buffer.buffer)
                        .offset(*offset)
                        .size(*len)
                        .src_queue_family_index(stream.queue_family_idx)
                        .dst_queue_family_index(stream.queue_family_idx)
                        .src_access_mask(
                            vk::AccessFlags2KHR::MEMORY_READ_KHR
                                | vk::AccessFlags2KHR::MEMORY_WRITE_KHR,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2KHR::MEMORY_READ_KHR
                                | vk::AccessFlags2KHR::MEMORY_WRITE_KHR,
                        ),
                ),
                BufferCommand::MemoryTransfer {
                    buffer:
                        HalBufferSlice {
                            buffer,
                            offset,
                            length: len,
                        },
                    import,
                } => barriers.push(
                    vk::BufferMemoryBarrier2KHR::default()
                        .buffer(buffer.buffer)
                        .offset(*offset)
                        .size(*len)
                        .src_queue_family_index(if *import {
                            vk::QUEUE_FAMILY_EXTERNAL
                        } else {
                            stream.queue_family_idx
                        })
                        .dst_queue_family_index(if *import {
                            stream.queue_family_idx
                        } else {
                            vk::QUEUE_FAMILY_EXTERNAL
                        })
                        .src_access_mask(
                            vk::AccessFlags2KHR::MEMORY_READ_KHR
                                | vk::AccessFlags2KHR::MEMORY_WRITE_KHR,
                        )
                        .dst_access_mask(
                            vk::AccessFlags2KHR::MEMORY_READ_KHR
                                | vk::AccessFlags2KHR::MEMORY_WRITE_KHR,
                        ),
                ),
                BufferCommand::PipelineBarrier { before, after } => {
                    pre_flags |= Self::stage_mask_khr(*before);
                    post_flags |= Self::stage_mask_khr(*after);
                }
                _ => unreachable!(),
            }
        }

        if pre_flags.is_empty() || post_flags.is_empty() {
            return Ok(());
        }
        for barrier in &mut barriers {
            *barrier = barrier.src_stage_mask(pre_flags).dst_stage_mask(post_flags);
        }
        let dependency_info = vk::DependencyInfoKHR::default().buffer_memory_barriers(&barriers);
        unsafe {
            stream
                .shared
                .functions
                .supa_cmd_pipeline_barrier2(cb, &dependency_info)
        };
        Ok(())
    }

    fn record_command(
        &mut self,
        stream: &VulkanStream,
        cb: vk::CommandBuffer,
        command: &BufferCommand<Vulkan>,
    ) -> Result<(), VulkanError> {
        match command {
            BufferCommand::CopyBuffer {
                src_buffer,
                dst_buffer,
                src_offset,
                dst_offset,
                len,
            } => self.copy_buffer(
                stream,
                src_buffer,
                dst_buffer,
                *src_offset,
                *dst_offset,
                *len,
                cb,
            )?,
            BufferCommand::ZeroMemory { buffer } => {
                self.zero_memory(stream, buffer.buffer, buffer.offset, buffer.length, cb)?;
            }
            BufferCommand::DispatchKernel {
                kernel,
                bind_group,
                push_constants,
                workgroup_dims,
            } => self.dispatch_kernel(
                stream,
                kernel,
                bind_group,
                push_constants,
                *workgroup_dims,
                cb,
            )?,

            BufferCommand::PipelineBarrier { .. }
            | BufferCommand::MemoryBarrier { .. }
            | BufferCommand::MemoryTransfer { .. } => {
                unreachable!()
            }
            BufferCommand::UpdateBindGroup { .. } => unreachable!(),
            BufferCommand::Dummy => (),
        }
        Ok(())
    }
}

impl CommandRecorder<Vulkan> for VulkanCommandRecorder {
    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn record_commands(
        &mut self,
        stream: &VulkanStream,
        commands: &[crate::BufferCommand<Vulkan>],
    ) -> Result<(), <Vulkan as Backend>::Error> {
        let cb = self.inner;
        self.begin(stream, cb)?;
        let mut pipeline_chain_start = None;
        for i in 0..commands.len() {
            match &commands[i] {
                BufferCommand::MemoryBarrier { .. }
                | BufferCommand::PipelineBarrier { .. }
                | BufferCommand::MemoryTransfer { .. } => {
                    if pipeline_chain_start.is_none() {
                        pipeline_chain_start = Some(i);
                    }
                }
                _ => {
                    if let Some(start) = pipeline_chain_start {
                        self.sync_command(stream, cb, &commands[start..i])?;
                        pipeline_chain_start = None;
                    }
                    self.record_command(stream, cb, &commands[i])?;
                }
            }
        }
        self.end(stream, cb)?;
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn clear(
        &mut self,
        stream: &<Vulkan as Backend>::Stream,
    ) -> Result<(), <Vulkan as Backend>::Error> {
        unsafe {
            stream
                .shared
                .functions
                .reset_command_buffer(self.inner, vk::CommandBufferResetFlags::RELEASE_RESOURCES)?;
        }
        Ok(())
    }

    #[cfg_attr(feature = "trace", tracing::instrument)]
    unsafe fn destroy(self, stream: &VulkanStream) -> Result<(), <Vulkan as Backend>::Error> {
        stream
            .unused_command_buffers
            .lock()
            .unwrap()
            .push(self.inner);
        Ok(())
    }
}
