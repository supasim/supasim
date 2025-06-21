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
use hal::{BackendInstance, CommandRecorder, HalBufferSlice, RecorderSubmitInfo, Semaphore};
use parking_lot::{Condvar, Mutex};
use std::collections::{HashMap, hash_map::Entry};
use std::marker::PhantomData;
use std::ops::Deref;
use std::panic::UnwindSafe;
use std::sync::Arc;
use std::sync::mpsc::{Receiver, RecvTimeoutError, Sender, channel};
use std::time::{Duration, Instant};
use thunderdome::Index;
use types::{Dag, NodeIndex, SyncOperations, Walker};

use crate::{
    BufferCommand, BufferCommandInner, BufferRange, BufferSlice, BufferUserId, BufferWeak,
    CommandRecorderInner, InstanceState, KernelWeak, MapSupasimError, SavedKernel, SupaSimError,
    SupaSimInstance, SupaSimResult,
};
use anyhow::anyhow;

pub type CommandDag<B> = Dag<BufferCommand<B>>;

pub enum HalCommandBuilder {
    CopyBuffer {
        src_buffer: Index,
        dst_buffer: Index,
        src_offset: u64,
        dst_offset: u64,
        len: u64,
    },
    CopyFromTemp {
        src_offset: u64,
        dst_buffer: Index,
        dst_offset: u64,
        len: u64,
    },
    DispatchKernel {
        kernel: Index,
        bg: u32,
        push_constants: Vec<u8>,
        workgroup_dims: [u32; 3],
    },
    /// Only for vulkan like synchronization
    PipelineBarrier {
        before: SyncOperations,
        after: SyncOperations,
    },
    /// Only for vulkan like synchronization. Will hitch a ride with the previous PipelineBarrier or WaitEvent
    MemoryBarrier {
        resource: Index,
        offset: u64,
        len: u64,
    },
    UpdateBindGroup {
        bg: Index,
        kernel: Index,
        resources: Vec<Index>,
    },
    Dummy,
}
pub struct BindGroupDesc {
    kernel_idx: Index,
    items: Vec<(Index, BufferRange)>,
}
pub struct CommandStream {
    pub commands: Vec<HalCommandBuilder>,
}
/// These are split into multiple streams so that certain operations can be waited without waiting for all
pub struct StreamingCommands {
    /// Contains the index of the kernel, the index of the buffer, and the range of the buffer
    pub bind_groups: Vec<BindGroupDesc>,
    pub streams: Vec<CommandStream>,
}

#[allow(clippy::type_complexity)]
pub fn assemble_dag<B: hal::Backend>(
    crs: &mut [&mut CommandRecorderInner<B>],
    used_kernels: &mut Vec<KernelWeak<B>>,
    instance: &InstanceState<B>,
) -> SupaSimResult<
    B,
    (
        CommandDag<B>,
        HashMap<Index, Vec<BufferRange>>,
        Option<B::Buffer>,
    ),
> {
    let mut buffers_tracker: HashMap<Index, Vec<(BufferRange, usize)>> = HashMap::new();

    let mut commands = Vec::new();
    let mut src_buffer_len = 0;
    for cr in crs.iter_mut() {
        let mut cmds = std::mem::take(&mut cr.commands);
        for cmd in &mut cmds {
            if let BufferCommandInner::CopyFromTemp { src_offset } = &mut cmd.inner {
                *src_offset += src_buffer_len;
            }
        }
        commands.extend(cmds);
        commands.push(BufferCommand {
            inner: BufferCommandInner::CommandRecorderEnd,
            buffers: vec![],
        });
        src_buffer_len += cr.writes_slice.len() as u64;
    }

    let mut dag = Dag::new();
    for cmd in commands {
        dag.add_node(cmd);
    }

    for i in 0..dag.node_count() {
        let mut work_on_buffer =
            |buffer: &BufferSlice<B>, dag: &mut Dag<BufferCommand<B>>| -> SupaSimResult<B, ()> {
                let range: BufferRange = buffer.into();
                let id = buffer.buffer.inner()?.id;
                match buffers_tracker.entry(id) {
                    Entry::Occupied(mut entry) => {
                        for (range2, j) in entry.get().iter() {
                            if range.overlaps(range2) {
                                dag.add_edge(NodeIndex::new(*j), NodeIndex::new(i), ())
                                    .unwrap();
                            }
                        }
                        entry.get_mut().push((range, i));
                    }
                    Entry::Vacant(entry) => {
                        entry.insert(vec![(range, i)]);
                    }
                }
                Ok(())
            };
        for bf_idx in 0..dag[NodeIndex::new(i)].buffers.len() {
            let buffer = dag[NodeIndex::new(i)].buffers[bf_idx].clone();
            work_on_buffer(&buffer, &mut dag)?;
        }
        if let BufferCommandInner::KernelDispatch { kernel, .. } = &dag[NodeIndex::new(i)].inner {
            used_kernels.push(kernel.downgrade());
        }
    }
    dag.add_node(BufferCommand {
        inner: BufferCommandInner::Dummy,
        buffers: vec![],
    });
    for i in 0..dag.node_count() - 1 {
        dag.add_edge(NodeIndex::new(dag.node_count() - 1), NodeIndex::new(i), ())
            .unwrap();
    }
    dag.transitive_reduce(vec![NodeIndex::new(dag.node_count() - 1)]);
    let out_map = buffers_tracker
        .into_iter()
        .map(|(key, value)| (key, value.iter().map(|a| a.0).collect()))
        .collect();
    let src_buffer = if src_buffer_len > 0 {
        let mut buf = unsafe {
            instance
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .create_buffer(&types::HalBufferDescriptor {
                    size: src_buffer_len,
                    memory_type: types::HalBufferType::Upload,
                    min_alignment: 16,
                    can_export: false,
                })
                .map_supasim()?
        };
        let mut current_offset = 0;
        for cr in crs.iter_mut() {
            unsafe {
                instance
                    .inner
                    .lock()
                    .as_mut()
                    .unwrap()
                    .write_buffer(&mut buf, current_offset, &cr.writes_slice)
                    .map_supasim()?;
            }
            current_offset += cr.writes_slice.len() as u64;
        }
        Some(buf)
    } else {
        None
    };
    Ok((dag, out_map, src_buffer))
}
#[allow(clippy::type_complexity)]
pub fn record_dag<B: hal::Backend>(
    _dag: &CommandDag<B>,
    _cr: &mut B::CommandRecorder,
) -> SupaSimResult<B, Vec<(B::BindGroup, Index)>> {
    // TODO: work on this when cuda support lands
    todo!()
}
pub fn dag_to_command_streams<B: hal::Backend>(
    dag: &CommandDag<B>,
    vulkan_style: bool,
) -> SupaSimResult<B, StreamingCommands> {
    let mut bind_groups = Vec::new();
    let mut stream = CommandStream {
        commands: Vec::new(),
    };
    {
        // This algorithm fucking sucks. Its like topological sort but the layers are distinct, so that synchronization can be applied only at specific points
        let mut already_in = Vec::new();
        already_in.resize(dag.node_count(), false);
        let mut layers = Vec::new();
        layers.push(Vec::new());
        // I think it looks nicer
        #[allow(clippy::needless_range_loop)]
        for i in 0..dag.node_count() {
            if dag.parents(NodeIndex::new(i)).walk_next(dag).is_none() {
                layers[0].push(i);
                already_in[i] = true;
            }
        }
        // This next code is awful. But the idea is that we separate things into "layers" based on
        // the highest parent layer plus 1. This way it always comes after each parent. So we loop
        // over all newly exposed children of the previous layer, and if they have no incomplete
        // parents, it can be added to the current layer.
        while !layers.last().unwrap().is_empty() {
            let mut next_layer = Vec::new();
            let last_layer = layers.last().unwrap();
            for &node in last_layer {
                let mut walker = dag.children(NodeIndex::new(node));
                while let Some((_, child)) = walker.walk_next(dag) {
                    if !already_in[child.index()] {
                        let mut walker2 = dag.parents(child);
                        let mut can_complete = true;
                        while let Some((_, parent)) = walker2.walk_next(dag) {
                            if !already_in[parent.index()] {
                                can_complete = false;
                            }
                        }
                        if can_complete {
                            next_layer.push(child.index());
                            already_in[child.index()] = true;
                        }
                    }
                }
            }
            layers.push(next_layer);
        }
        layers.pop();
        let nodes = dag.raw_nodes();
        for (i, layer) in layers.into_iter().enumerate() {
            // No synchronization needed for the first layer
            // This following code is bad. More barriers than needed are used, and the first layer isn't actually skipped.
            // I suspect this is because the first layer has a kind of "root" dummy node
            if vulkan_style && i != 0 {
                stream.commands.push(HalCommandBuilder::PipelineBarrier {
                    before: SyncOperations::Both,
                    after: SyncOperations::Both,
                });
                for &idx in &layer {
                    let cmd = &nodes[idx].weight;
                    if let BufferCommandInner::CopyBufferToBuffer = &cmd.inner {
                        stream.commands.push(HalCommandBuilder::MemoryBarrier {
                            resource: cmd.buffers[0].buffer.inner()?.id,
                            offset: cmd.buffers[0].start,
                            len: cmd.buffers[0].len,
                        });
                        stream.commands.push(HalCommandBuilder::MemoryBarrier {
                            resource: cmd.buffers[1].buffer.inner()?.id,
                            offset: cmd.buffers[1].start,
                            len: cmd.buffers[1].len,
                        });
                    } else {
                        for buffer in &cmd.buffers {
                            let id = buffer.buffer.inner()?.id;
                            stream.commands.push(HalCommandBuilder::MemoryBarrier {
                                resource: id,
                                offset: buffer.start,
                                len: buffer.len,
                            });
                        }
                    }
                }
            }
            for idx in layer {
                let cmd = &nodes[idx].weight;
                let hal = match &cmd.inner {
                    BufferCommandInner::Dummy => continue,
                    BufferCommandInner::CopyBufferToBuffer => HalCommandBuilder::CopyBuffer {
                        src_buffer: cmd.buffers[0].buffer.inner()?.id,
                        dst_buffer: cmd.buffers[1].buffer.inner()?.id,
                        src_offset: cmd.buffers[0].start,
                        dst_offset: cmd.buffers[1].start,
                        len: cmd.buffers[0].len,
                    },
                    BufferCommandInner::KernelDispatch {
                        kernel,
                        workgroup_dims,
                    } => {
                        let bg_index = bind_groups.len() as u32;
                        let bg = BindGroupDesc {
                            kernel_idx: kernel.inner()?.inner.lock().id,
                            items: cmd
                                .buffers
                                .iter()
                                .map(|a| {
                                    (
                                        a.buffer.inner().unwrap().id,
                                        BufferRange {
                                            start: a.start,
                                            len: a.len,
                                            needs_mut: a.needs_mut,
                                        },
                                    )
                                })
                                .collect(),
                        };
                        bind_groups.push(bg);
                        HalCommandBuilder::DispatchKernel {
                            kernel: kernel.inner()?.inner.lock().id,
                            bg: bg_index,
                            push_constants: Vec::new(),
                            workgroup_dims: *workgroup_dims,
                        }
                    }
                    BufferCommandInner::CopyFromTemp { src_offset } => {
                        HalCommandBuilder::CopyFromTemp {
                            src_offset: *src_offset,
                            dst_buffer: cmd.buffers[0].buffer.inner()?.id,
                            dst_offset: cmd.buffers[0].start,
                            len: cmd.buffers[0].len,
                        }
                    }
                    BufferCommandInner::CommandRecorderEnd => HalCommandBuilder::Dummy,
                };
                stream.commands.push(hal);
            }
        }
    }
    Ok(StreamingCommands {
        bind_groups,
        streams: vec![stream],
    })
}
#[allow(clippy::type_complexity)]
pub fn record_command_streams<B: hal::Backend>(
    streams: &StreamingCommands,
    instance: SupaSimInstance<B>,
    recorder: &mut B::CommandRecorder,
    write_buffer: &Option<B::Buffer>,
) -> SupaSimResult<B, Vec<(B::BindGroup, Index)>> {
    let instance = instance.inner()?;
    let mut bindgroups = Vec::new();
    for bg in &streams.bind_groups {
        let _k = instance
            .kernels
            .lock()
            .get(bg.kernel_idx)
            .ok_or(SupaSimError::AlreadyDestroyed("Kernel".to_owned()))?
            .loaded_ref()
            .upgrade()?;
        let kernel = _k.inner()?;
        let mut resources_a = Vec::new();
        for res in &bg.items {
            resources_a.push(
                instance
                    .buffers
                    .lock()
                    .get(res.0)
                    .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                    .as_ref()
                    .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                    .upgrade()?,
            );
        }
        let mut resource_locks = Vec::new();
        for res in &resources_a {
            resource_locks.push(res.inner()?);
        }
        let mut resources = Vec::new();
        for (i, res) in resource_locks.iter().enumerate() {
            let range = bg.items[i].1;
            resources.push(hal::HalBufferSlice {
                buffer: res.inner.as_ref().unwrap(),
                offset: range.start,
                len: range.len,
            });
        }
        let bg = unsafe {
            instance
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .create_bind_group(kernel.inner.lock().inner.as_mut().unwrap(), &resources)
                .map_supasim()?
        };
        bindgroups.push((bg, kernel.inner.lock().id));
    }
    for stream in &streams.streams {
        let mut buffer_refs = Vec::new();
        let mut kernel_refs = Vec::new();
        for cmd in &stream.commands {
            match cmd {
                HalCommandBuilder::CopyBuffer {
                    src_buffer,
                    dst_buffer,
                    ..
                } => {
                    buffer_refs.push(
                        instance
                            .buffers
                            .lock()
                            .get(*src_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                    buffer_refs.push(
                        instance
                            .buffers
                            .lock()
                            .get(*dst_buffer)
                            .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                }
                HalCommandBuilder::DispatchKernel { kernel, .. } => {
                    kernel_refs.push(
                        instance
                            .kernels
                            .lock()
                            .get(*kernel)
                            .ok_or(SupaSimError::AlreadyDestroyed("Kernel".to_owned()))?
                            .loaded_ref()
                            .upgrade()?
                            .inner()?
                            .inner
                            .clone(),
                    );
                }
                HalCommandBuilder::MemoryBarrier { resource, .. } => {
                    buffer_refs.push(
                        instance
                            .buffers
                            .lock()
                            .get(*resource)
                            .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                            .as_ref()
                            .unwrap()
                            .upgrade()?,
                    );
                }
                HalCommandBuilder::CopyFromTemp { dst_buffer, .. } => buffer_refs.push(
                    instance
                        .buffers
                        .lock()
                        .get(*dst_buffer)
                        .ok_or(SupaSimError::AlreadyDestroyed("Buffer".to_owned()))?
                        .as_ref()
                        .unwrap()
                        .upgrade()?,
                ),
                _ => (),
            }
        }
        let mut buffer_locks = Vec::new();
        for buffer_ref in &buffer_refs {
            buffer_locks.push(buffer_ref.inner()?);
        }
        let mut kernel_locks = Vec::new();
        for kernel_ref in &kernel_refs {
            kernel_locks.push(kernel_ref.lock());
        }
        let mut hal_commands = Vec::new();
        {
            let mut current_buffer_index = 0;
            let mut current_kernel_index = 0;
            let mut get_buffer = || {
                let buffer = buffer_locks[current_buffer_index]
                    .deref()
                    .inner
                    .as_ref()
                    .unwrap();
                current_buffer_index += 1;
                buffer
            };
            let mut get_kernel = || {
                let kernel = kernel_locks[current_kernel_index].inner.as_ref().unwrap();
                current_kernel_index += 1;
                kernel
            };
            for cmd in &stream.commands {
                let cmd = match cmd {
                    HalCommandBuilder::CopyBuffer {
                        src_offset,
                        dst_offset,
                        len,
                        ..
                    } => hal::BufferCommand::CopyBuffer {
                        src_buffer: get_buffer(),
                        dst_buffer: get_buffer(),
                        src_offset: *src_offset,
                        dst_offset: *dst_offset,
                        len: *len,
                    },
                    HalCommandBuilder::DispatchKernel {
                        bg,
                        push_constants,
                        workgroup_dims,
                        ..
                    } => hal::BufferCommand::DispatchKernel {
                        kernel: get_kernel(),
                        bind_group: &bindgroups[*bg as usize].0,
                        push_constants,
                        workgroup_dims: *workgroup_dims,
                    },
                    HalCommandBuilder::MemoryBarrier { offset, len, .. } => {
                        hal::BufferCommand::MemoryBarrier {
                            buffer: HalBufferSlice {
                                buffer: get_buffer(),
                                offset: *offset,
                                len: *len,
                            },
                        }
                    }
                    HalCommandBuilder::PipelineBarrier { before, after } => {
                        hal::BufferCommand::PipelineBarrier {
                            before: *before,
                            after: *after,
                        }
                    }
                    HalCommandBuilder::CopyFromTemp {
                        src_offset,
                        dst_offset,
                        len,
                        ..
                    } => hal::BufferCommand::CopyBuffer {
                        src_buffer: write_buffer.as_ref().unwrap(),
                        dst_buffer: get_buffer(),
                        src_offset: *src_offset,
                        dst_offset: *dst_offset,
                        len: *len,
                    },
                    HalCommandBuilder::UpdateBindGroup { .. } => todo!(),
                    HalCommandBuilder::Dummy => hal::BufferCommand::Dummy,
                };
                hal_commands.push(cmd);
                // Add the commands n shit
            }
        }
        unsafe {
            recorder
                .record_commands(instance.inner.lock().as_mut().unwrap(), &mut hal_commands)
                .map_supasim()?
        };
    }
    Ok(bindgroups)
}
pub struct GpuSubmissionInfo<B: hal::Backend> {
    pub command_recorder: Option<B::CommandRecorder>,
    pub bind_groups: Vec<(B::BindGroup, Index)>,
    pub used_buffer_ranges: Vec<(BufferUserId, BufferWeak<B>)>,
    pub used_buffers: Vec<BufferWeak<B>>,
}
/// A job for the CPU to run when some GPU work has completed or immediately, without ideally blocking for long. This won't necessarily run before other submissions
pub enum SemaphoreFinishedJob<B: hal::Backend> {
    DestroyBuffer(B::Buffer),
    DestroyKernel(Index),
}
impl<B: hal::Backend> SemaphoreFinishedJob<B> {
    pub fn run(self, instance: &InstanceState<B>) -> SupaSimResult<B, ()> {
        match self {
            Self::DestroyBuffer(b) => unsafe {
                instance
                    .inner
                    .lock()
                    .as_mut()
                    .unwrap()
                    .destroy_buffer(b)
                    .map_supasim()?;
            },
            Self::DestroyKernel(k) => unsafe {
                let k = instance.kernels.lock().remove(k).unwrap();
                match k {
                    SavedKernel::WaitingForDestroy { inner } => instance
                        .inner
                        .lock()
                        .as_mut()
                        .unwrap()
                        .destroy_kernel(inner)
                        .map_supasim()?,
                    _ => unreachable!(),
                }
            },
        }
        Ok(())
    }
}
/// A job for the CPU to run in between GPU submissions
pub enum CpuSubmission<B: hal::Backend> {
    CreateGpuBuffer { buffer_id: Index },
    DestroyGpuBuffer { buffer_id: Index },
    Dummy(PhantomData<B>),
}
impl<B: hal::Backend> CpuSubmission<B> {
    pub fn run(self, _instance: &InstanceState<B>) -> SupaSimResult<B, ()> {
        match self {
            Self::CreateGpuBuffer { .. } => todo!(),
            Self::DestroyGpuBuffer { .. } => todo!(),
            Self::Dummy(_) => (),
        }
        Ok(())
    }
}
/// An event sent to the sync thread
pub enum SendSyncThreadEvent<B: hal::Backend> {
    /// GPU work to be done when all prior work is completed
    AddSubmission(GpuSubmissionInfo<B>),
    /// CPU work to be completed when a submission is done or immediately if it is already complete
    AddFinishedJob(u64, SemaphoreFinishedJob<B>),
    /// CPU work to be completed between submissions
    CpuWork(CpuSubmission<B>),
    /// Any currently queued work will begin immediately instead of waiting for more
    SubmitBatchNow,
    WaitFinishAndShutdown,
}
pub struct SyncThreadSharedData<B: hal::Backend> {
    pub next_job: u64,
    pub error: Option<SupaSimError<B>>,
    pub next_submission_idx: u64,
}
pub type SyncThreadShared<B> = Arc<(Mutex<SyncThreadSharedData<B>>, Condvar)>;
struct SyncThreadData<B: hal::Backend> {
    shared: SyncThreadShared<B>,
    receiver: Receiver<SendSyncThreadEvent<B>>,
    instance: Arc<InstanceState<B>>,
}
impl<B: hal::Backend> UnwindSafe for SyncThreadData<B> {}
pub struct SyncThreadHandle<B: hal::Backend> {
    pub sender: Mutex<Sender<SendSyncThreadEvent<B>>>,
    pub shared_thread: SyncThreadShared<B>,
    pub thread: std::thread::JoinHandle<()>,
}
impl<B: hal::Backend> SyncThreadHandle<B> {
    pub fn submit_gpu(&self, submission: GpuSubmissionInfo<B>) -> SupaSimResult<B, u64> {
        self.sender
            .lock()
            .send(SendSyncThreadEvent::AddSubmission(submission))
            .unwrap();
        let mut lock = self.shared_thread.0.lock();
        if let Some(SupaSimError::SyncThreadPanic(e)) = &lock.error {
            return Err(SupaSimError::SyncThreadPanic(e.clone()));
        }
        let id = lock.next_submission_idx;
        lock.next_submission_idx += 1;
        drop(lock);
        Ok(id)
    }
    pub fn append_finished_job(
        &self,
        idx: u64,
        job: SemaphoreFinishedJob<B>,
    ) -> SupaSimResult<B, ()> {
        self.sender
            .lock()
            .send(SendSyncThreadEvent::AddFinishedJob(idx, job))
            .unwrap();
        Ok(())
    }
    pub fn wait_for(&self, idx: u64, force_wait: bool) -> SupaSimResult<B, bool> {
        if force_wait {
            let mut lock = self.shared_thread.0.lock();
            while lock.next_job <= idx {
                if let Some(SupaSimError::SyncThreadPanic(e)) = &lock.error {
                    return Err(SupaSimError::SyncThreadPanic(e.clone()));
                }
                self.shared_thread.1.wait(&mut lock);
            }
            Ok(true)
        } else {
            Ok(self.shared_thread.0.lock().next_job > idx)
        }
    }
    pub fn wait_for_idle(&self) -> SupaSimResult<B, ()> {
        let mut lock = self.shared_thread.0.lock();
        while lock.next_job < lock.next_submission_idx {
            if let Some(SupaSimError::SyncThreadPanic(e)) = &lock.error {
                return Err(SupaSimError::SyncThreadPanic(e.clone()));
            }
            self.shared_thread.1.wait(&mut lock);
        }
        Ok(())
    }
}
pub fn create_sync_thread<B: hal::Backend>(
    instance: SupaSimInstance<B>,
) -> SupaSimResult<B, SyncThreadHandle<B>> {
    let shared_thread = SyncThreadSharedData::<B> {
        next_job: 1,
        error: None,
        next_submission_idx: 1,
    };
    let shared = Arc::new((Mutex::new(shared_thread), Condvar::new()));
    let shared_copy = shared.clone();
    let (sender, receiver) = channel::<SendSyncThreadEvent<B>>();
    let thread = std::thread::spawn(move || {
        let shared = shared_copy;
        let data = SyncThreadData {
            shared: shared.clone(),
            receiver,
            instance: instance.inner().unwrap()._inner.clone(),
        };
        drop(instance);

        if let Err(e) = std::panic::catch_unwind(|| {
            let mut data = data;
            sync_thread_main(&mut data).unwrap()
        }) {
            let mut lock = shared.0.lock();
            let mut error = String::from("Unknown panic");
            if e.is::<String>() {
                error = *e.downcast::<String>().unwrap();
            }
            lock.error = Some(SupaSimError::SyncThreadPanic(error.clone()));
            shared.1.notify_all();
            drop(lock);
            panic!("Sync thread encountered error: {error}");
        }
    });
    Ok(SyncThreadHandle {
        sender: Mutex::new(sender),
        shared_thread: shared,
        thread,
    })
}
enum Work<B: hal::Backend> {
    GpuSubmission(GpuSubmissionInfo<B>),
    CpuWork(CpuSubmission<B>),
}
fn sync_thread_main<B: hal::Backend>(logic: &mut SyncThreadData<B>) -> Result<(), SupaSimError<B>> {
    const SUBMISSION_WAIT_PERIOD: Duration = Duration::from_millis(10);
    const MAX_SUBMISSION_WINDOW: Duration = Duration::from_millis(50);

    // Loop logic:
    // First, wait for a submission. Record that submission and wait the rest of ~5ms.
    // If there are more submissions during or by the end of this time, also record those. Then submit altogether.
    // A CPU submission before a GPU submission must break up the submission if the device doesn't support CPU semaphore signalling.
    // Otherwise, the following submission must wait on a CPu signalled semaphore
    //
    // Downsides of this are that if recording takes a long time there will be significant downtime. This can be prevented in the future using other methods, such as an intermediate recorder thread.
    let mut jobs = Vec::new();
    let mut next_submission_idx = 1;
    let mut _num_submitted_so_far = 1;
    let (semaphore_signal, map_buffer_while_gpu_use) = {
        (
            logic.instance.inner_properties.semaphore_signal,
            logic.instance.inner_properties.map_buffer_while_gpu_use,
        )
    };
    let mut semaphores = Vec::new();
    let acquire_semaphore = |sems: &mut Vec<B::Semaphore>| -> SupaSimResult<B, B::Semaphore> {
        Ok(if let Some(s) = sems.pop() {
            s
        } else {
            unsafe {
                logic
                    .instance
                    .inner
                    .lock()
                    .as_mut()
                    .unwrap()
                    .create_semaphore()
                    .map_supasim()?
            }
        })
    };
    loop {
        let mut temp_submission_vec = Vec::new();
        let mut submits = Vec::new();
        let mut used_semaphores = Vec::new();
        // Initial stuff - any non GPU work can be completed immediately
        loop {
            match logic.receiver.recv().unwrap() {
                SendSyncThreadEvent::AddFinishedJob(_, job) => job.run(&logic.instance)?,
                SendSyncThreadEvent::CpuWork(job) => job.run(&logic.instance)?,
                SendSyncThreadEvent::WaitFinishAndShutdown => {
                    for semaphore in semaphores {
                        unsafe {
                            logic
                                .instance
                                .inner
                                .lock()
                                .as_mut()
                                .unwrap()
                                .destroy_semaphore(semaphore)
                                .map_supasim()?;
                        }
                    }
                    return Ok(());
                }
                SendSyncThreadEvent::AddSubmission(submission) => {
                    temp_submission_vec.push(Work::GpuSubmission(submission));
                    break;
                }
                SendSyncThreadEvent::SubmitBatchNow => (),
            }
        }
        let first_submission_time = Instant::now();
        let mut last_submission_time = first_submission_time;
        let mut last_was_submission = true;
        let mut final_cpu = None;
        loop {
            let now = Instant::now();
            if (now - first_submission_time) > MAX_SUBMISSION_WINDOW
                || (!last_was_submission && (now - last_submission_time) > SUBMISSION_WAIT_PERIOD)
            {
                break;
            }
            let max_wait = if last_was_submission {
                SUBMISSION_WAIT_PERIOD
            } else {
                SUBMISSION_WAIT_PERIOD - (now - last_submission_time)
            }
            .min(MAX_SUBMISSION_WINDOW - (now - first_submission_time));
            match logic.receiver.recv_timeout(max_wait) {
                Ok(SendSyncThreadEvent::WaitFinishAndShutdown) => {
                    for semaphore in semaphores {
                        unsafe {
                            logic
                                .instance
                                .inner
                                .lock()
                                .as_mut()
                                .unwrap()
                                .destroy_semaphore(semaphore)
                                .map_supasim()?;
                        }
                    }
                    return Ok(());
                }
                Ok(SendSyncThreadEvent::CpuWork(job)) => {
                    if !semaphore_signal {
                        final_cpu = Some(job);
                        break;
                    }
                    temp_submission_vec.push(Work::CpuWork(job));
                    last_submission_time = Instant::now();
                    last_was_submission = true;
                }
                Ok(SendSyncThreadEvent::AddFinishedJob(idx, job)) => {
                    last_was_submission = false;
                    if idx < next_submission_idx {
                        job.run(&logic.instance)?;
                    } else {
                        jobs.push((idx, job));
                    }
                }
                Ok(SendSyncThreadEvent::AddSubmission(submission)) => {
                    temp_submission_vec.push(Work::GpuSubmission(submission));
                    last_submission_time = Instant::now();
                    last_was_submission = true;
                }
                Ok(SendSyncThreadEvent::SubmitBatchNow) => break,
                Err(RecvTimeoutError::Timeout) => break,
                Err(RecvTimeoutError::Disconnected) => {
                    return Err(SupaSimError::Other(anyhow!(
                        "Main thread disconnected from sender"
                    )));
                }
            }
        }
        // Reverse sort so we can pop off the end
        jobs.sort_unstable_by_key(|a| u64::MAX - a.0);
        while let Some(job) = jobs.last() {
            if job.0 < next_submission_idx {
                let job = jobs.pop().unwrap();
                job.1.run(&logic.instance)?;
            } else {
                break;
            }
        }
        let mut recorders = Vec::new();
        // Setup the submits and collect all needed semaphores
        {
            let mut prev_was_cpu = false;
            // First submit is always guaranteed to be a GPU submission
            for item in temp_submission_vec.iter_mut() {
                match item {
                    Work::CpuWork(_) => {
                        prev_was_cpu = true;
                    }
                    Work::GpuSubmission(g) => {
                        used_semaphores.push(acquire_semaphore(&mut semaphores)?);
                        if prev_was_cpu {
                            used_semaphores.push(acquire_semaphore(&mut semaphores)?);
                        }
                        prev_was_cpu = false;
                        recorders.push(std::mem::take(&mut g.command_recorder).unwrap());
                    }
                }
            }
        }
        _num_submitted_so_far += temp_submission_vec.len();
        // Give the wait/signal semaphores to the submits
        {
            let mut prev_was_cpu = false;
            let mut recorders_iter = recorders.iter_mut();
            let mut semaphore_idx = 0;
            for item in temp_submission_vec.iter_mut() {
                match item {
                    Work::CpuWork(_) => prev_was_cpu = true,
                    Work::GpuSubmission(_) => {
                        submits.push(RecorderSubmitInfo {
                            command_recorder: recorders_iter.next().unwrap(),
                            wait_semaphore: if prev_was_cpu {
                                semaphore_idx += 1;

                                Some(&used_semaphores[semaphore_idx])
                            } else {
                                None
                            },
                            signal_semaphore: Some(&used_semaphores[semaphore_idx]),
                        });
                        semaphore_idx += 1;
                        prev_was_cpu = false;
                    }
                }
            }
        }
        // Submit
        unsafe {
            logic
                .instance
                .inner
                .lock()
                .as_mut()
                .unwrap()
                .submit_recorders(&mut submits)
                .map_supasim()?;
        }
        // Do the incremental waiting
        {
            let mut submit_idx = 0;
            let mut semaphore_idx = 0;
            for s in temp_submission_vec {
                match s {
                    Work::CpuWork(w) => {
                        w.run(&logic.instance)?;
                        next_submission_idx += 1;
                        logic.shared.0.lock().next_job = next_submission_idx;
                        logic.shared.1.notify_all();
                    }
                    Work::GpuSubmission(item) => {
                        if let Some(s) = submits[submit_idx].wait_semaphore {
                            unsafe {
                                s.signal().map_supasim()?;
                            }
                            semaphore_idx += 1;
                        }
                        unsafe {
                            used_semaphores[semaphore_idx].wait().map_supasim()?;
                        }
                        semaphore_idx += 1;
                        submit_idx += 1;

                        next_submission_idx += 1;
                        let mut lock = logic.shared.0.lock();
                        lock.next_job = next_submission_idx;
                        logic.shared.1.notify_all();
                        drop(lock);

                        for b in item.used_buffer_ranges {
                            if let Ok(buffer) = b.1.upgrade() {
                                if let Ok(b_inner) = buffer.inner() {
                                    b_inner.slice_tracker.release(b.0);
                                }
                            }
                        }
                        if !map_buffer_while_gpu_use {
                            for b in item.used_buffers {
                                if let Ok(buffer) = b.upgrade() {
                                    if let Ok(b_inner) = buffer.inner() {
                                        if b_inner.last_used == next_submission_idx {
                                            b_inner.slice_tracker.release_cpu();
                                        }
                                    }
                                }
                            }
                        }
                        for (bg, kernel) in item.bind_groups {
                            // TODO: fix issue here if kernel is already destroyed or is destroyed during this
                            match logic.instance.kernels.lock().get_mut(kernel).unwrap() {
                                SavedKernel::Loaded(k) => {
                                    let kernel = k.upgrade()?;
                                    let _k = kernel.inner_mut()?;
                                    let mut k = _k.inner.lock();
                                    unsafe {
                                        logic
                                            .instance
                                            .inner
                                            .lock()
                                            .as_mut()
                                            .unwrap()
                                            .destroy_bind_group(k.inner.as_mut().unwrap(), bg)
                                            .map_supasim()?;
                                    }
                                }
                                SavedKernel::WaitingForDestroy { inner } => unsafe {
                                    logic
                                        .instance
                                        .inner
                                        .lock()
                                        .as_mut()
                                        .unwrap()
                                        .destroy_bind_group(inner, bg)
                                        .map_supasim()?;
                                },
                            }
                        }
                    }
                }
                while let Some(last) = jobs.last() {
                    if last.0 < next_submission_idx {
                        jobs.pop().unwrap().1.run(&logic.instance)?;
                    } else {
                        break;
                    }
                }
            }
        }
        if let Some(final_cpu) = final_cpu {
            final_cpu.run(&logic.instance)?;
        }
        semaphores.append(&mut used_semaphores);
        logic
            .instance
            .hal_command_recorders
            .lock()
            .append(&mut recorders);
    }
}
