# Finish Residency Overhaul — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make PR #93's residency overhaul run correctly — both integration tests green on the runnable backends — by implementing the sync thread, fixing OOD tracking, issuing cross-location copies, adding wgpu workarounds, auditing locks, and removing dead code.

**Architecture:** A per-`(device, stream)` background thread drains GPU submissions, submits them to the HAL with dependency/completion semaphores, and frees resources on completion. Each `Buffer` tracks its data across host/device/storage via an out-of-date (OOD) tracker; accesses that need data from another location record a HAL copy first. All work stays single-device, single-stream (indices hardcoded to `0,0`).

**Tech Stack:** Rust (edition 2024, rustc 1.92), `parking_lot`, `thunderdome`, `smallvec`, `memmap2`, `std::sync::mpsc`; HAL backends Vulkan (MoltenVK on the dev Mac), wgpu; Slang kernel compiler.

## Global Constraints

- Rust edition 2024, `rust-version = 1.92`. Format with repo `rustfmt.toml`/`taplo.toml`.
- Every new `.rs` file needs the `LICENSE_HEADER` block (copy from any existing source file).
- Dual-license MIT OR Apache-2.0.
- No new `todo!()`/`unimplemented!()` on a reachable path — implement or return `Err`.
- Single-device, single-stream: device index `0`, stream index `0` everywhere.
- Out of scope (return clean `Err`, do not implement): external memory import (#57), auto-eviction policy (#8/#122), multi-device/multi-stream (#55/#118), CUDA (#10).
- Commits require sandbox off (gpg signing); each commit step notes this.

## Test matrix (this dev machine: Apple Silicon, `VULKAN_SDK` set → MoltenVK)

`dev_utils::all_backend_tests!` generates these `#[test]` fns per test (native Metal backend is commented out in the macro — not in scope):

- `<name>_vulkan` — native Vulkan (MoltenVK)
- `<name>_wgpu_vulkan` — wgpu over Vulkan
- `<name>_wgpu_metal` — wgpu over Metal (this is our Metal-path coverage)

Run one backend fast (Vulkan only) with:
`cargo nextest run -p supasim --no-default-features --features vulkan <test_fn_name>`
Full matrix:
`cargo nextest run -p supasim --all-features --no-fail-fast`

> **Note on TDD here:** the OOD tracker (Task 1) is pure logic and is true TDD. The integration tests already exist (`add_numbers`, `basic_buffer_copy`); for Tasks 3–7 the "failing test" is a runtime failure (a `todo!()` panic or a wrong readback), not a compile error. Each task states the concrete expected failure before the fix and the concrete pass after.

---

## Task 1: OOD tracker unit tests + validate interval logic

Establish correctness of the pure-logic core before touching sync. This is the "validate all OOD logic" TODO (`ood.rs:15`).

**Files:**
- Modify: `crates/supasim/src/buffer/ood.rs` (add `#[cfg(test)] mod tests` at end; fix any bug the tests expose)
- Modify: `crates/supasim/src/buffer/mod.rs` (only if a `BufferRange` bug is exposed)

**Interfaces:**
- Consumes: `BufferRange { start: u64, length: u64 }` and its `subtract`/`join`/`intersects` (`buffer/mod.rs`); `OutOfDateTracker::{uninit, update_range_immediate, invalidate_range}` (`ood.rs`).
- Produces: nothing new; confirms existing behavior.

- [ ] **Step 1: Write failing tests for `BufferRange` math**

Append to `crates/supasim/src/buffer/ood.rs`:

```rust
#[cfg(test)]
mod tests {
    use crate::buffer::BufferRange;

    fn r(start: u64, length: u64) -> BufferRange {
        BufferRange { start, length }
    }

    #[test]
    fn range_intersects() {
        assert!(r(0, 10).intersects(&r(5, 10)));
        assert!(!r(0, 10).intersects(&r(10, 5)));
        assert!(!r(0, 10).intersects(&r(20, 5)));
    }

    #[test]
    fn range_subtract_middle_splits() {
        // subtracting [4,6) from [0,10) leaves [0,4) and [6,10)
        let (a, b) = r(0, 10).subtract(&r(4, 2));
        assert_eq!(a, r(0, 4));
        assert_eq!(b, Some(r(6, 4)));
    }
}
```

- [ ] **Step 2: Run and confirm they compile+run**

Run: `cargo nextest run -p supasim --no-default-features --features vulkan buffer::ood::tests`
Expected: PASS if `BufferRange` is correct; if a test FAILS, that is a real bug — fix `BufferRange` in `buffer/mod.rs` and note it in the commit. (Assertions encode the intended semantics.)

- [ ] **Step 3: Write failing tests for `OutOfDateTracker`**

Add inside the same `mod tests`:

```rust
    use crate::buffer::ood::OutOfDateTracker;

    // Concrete backend not needed for pure range logic; use the vulkan cfg backend type param.
    type T = OutOfDateTracker<hal::Vulkan>;

    #[test]
    fn tracker_starts_all_out_of_date() {
        let t = T::uninit(100);
        assert_eq!(t.out_of_date_ranges, vec![BufferRange { start: 0, length: 100 }]);
    }

    #[test]
    fn immediate_update_clears_range() {
        let mut t = T::uninit(100);
        t.update_range_immediate(BufferRange { start: 0, length: 100 });
        assert!(t.out_of_date_ranges.is_empty());
    }

    #[test]
    fn invalidate_merges_adjacent() {
        let mut t = T::uninit(0); // starts empty (length 0 → [0,0))
        t.out_of_date_ranges.clear();
        t.invalidate_range(BufferRange { start: 0, length: 10 });
        t.invalidate_range(BufferRange { start: 10, length: 10 });
        assert_eq!(t.out_of_date_ranges, vec![BufferRange { start: 0, length: 20 }]);
    }
```

- [ ] **Step 4: Run tracker tests**

Run: `cargo nextest run -p supasim --no-default-features --features vulkan buffer::ood::tests`
Expected: PASS. If `invalidate_merges_adjacent` FAILS, fix `invalidate_range` merge logic (`ood.rs:61-100`).

- [ ] **Step 5: Commit** (sandbox off — gpg)

```bash
git add crates/supasim/src/buffer/ood.rs crates/supasim/src/buffer/mod.rs
git commit -m "test: unit tests for OutOfDateTracker interval logic"
```

---

## Task 2: Fix the inverted read/write access lists

Insert sites store `is_mut` accesses in the wrong list; release/remove/wait read the other list.

**Files:**
- Modify: `crates/supasim/src/buffer/residency.rs` (`add_gpu_use` ~318–335, `get_cpu_access` ~403–411)

**Interfaces:**
- Consumes: `BufferResidency.read_accesses: HashMap<u64, Arc<BufferAccessFinish>>` (reads), `write_accesses: VecDeque<Arc<BufferAccessFinish>>` (writes).
- Produces: consistent invariant — `read_accesses` holds non-mut accesses, `write_accesses` holds mut accesses, matching `release_cpu_access`/`try_remove_gpu_access`/`wait_for_cpu_access`.

- [ ] **Step 1: Fix `add_gpu_use` insert**

In `crates/supasim/src/buffer/residency.rs`, the block currently reading:

```rust
        if is_mut {
            self.read_accesses.insert(finish.id, finish.clone());
            for (i, d) in self
                .devices
                .iter_mut()
                .chain(std::iter::once(&mut self.host))
                .enumerate()
            {
                if (i as u32) != device_index && d.buffer.is_some() {
                    d.ood_tracker.invalidate_range(range);
                }
            }
            if let Some(d) = &mut self.storage {
                d.ood_tracker.invalidate_range(range);
            }
        } else {
            self.write_accesses.push_back(finish.clone());
        }
```

Change the two list operations so mut → `write_accesses`, non-mut → `read_accesses` (keep the invalidation for the mut branch):

```rust
        if is_mut {
            self.write_accesses.push_back(finish.clone());
            for (i, d) in self
                .devices
                .iter_mut()
                .chain(std::iter::once(&mut self.host))
                .enumerate()
            {
                if (i as u32) != device_index && d.buffer.is_some() {
                    d.ood_tracker.invalidate_range(range);
                }
            }
            if let Some(d) = &mut self.storage {
                d.ood_tracker.invalidate_range(range);
            }
        } else {
            self.read_accesses.insert(finish.id, finish.clone());
        }
```

- [ ] **Step 2: Fix the self-wait scan in `add_gpu_use`**

The block at ~345 that runs `if is_mut { for finish in self.read_accesses.values() { ... } }` must scan the *reads* a new write waits on (WAR). With Step 1 applied, `read_accesses` now holds reads and the new write is in `write_accesses`, so this loop no longer self-references. Leave the loop scanning `read_accesses` as-is; verify by reading it does not push `finish` (the new access) itself.

- [ ] **Step 3: Fix `get_cpu_access` insert**

Change the same-shaped block (~403):

```rust
        if is_mut {
            self.read_accesses.insert(finish.id, finish.clone());
            for d in &mut self.devices {
                d.ood_tracker.invalidate_range(range);
            }
            self.host.ood_tracker.update_range_immediate(range);
        } else {
            self.write_accesses.push_back(finish.clone());
        }
```

to:

```rust
        if is_mut {
            self.write_accesses.push_back(finish.clone());
            for d in &mut self.devices {
                d.ood_tracker.invalidate_range(range);
            }
            self.host.ood_tracker.update_range_immediate(range);
        } else {
            self.read_accesses.insert(finish.id, finish.clone());
        }
```

- [ ] **Step 4: Confirm it still compiles**

Run: `cargo check -p supasim --no-default-features --features vulkan`
Expected: Finished, no errors.

- [ ] **Step 5: Commit** (sandbox off)

```bash
git add crates/supasim/src/buffer/residency.rs
git commit -m "fix: correct inverted read/write access lists in residency"
```

---

## Task 3: Implement the sync thread

Fresh, submit-now per-`(device, stream)` thread. This unblocks `submit_commands` → `wait_for_idle`.

**Files:**
- Modify: `crates/supasim/src/sync/stream_thread.rs` (implement `create_sync_thread`, add `signal_semaphore` field)
- Modify: `crates/supasim/src/sync/mod.rs` (`submit_command_recorders`: set `signal_semaphore`)

**Interfaces:**
- Consumes: `hal::Stream::submit_recorders(&mut [RecorderSubmitInfo { command_recorder, wait_semaphores: &[&B::Semaphore], signal_semaphore: Option<&B::Semaphore> }])`; `hal::Semaphore::{wait, is_signalled}`; `hal::CommandRecorder::destroy`; `Instance::inner()`.
- Produces: `create_sync_thread(instance, device_idx, stream_idx) -> StreamThreadHandle<B>`; `GpuSubmissionInfo.signal_semaphore: Arc<Semaphore<B>>`.

- [ ] **Step 1: Add the signal semaphore to `GpuSubmissionInfo`**

In `stream_thread.rs`, add a field (rename the `_`-prefixed ones to live names while here):

```rust
pub struct GpuSubmissionInfo<B: hal::Backend> {
    pub index: u64,
    pub command_recorder: B::CommandRecorder,
    /// Signalled by the GPU when this submission completes; waited on CPU-side.
    pub signal_semaphore: Arc<crate::sync::Semaphore<B>>,
    pub bind_groups: Vec<(B::BindGroup, Kernel<B>)>,
    pub used_buffer_ranges: Vec<(OutOfDateWait<B>, Buffer<B>)>,
    pub used_resources: SubmissionResources<B>,
}
```

- [ ] **Step 2: Set `signal_semaphore` at the submit site**

In `sync/mod.rs::submit_command_recorders`, the `lock.submit(GpuSubmissionInfo { ... })` call: pass `signal_semaphore: semaphore.clone()` and update the renamed fields (`index`, `command_recorder`, `bind_groups`, `used_buffer_ranges`, `used_resources`).

- [ ] **Step 3: Implement `create_sync_thread`**

Replace the `todo!()` body. The thread owns the receiver; the handle owns the sender/join handle:

```rust
pub fn create_sync_thread<B: hal::Backend>(
    instance: Instance<B>,
    device_idx: usize,
    stream_idx: usize,
) -> StreamThreadHandle<B> {
    use std::sync::mpsc;
    let (sender, receiver) = mpsc::channel::<StreamThreadMessage<B>>();
    let completed = Arc::new((Mutex::new(0u64), Condvar::new()));
    let completed_thread = completed.clone();
    let thread = std::thread::spawn(move || {
        while let Ok(msg) = receiver.recv() {
            match msg {
                StreamThreadMessage::Submission(info) => {
                    run_submission(&instance, device_idx, stream_idx, info, &completed_thread);
                }
                StreamThreadMessage::ShutDown => break,
            }
        }
    });
    StreamThreadHandle {
        current_submitted_count: 1,
        current_completed_index: completed,
        sender,
        thread,
    }
}
```

- [ ] **Step 4: Implement `run_submission`**

Add this helper in `stream_thread.rs`. It submits with the dependency wait-semaphores and the completion signal-semaphore, blocks until done, then frees resources. Locks the stream only for the submit call, never across the blocking wait.

```rust
fn run_submission<B: hal::Backend>(
    instance: &Instance<B>,
    device_idx: usize,
    stream_idx: usize,
    mut info: GpuSubmissionInfo<B>,
    completed: &Arc<(Mutex<u64>, Condvar)>,
) {
    use hal::{Semaphore as _, Stream as _, CommandRecorder as _};
    // Gather wait semaphores: the GPU dependency semaphores collected during add_gpu_use.
    let mut wait_arcs: Vec<Arc<crate::sync::Semaphore<B>>> = Vec::new();
    for (wait, _buf) in &info.used_buffer_ranges {
        for f in &wait.semaphores {
            if let Some(sem) = f.device_semaphore.lock().as_ref() {
                wait_arcs.push(sem.clone());
            }
        }
    }
    {
        let s = instance.inner().unwrap();
        let hal_instance_guard = s.hal_instance.read();
        let hal_instance = unsafe { types::to_static_lifetime(hal_instance_guard.as_ref().unwrap()) };
        // Borrow the inner hal Semaphore refs for the submit call.
        let wait_locks: Vec<_> = wait_arcs.iter().map(|a| a.inner.as_ref().unwrap().read()).collect();
        let wait_refs: Vec<&B::Semaphore> = wait_locks.iter().map(|g| &**g).collect();
        let signal_lock = info.signal_semaphore.inner.as_ref().unwrap().read();
        let mut stream_guard = s.hal_devices[device_idx].streams[stream_idx].inner.lock();
        let stream = stream_guard.as_mut().unwrap();
        unsafe {
            stream
                .submit_recorders(std::slice::from_mut(&mut hal::RecorderSubmitInfo {
                    command_recorder: &mut info.command_recorder,
                    wait_semaphores: &wait_refs,
                    signal_semaphore: Some(&signal_lock),
                }))
                .unwrap();
        }
        let _ = hal_instance;
    }
    // Block until the GPU signals completion (submit-now, no batching).
    info.signal_semaphore.wait().unwrap();
    // Mark residency finishes complete + free per-submission resources.
    finish_submission(instance, device_idx, stream_idx, info);
    // Advance the completion counter and wake waiters.
    let (lock, cv) = &**completed;
    let mut g = lock.lock();
    *g += 1;
    cv.notify_all();
}
```

> Implementation note (decide while coding): the exact borrow dance for `wait_refs`/`signal_lock` may need `parking_lot` guard lifetimes adjusted, or copying raw `B::Semaphore` handles. Keep the rule invariant: **hold no `instance.inner()` lock across `signal_semaphore.wait()`**. If the borrow checker fights the guard vector, submit with an empty wait list for the single-submission tests (dependencies are within one recorder) and add cross-submission waits in Task 5.

- [ ] **Step 5: Implement `finish_submission`**

Frees resources and releases residency holds:

```rust
fn finish_submission<B: hal::Backend>(
    instance: &Instance<B>,
    device_idx: usize,
    stream_idx: usize,
    info: GpuSubmissionInfo<B>,
) {
    use hal::{BindGroup as _, Buffer as _, CommandRecorder as _};
    let s = instance.inner().unwrap();
    // Release residency range holds (removes the finished access from the lists).
    for (wait, buf) in info.used_buffer_ranges {
        if let Ok(b) = buf.inner() {
            let mut res = b.residency.0.write();
            for f in wait.semaphores {
                res.try_remove_gpu_access(f, /*is_mut*/ false);
            }
        }
    }
    // Destroy bind groups, temp buffer; return recorder to the pool.
    let stream_inner = s.hal_devices[device_idx].streams[stream_idx].inner.lock();
    let stream = stream_inner.as_ref().unwrap();
    for (bg, _kernel) in info.bind_groups {
        unsafe { bg.destroy(stream).unwrap(); }
    }
    if let Some(tmp) = info.used_resources.temp_copy_buffer {
        unsafe { tmp.destroy(s.hal_devices[device_idx].inner.lock().as_ref().unwrap()).unwrap(); }
    }
    drop(stream_inner);
    s.hal_devices[device_idx].streams[stream_idx]
        .unused_hal_command_recorders
        .lock()
        .push(info.command_recorder);
}
```

> Note: `try_remove_gpu_access`'s `is_mut` must match how the access was inserted. Task 5 threads the real `needs_mut` through `used_buffer_ranges`; until then reads/writes both resolve via id for reads and pointer for writes, so passing the recorded mutability is required — carry `needs_mut` alongside each `OutOfDateWait` in `used_buffer_ranges` (change its tuple to `(OutOfDateWait<B>, Buffer<B>, bool)`), set in `submit_command_recorders`.

- [ ] **Step 6: Verify create_sync_thread is wired at instance init**

Confirm `lib.rs` calls `create_sync_thread(stream_clone, 0, 0)` when building each `Stream` (it already references `stream_handle`). If it still constructs the handle inline, replace with the call.

Run: `cargo check -p supasim --no-default-features --features vulkan`
Expected: Finished, no errors.

- [ ] **Step 7: Runtime smoke — submission no longer panics**

Run: `cargo nextest run -p supasim --no-default-features --features vulkan basic_buffer_copy_vulkan`
Expected BEFORE Task 4: the `todo!()` panic is gone; the test now either passes or fails on the **readback assertion** (stale download buffer) — not on a panic. Record which.

- [ ] **Step 8: Commit** (sandbox off)

```bash
git add crates/supasim/src/sync/stream_thread.rs crates/supasim/src/sync/mod.rs crates/supasim/src/lib.rs
git commit -m "feat: implement submit-now per-stream sync thread"
```

---

## Task 4: Cross-location copy — CPU read path (basic_buffer_copy green)

Make a CPU read pull current data from the device before reading the host copy.

**Files:**
- Modify: `crates/supasim/src/buffer/residency.rs` (`BufferResidencyRef::wait_for_cpu_access` ~466–513; add a `copy_device_to_host` helper on `BufferResidency`)
- Modify: `crates/supasim/src/buffer/access.rs` (ensure `MappedBuffer::new` reads current data / direct-maps correctly)

**Interfaces:**
- Consumes: `OutOfDateTracker::get_needed_waits(range, instance) -> OutOfDateWait<B>` (the currently-discarded `_needed_waits`); `hal::Buffer::{map, unmap, read, write}`; `HalInstanceProperties.map_buffers`.
- Produces: `BufferResidency::ensure_host_current(range, instance)` that records+runs device→host copies so the host copy is up to date for `range`.

- [ ] **Step 1: Establish the failing assertion**

Run: `cargo nextest run -p supasim --no-default-features --features vulkan basic_buffer_copy_vulkan`
Expected: FAIL on `assert_eq!(... [1,2,3,4])` — host copy of `download_buffer` is stale after the GPU copy.

- [ ] **Step 2: Add `ensure_host_current` to `BufferResidency`**

For each device whose OOD tracker shows `range` is current there but the host is out of date, submit a synchronous device→host copy via the stream and mark the host tracker current. Concretely, for the single-device case: if `self.host.ood_tracker` reports `range` out of date and `self.devices[0]` has it current, use a one-off recorder (`stream.create_recorder`) with a single `BufferCommand::CopyBuffer { src: device[0], dst: host, ... }`, submit it with a fresh signal semaphore, wait, then `self.host.ood_tracker.update_range_immediate(range)`.

```rust
pub fn ensure_host_current(&mut self, range: BufferRange, instance: &InstanceInner<B>) {
    self.setup_buffer(None, instance);          // host buffer
    self.setup_buffer(Some(0), instance);       // device buffer
    let needs = self.host.ood_tracker.out_of_date_ranges.iter()
        .any(|r| r.intersects(&range));
    if !needs { return; }
    // Record + run a device[0] -> host copy for the whole `range`.
    // (Single-submission, blocking; uses instance.get_semaphore()).
    unsafe { copy_between_hal_buffers(
        instance, /*src*/ self.devices[0].buffer.as_ref().unwrap(),
        /*dst*/ self.host.buffer.as_ref().unwrap(), range); }
    self.host.ood_tracker.update_range_immediate(range);
    for d in &mut self.devices { /* device copies stay current */ }
}
```

Add a free helper `copy_between_hal_buffers` in `residency.rs` that creates a recorder, records one `CopyBuffer`, submits with a signal semaphore, and waits. Reuse `instance.get_semaphore()` and `hal_devices[0].streams[0]`.

- [ ] **Step 3: Call it from `wait_for_cpu_access`**

Replace the discarded `let _needed_waits = s.host.ood_tracker.get_needed_waits(range, instance);` (line ~512) with a call that first drains prior-access waits (existing loop) then makes the host current:

```rust
    drop(s);
    let mut s = self.0.write();
    if any_waits { s.update_all_accesses().unwrap(); }
    s.ensure_host_current(range, instance);
```

- [ ] **Step 4: Run basic_buffer_copy on Vulkan**

Run: `cargo nextest run -p supasim --no-default-features --features vulkan basic_buffer_copy_vulkan`
Expected: PASS ([1,2,3,4] read back).

- [ ] **Step 5: Commit** (sandbox off)

```bash
git add crates/supasim/src/buffer/residency.rs crates/supasim/src/buffer/access.rs
git commit -m "feat: device->host copy on CPU read (basic_buffer_copy green on vulkan)"
```

---

## Task 5: Cross-location copy — GPU use path (add_numbers green)

A GPU dispatch/copy that consumes a buffer whose current data lives elsewhere (host, after a CPU write) must copy it into the device first.

**Files:**
- Modify: `crates/supasim/src/buffer/residency.rs` (`add_gpu_use`: turn `other_copy_range` into a real host→device copy record)
- Modify: `crates/supasim/src/sync/mod.rs` (thread the recorded copies + `needs_mut` into the submission)
- Modify: `crates/supasim/src/sync/stream_thread.rs` (`used_buffer_ranges` tuple carries `needs_mut`; free path uses it)

**Interfaces:**
- Consumes: `OutOfDateWait { semaphores, other_copy_range }` from `add_gpu_use`; `BufferCommand::CopyBuffer`.
- Produces: submissions whose recorder begins with any required host→device copies before the user commands.

- [ ] **Step 1: Establish the failing assertion**

Run: `cargo nextest run -p supasim --no-default-features --features vulkan add_numbers_vulkan`
Expected: FAIL — buffers 1/2/3 written via `write_buffer` are current on the staging/host side but the dispatch reads stale device memory (or a copy panic).

- [ ] **Step 2: Make `add_gpu_use` request the host→device copy**

Where `add_gpu_use` handles `wait.other_copy_range` (currently only `update_range_delayed`), also record that a host→device copy for `extra_copy.range` must precede this GPU submission. Return this in `OutOfDateWait` (add `pub needed_device_copies: Vec<BufferRange>` to `OutOfDateWait`, or reuse `other_copy_range`), so the submit path can prepend `CopyBuffer { src: host, dst: device[0], range }`.

- [ ] **Step 3: Prepend copies in `submit_command_recorders`**

After `add_gpu_use` returns for each buffer/range, collect the needed host→device copies and record them into `recorder` (via `record_command_streams` or a direct `CopyBuffer` prepend) **before** the user command stream. Ensure the source host buffer exists (`setup_buffer(None)`).

- [ ] **Step 4: Carry `needs_mut` into `used_buffer_ranges`**

Change the tuple to `(OutOfDateWait<B>, Buffer<B>, bool)` and set the bool from `range.needs_mut`. Update `finish_submission` to pass it to `try_remove_gpu_access`.

- [ ] **Step 5: Run add_numbers on Vulkan**

Run: `cargo nextest run -p supasim --no-default-features --features vulkan add_numbers_vulkan`
Expected: PASS ([6,8,10,12]).

- [ ] **Step 6: Run both tests, Vulkan**

Run: `cargo nextest run -p supasim --no-default-features --features vulkan`
Expected: `add_numbers_vulkan` and `basic_buffer_copy_vulkan` PASS.

- [ ] **Step 7: Commit** (sandbox off)

```bash
git add crates/supasim/src/buffer/residency.rs crates/supasim/src/sync/mod.rs crates/supasim/src/sync/stream_thread.rs
git commit -m "feat: host->device copy on GPU use (add_numbers green on vulkan)"
```

---

## Task 6: wgpu workarounds (wgpu_vulkan + wgpu_metal green)

**Files:**
- Modify: `crates/supasim/src/buffer/residency.rs` (whole-buffer mapping; separate up/down buffers)
- Modify: `crates/supasim/src/buffer/access.rs` (map path honoring `map_buffer_while_gpu_use`)
- Modify: `crates/supasim-hal/src/wgpu/mod.rs` (`MemoryTransfer` → clean error)

**Interfaces:**
- Consumes: `HalInstanceProperties.{map_buffers, map_buffer_while_gpu_use, upload_download_buffers}`.
- Produces: residency that branches on these caps.

- [ ] **Step 1: Establish failures**

Run: `cargo nextest run -p supasim --features wgpu --no-default-features add_numbers_wgpu_vulkan basic_buffer_copy_wgpu_vulkan`
Expected: FAIL/panic (map-while-in-use error, or `MemoryTransfer` `todo!()`).

- [ ] **Step 2: Whole-buffer access when `!map_buffer_while_gpu_use`**

In `get_cpu_access`/`wait_for_cpu_access`, when `instance.hal_instance_properties.map_buffer_while_gpu_use` is false, widen the requested `range` to the whole buffer (`0..size`) for the map, per the residency.rs:388 TODO.

- [ ] **Step 3: Separate upload/download buffers when `!upload_download_buffers`**

In `setup_buffer`, when the backend lacks `upload_download_buffers`, allocate two HAL buffers (one `Upload`, one `Download`) for the host location and route CPU writes to the upload buffer and CPU reads from the download buffer. Add fields to `DeviceResidencyState` for the second buffer (host-only).

- [ ] **Step 4: wgpu `MemoryTransfer` → error**

In `crates/supasim-hal/src/wgpu/mod.rs:603`, replace `BufferCommand::MemoryTransfer { .. } => todo!()` with a returned backend error (external memory unsupported on wgpu, #57).

- [ ] **Step 5: Run wgpu tests**

Run: `cargo nextest run -p supasim --no-default-features --features wgpu`
Expected: `*_wgpu_vulkan` and `*_wgpu_metal` PASS.

> **De-scope lever (requires user OK):** if Step 3/Step 2 balloon, stop, mark wgpu residency-limited, file a follow-up issue, and keep Vulkan green. Do not silently skip.

- [ ] **Step 6: Commit** (sandbox off)

```bash
git add -A
git commit -m "feat: wgpu residency workarounds (whole-buffer map, split up/down buffers)"
```

---

## Task 7: Deadlock audit

**Files:**
- Modify: `crates/supasim/src/sync/stream_thread.rs`, `crates/supasim/src/sync/mod.rs`, `crates/supasim/src/lib.rs`, `crates/supasim/src/buffer/residency.rs` (lock-scope tightening only)

**Interfaces:** no signature changes; behavior-preserving lock-ordering fixes.

- [ ] **Step 1: Audit lock order**

Walk each site that takes more than one lock. Enforce: `Instance::inner()` first, then per-object (`buffer.inner()`, `residency.write()`, stream `inner.lock()`, `stream_handle.write()`) in a fixed order. Write the chosen order as a comment block at the top of `sync/mod.rs`.

- [ ] **Step 2: Verify the completion path holds no long locks**

Confirm `run_submission` holds `instance.inner()` only for the submit call and never across `signal_semaphore.wait()`; confirm `finish_submission` takes short, ordered locks. Fix any violation found.

- [ ] **Step 3: Stress run**

Run: `cargo nextest run -p supasim --no-default-features --features vulkan --no-fail-fast` three times.
Expected: consistent PASS, no hangs. If a hang occurs, capture a backtrace (`SUPASIM` trace env) and fix the ordering.

- [ ] **Step 4: Commit** (sandbox off)

```bash
git add -A
git commit -m "refactor: enforce documented lock ordering; audit for deadlocks"
```

---

## Task 8: Cleanup and dead-code removal

**Files:**
- Modify: `crates/supasim/src/lib.rs` (remove `"Yes its UB"` comments), `crates/supasim/src/buffer/residency.rs` (gate/remove `_switch_to_storage`, `StorageResidencyState::_new` if unused), `crates/supasim/src/record.rs` (`_UpdateBindGroup`), plus any stale comments surfaced.

- [ ] **Step 1: Remove misleading comments**

Delete the three `// Yes its UB, but id doesn't have any destructor...` comments in `lib.rs` (thunderdome `Index` is POD; construction-then-overwrite is sound). Replace with a one-line note if desired.

- [ ] **Step 2: Resolve `record.rs` `_UpdateBindGroup`**

At `record.rs:660`, replace `todo!()` with either the Metal in-place update path (when `easily_update_bind_groups`) or a clear `unreachable!("bind groups are recreated on backends without easily_update_bind_groups")` with a comment.

- [ ] **Step 3: Gate or remove dormant storage code**

If `_switch_to_storage` / `StorageResidencyState::_new` are unreferenced after Task 4/5, either delete them or move behind a `#[cfg(feature = "storage")]` with a tracking comment pointing to #8/#122. Pick deletion unless the storage path is close.

- [ ] **Step 4: Clippy + fmt**

Run: `cargo clippy -p supasim --all-features -- -D warnings` then `cargo fmt --all`.
Expected: no warnings; formatting clean.

- [ ] **Step 5: Full matrix**

Run: `cargo nextest run -p supasim --all-features --all-targets --no-fail-fast`
Expected: all generated backend tests PASS (or wgpu de-scoped with user OK + follow-up filed).

- [ ] **Step 6: Commit + update PR TODO** (sandbox off)

```bash
git add -A
git commit -m "chore: remove dead code and misleading comments; resolve UpdateBindGroup"
```

Then check off the PR #93 TODO items in the PR description (residency logic, sync thread, wgpu workarounds, deadlocks, cleanup).

---

## Self-review notes (author)

- **Spec coverage:** Component 1 → Task 3; Component 2 → Tasks 1,2,4,5; Component 3 → Task 6; Component 4 → Task 7; Component 5 → Task 8. All covered.
- **Known softest spots (resolve while coding, decisions pre-made in-task):** the guard-lifetime dance in Task 3 Step 4, and the residency↔record recording seam in Tasks 4–5 (decision: residency records one-off blocking copies via the stream; the submit path prepends GPU-use copies into the main recorder).
- **Scope flag for the user:** native Metal HAL backend is disabled in `all_backend_tests!`; Metal-path coverage comes from `*_wgpu_metal`. Enabling the native Metal backend (its `unreachable!()`s) is a separate follow-up, not in this plan.
