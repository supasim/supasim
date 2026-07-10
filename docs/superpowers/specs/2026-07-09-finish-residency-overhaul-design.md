# Design: Finish the residency-overhaul PR (#93)

Date: 2026-07-09
Branch: `residency-overhaul`
Status: approved, pre-implementation

## Background

PR #93 ("Residency overhaul") is a large rewrite of the `supasim` frontend. It splits the old
1265-line `sync.rs` into `sync/{mod,stream_thread}.rs`, adds `buffer/{mod,access,ood,residency}.rs`
and `record.rs`, and splits the HAL Vulkan backend into `vulkan/{mod,init,command_recorder}.rs`.
It introduces **multi-location buffer residency**: every `Buffer` can have a copy of its data on
each device, on the host, and (eventually) on disk, with an **out-of-date (OOD) tracker** per
location deciding which byte ranges are current and what must be copied/waited on before an access.

The branch **compiles clean** (`cargo check -p supasim --features vulkan`, exit 0). It fails at
**runtime**: the per-stream sync thread is an unimplemented `todo!()`, and the residency layer
never issues the cross-location copies it schedules. This spec finishes the PR.

## Goals

- Both integration tests pass end-to-end on **Vulkan, Metal, and wgpu**:
  - `crates/supasim/src/tests/basic_buffer_copy.rs` — upload→gpu→download copy, CPU readback.
  - `crates/supasim/src/tests/add_numbers.rs` — CPU writes, Slang kernel dispatch, CPU readback.
  Both run across all backends via `dev_utils::all_backend_tests!`.
- Address every item on the PR TODO list: finish residency logic; sync-thread implementation;
  finish wgpu workarounds; check for deadlocks; remove unused code & cleanup.
- Replace `todo!()` on any reachable path with either a real implementation or a clean `Err`.

## Non-goals (explicitly out of scope — remain follow-up issues)

- External memory import (`import_buffer` / `import_semaphore`, wgpu `MemoryTransfer`) — issue #57.
- Automatic GPU→CPU→disk eviction *policy* — issues #8 / #122. We keep the host↔device↔storage
  copy *machinery* but add no eviction trigger. `_switch_to_storage` stays dormant.
- Multi-device and multi-stream scheduling — issues #55 / #118 / #111. Device/stream indices stay
  hardcoded to `(0, 0)`; the sync thread is written per-`(device, stream)` so it extends later.
- CUDA backend (#10), pipeline cache (#139), command-recording optimization / batching (#117),
  bindless alternatives (#125).

## Current state (what exists vs. what is missing)

Works today (verified by reading the code):
- Frontend object model: `Arc<RwLock<Inner>>` handles + weak-ref `thunderdome` arenas.
- `submit_command_recorders` (`sync/mod.rs`): assembles streams, allocates a pooled semaphore,
  registers GPU uses on each buffer's residency (`add_gpu_use`), records commands, and sends a
  `GpuSubmissionInfo` to the stream thread. Returns a `WaitHandle`.
- `record::assemble_streams` / `record_command_streams`: lower commands to HAL, partition into
  parallel streams, dedup bind groups, build one temp upload buffer.
- OOD interval math (`ood.rs`): `update_range_immediate`, `invalidate_range`, `get_needed_waits`,
  `check_all_current_copies`.

Missing / broken (the work):
1. `sync/stream_thread.rs::create_sync_thread` is `todo!()` — nothing drains submissions, so
   `submit_commands` → `wait_for_idle` / `access` never completes.
2. Cross-location copies are scheduled but never issued: `wait_for_cpu_access` (residency.rs:512)
   computes `_needed_waits` and discards it; `add_gpu_use` records a delayed range but emits no
   copy command. → a CPU read of a GPU-written buffer reads stale host memory.
3. Inverted read/write access lists: insert sites (`add_gpu_use` ~318, `get_cpu_access` ~403) use
   `is_mut → read_accesses`, but release/remove/wait (`release_cpu_access` ~434,
   `try_remove_gpu_access` ~447, `wait_for_cpu_access` ~478) use `is_mut → write_accesses`. Insert
   is flipped → an op waits on itself and `release_cpu_access`'s `idx.unwrap()` can panic.
4. wgpu residency workarounds unimplemented: whole-buffer mapping when `map_buffer_while_gpu_use`
   is false; separate buffers when `upload_download_buffers` is false; wgpu `MemoryTransfer` is
   `todo!()`.
5. No deadlock discipline: lock order documented but "not currently followed" (#120).
6. Dead code / misleading comments (`"Yes its UB"`, unused `_`-fields, `record.rs` `_UpdateBindGroup`).

## Design

Phased and correctness-first. Each phase ends compiling with the relevant tests green, giving
review checkpoints. Backend order: Vulkan → Metal → wgpu.

### Component 1 — Sync thread (`sync/stream_thread.rs`, `sync/mod.rs`, `lib.rs`)

Implement `create_sync_thread` as a fresh, submit-now (no batching window) per-`(device, stream)`
thread.

- **Thread infra**: spawn a `std::thread`, own the `mpsc::Receiver<StreamThreadMessage<B>>`,
  populate `StreamThreadHandle { current_submitted_count, current_completed_index, sender, thread }`.
- **Signal semaphore gap**: `GpuSubmissionInfo` does not currently carry the submission's own
  completion semaphore (it lives only inside the per-range `BufferAccessFinish`es and the returned
  `WaitHandle`). Add `signal_semaphore: Arc<Semaphore<B>>` to `GpuSubmissionInfo` and set it in
  `submit_command_recorders`.
- **Submission handling**:
  1. Collect wait semaphores from every `OutOfDateWait.semaphores` in `_used_buffer_ranges`, plus
     any copy dependencies produced by Component 2.
  2. Submit via the HAL `Stream::submit_recorders(recorder, waits, signal = signal_semaphore)`.
  3. Block until the signal semaphore fires (submit-now).
  4. On completion: set each `BufferAccessFinish.is_complete = true` and notify its condvar /
     signal its semaphore; increment `current_completed_index` and notify its condvar; destroy the
     bind groups (`_bind_groups`); destroy `temp_copy_buffer`; return the command recorder to the
     stream's `unused_hal_command_recorders` pool; release residency holds via
     `try_remove_gpu_access`.
- **ShutDown**: drain outstanding submissions to completion, then return (joined by
  `InstanceInner::destroy`).
- De-`_`-prefix the fields that become live (`index`, `command_recorder`, `bind_groups`,
  `used_buffer_ranges`, `used_resources`).

Interface: `StreamThreadHandle::submit` and `wait_for_submission` already exist and are the only
things `submit_command_recorders` / `wait_for_idle` depend on — no frontend API change.

### Component 2 — OOD correctness (`buffer/residency.rs`, `buffer/ood.rs`)

- **Fix the inverted insert**: in `add_gpu_use` and `get_cpu_access`, insert `is_mut → write_accesses`
  and `!is_mut → read_accesses`, matching the release/remove/wait sites. This removes the
  self-wait (an op adding itself to the list it then scans) and the `release_cpu_access` panic.
- **Issue cross-location copies** (the core "finish residency logic"): give the residency layer a
  way to record HAL copies between locations. When `get_needed_waits` returns an out-of-date range
  that a current copy elsewhere can satisfy:
  - GPU path (`add_gpu_use`): before the dependent GPU work, record a `CopyBuffer`
    host→device (or device→device) into the submission's command stream and bind it to the
    `BufferAccessFinish` id/semaphore that `get_needed_waits` created.
  - CPU path (`wait_for_cpu_access`): consume the currently-discarded `_needed_waits`. If the
    backend supports direct mapping (`map_buffers`) and mapping-while-in-use rules allow it, map
    the device buffer; otherwise submit a device→host copy, wait on it, then read the host copy.
  - The exact recording seam between `residency` and `record`/the stream thread is the riskiest
    detail; resolve it during implementation (likely: residency returns copy descriptors that the
    submit path / stream thread records, keeping residency free of HAL command-buffer specifics).
- **Tests**: add unit tests for `OutOfDateTracker` (interval subtract/join/invalidate and
  `get_needed_waits` copy scheduling) — the module is self-labelled "validate all OOD logic".

### Component 3 — wgpu workarounds (`hal/src/wgpu`, `buffer/residency.rs`, `buffer/access.rs`)

- `map_buffer_while_gpu_use == false`: when a CPU map is requested, request access to the whole
  buffer (not just the slice), so wgpu's "no map while in use" rule is respected.
- `upload_download_buffers == false`: represent a location with separate upload and download HAL
  buffers instead of one `UploadDownload` buffer; `setup_buffer` and the copy paths branch on the
  capability.
- wgpu `MemoryTransfer` / external memory: return a clean `SupaSimError`/`Err` (external import is
  out of scope, #57) rather than `todo!()`.

### Component 4 — Deadlock check

- Enforce the documented order: lock `Instance` first, then per-object resources in a fixed order.
- Verify the sync thread never holds `instance.inner()` while blocked on the GPU, and that the
  submit path's read-lock + `residency.write()` cannot cycle against the thread's completion path
  (which also takes `instance.inner()` to signal semaphores). Adjust lock scopes so completion-side
  locking is short and ordered. Add cheap `debug_assert`s where useful.

### Component 5 — Cleanup

- Remove or feature-gate dead code (`_switch_to_storage`, `StorageResidencyState::_new` if still
  unreferenced after Component 2).
- Delete the misleading `"Yes its UB"` comments (constructing with `Index::DANGLING` then
  overwriting is sound — `thunderdome::Index` is POD).
- Resolve `record.rs` `_UpdateBindGroup`: route Metal's in-place bind-group update
  (`easily_update_bind_groups`), clean error/omit elsewhere.
- Sweep remaining stale comments and unused imports.

## Data flow (target behavior of the two tests)

`basic_buffer_copy`: `write` stages into the upload buffer's host copy → recorder `copy_buffer`
upload→gpu and gpu→download run on device copies → `access(download, read)` makes the host copy of
`download` current (device→host copy or direct map, Component 2) → readback asserts `[1,2,3,4]`.

`add_numbers`: recorded `write_buffer` uploads into buffers 1/2/3 (staging copy) → `dispatch_kernel`
reads 1/2, writes 3 on the device → `copy_buffer` 3→download → `access(download, read)` pulls
device→host → asserts `[6,8,10,12]`.

## Testing

- `cargo nextest run --all-features --all-targets --no-fail-fast` (config `.config/nextest.toml`).
- Per-backend runs via `all_backend_tests!` (feature-gated).
- New `OutOfDateTracker` unit tests.
- Vulkan CI is separately broken (#140); validate locally. Fixing CI is a separate issue (a
  `fix-vulkan-ci-140` worktree already exists) and is not part of this spec.

## Risks & de-scope levers

- **Copy insertion (Component 2)** is the most complex change — the residency↔record seam. If the
  seam proves invasive, prefer having residency return copy *descriptors* recorded by the submit
  path over embedding HAL calls in residency.
- **wgpu (Component 3)** is the fiddliest backend. De-scope lever, used only after checking in:
  keep Vulkan + Metal green, mark wgpu residency-limited, and split remaining wgpu work to a
  follow-up issue.

## Acceptance criteria

1. `create_sync_thread` and all reachable `todo!()`s in the residency/sync/record path are gone.
2. Both integration tests pass on Vulkan and Metal; wgpu too, or the wgpu de-scope lever is taken
   with the user's agreement and a follow-up issue filed.
3. New OOD unit tests pass.
4. No `idx.unwrap()` / self-wait hazards in the access-tracking paths.
5. Clean `cargo clippy` (no new warnings) and dead code removed.
