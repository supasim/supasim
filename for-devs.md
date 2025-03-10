## Dependencies template
anyhow.workspace = true
bitflags.workspace = true
bytemuck.workspace = true
getrandom.workspace = true
libloading.workspace = true
log.workspace = true
nalgebra.workspace = true
rand.workspace = true
rayon.workspace = true
serde.workspace = true
serde_json.workspace = true
thiserror.workspace = true

# Work for abstractions in supasim
* Sharing references/multithreading
* Moving buffers in and out of GPU memory when OOM is hit
* Synchronization/creation and synchronization of command buffers
* Lazy operations
* Combine/optimize allocations and creation of things

* In buffer creation, specify a "frequency" to prevent it from being swapped in/out of memory constantly
* Give a buffer an "item size" that it can be split along when swapping in/out of GPU memory
* In kernel dispatch, specify which ranges of a buffer might be used

# Options for vulkan without descriptor sets
* `VK_KHR_push_descriptor` - only in vk1.4, push all descriptors at dispatch time
* `VK_KHR_buffer_device_address` - in vk1.2, push buffers specifically, requires modifications to shader code


## GUI options
* [Iced](https://github.com/iced-rs/iced) - popular, mature
  * Supports wgpu integration using special widgets
  * Supports cpu and gpu, using its own wgpu renderer or tiny-skia
  * Release, counter example - 61, 23.5MB
* [Xilem](https://github.com/linebender/xilem) - modern, focus on important features such as native integration and accessibility
  * See [wgpu integration issue](https://github.com/linebender/xilem/issues/395)
  * Supports only gpu rendering using vello, potentially better performance
  * Release, calc example - 68s, 25.7MB
* [Floem](https://github.com/lapce/floem) - used in production, probably wgpu rendering
  * See [wgpu integration issue](https://github.com/lapce/floem/issues/687)
  * Supports cpu and gpu, using vello, vger, its own renderer, or tiny-skia for cpu
  * Release, counter example - 64s, 25.9MB