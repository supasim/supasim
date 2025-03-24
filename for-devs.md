## Dependencies template(copy from source)
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

## Synchronization things
### Metal
Metal supports similar synchronization to vulkan, except the different primitives are named differently(fences instead of semaphores and the like). It also supports automatic synchronization of certain resources, depending on how they are allocated. Resources allocated directly from a device default to synchronized, while resources allocated from an explicit memory heap must have an additional flag at creation time.
### Vulkan
Vulkan has very intricate synchronization, but libraries like vulkano can help(although vulkano's task graph is experimental, the use case here is relatively narrow, consisting of only compute dispatches and memory copies, so the few issues should be easy to find). It may also be feasible to translate a task graph system from another library written in another language.
### Cuda
Cuda supports cuda-graphs, but these are only in "recent" versions of cuda. Otherwise, the synchronization is rather strange. Essentially, each item submitted to a stream must wait for all previous items in the stream, and you can also optionally add extra dependencies. There are different numbers of streams on different cards. This may be easier to manually synchronize than other apis, particularly for simpler workloads, as 16 streams is a lot, far more than the number of queues usually used with vulkan for example. It also helps that there aren't too many synchronization requirements.

## Work for abstractions in supasim
* Sharing references/multithreading
* Moving buffers in and out of GPU memory when OOM is hit
* Synchronization/creation and synchronization of command buffers
* Lazy operations
* Combine/optimize allocations and creation of things

* In buffer creation, specify a "priority" to prevent it from being swapped in/out of memory constantly
* Give a buffer an "item size" that it can be split along when swapping in/out of GPU memory
* In kernel dispatch, specify which ranges of a buffer might be used
* Split certain dispatches into multiple if the gpu doesn't support the size

## ~~How to handle types/references in front-facing supasim~~ (resolved)
* Front facing type
  * Should have a destroy method that can be called either way
  * Needs to be reference counting somehow, and a destructor that calls destroy
  * Needs to avoid circular references
  * Needs to still be trackable from the main instance, so it must have access to the instance and the instance must have access to it
    * This should be fine. The instance should destroy it when the instance is destroyed, and destorying it should tell the instance to remove the reference
  * References to things other than the instance should be done through an "id" system rather than direct references

## Bindless details
### Cuda
Cuda doesn't use anything like bind groups or descriptor sets. You specify the resources at dispatch time.
### Metal
Metal dispatches are similar to cuda.
### Vulkan
* `VK_KHR_push_descriptor` - only in vk1.4, push all descriptors at dispatch time
* `VK_KHR_buffer_device_address` - in vk1.2, push buffers specifically, requires modifications to shader code
* Bindless - may be equivalent to one of the above, this is almost a buzzword.


## GUI options
Note that CPU support is desired due to relieving gpu/not using its memory, even if it is not always the first choice. Of course wgpu rendering in app is desired, as that is how visualizations will be done. Nice image and svg with updating support is also desired due to graphing with plotters.
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
