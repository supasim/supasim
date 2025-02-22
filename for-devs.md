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