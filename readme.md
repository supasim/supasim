[![CI](https://github.com/supasim/supasim/actions/workflows/ci.yml/badge.svg)](https://github.com/supasim/supasim/actions/workflows/ci.yml)
[![Lines of code](https://tokei.rs/b1/github/supasim/supasim)](https://github.com/supasim/supasim)

# SupaSim

## Planned features

### Backends
* Wgpu for when certain assumptions can't be made(old/crappy/esoteric setups), or where vulkan isn't supported for some reason. This will massively limit concurrency and performance. In case certain assumptions can't be met, they will be emulated, further degrading performance.
* Cuda for nvidia GPUs
* HIP(maybe, long term)
* Metal(long term) 2.3 or 2.4 for apple devices with M1 or newer
* Vulkan for GPUs with somewhat modern capabilities. Currently, that means
  * At least 16 concurrently executing threadgroups(aka streaming multiprocessors). It seems that all discrete GPUs and desktop integrated GPUs from the past decade have this.
  * Allow threadgroups with 1024 threads. All systems I checked have exactly this limit.
  * Warp size that is a factor/multiple of 64 - lowest I could find is 15 year old Nvidia GPUs with 16. No reason this wouldn't be a power of 2.

### Graphics
* WGPU powered
* Graphing with plotters

### GUI with hot reloading
* Context and device stuff will be handled by the GUI, to preserve it across runs
* GUI will handle reloading shaders, rebuilding and reloading simulations themselves
* GUI will have settings that are saved across simulations. GUI can be installed itself independent of any project.
* Possibly support python projects?

### Shaders
* Shaders will be written in slang, then transpiled at runtime
