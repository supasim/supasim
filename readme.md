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
  * (Not explicitly required but this is optimized for) At least 16 concurrently executing threadgroups(aka streaming multiprocessors). It seems that all discrete GPUs and desktop integrated GPUs from the past decade have this.
  * Allow threadgroups with 1024 threads. All systems I checked have exactly this limit.
  * Warp size that is a factor/multiple of 32 - lowest I could find is 15 year old Nvidia GPUs with 16. No reason this wouldn't be a power of 2.

### Graphics
* WGPU powered
* Graphing with plotters

### GUI with hot reloading
* Context and device stuff will be handled by the GUI, to preserve it across runs
* GUI will handle reloading kernels, rebuilding and reloading simulations themselves
* GUI will have settings that are saved across simulations. GUI can be installed itself independent of any project.
* Possibly support python projects?
* Support taking or loading snapshots of a run (no guarantees of determinism)

### Kernels
* Kernels will be written in slang, then transpiled at runtime

## Compiling
Compiling is a somewhat involved process.
### Requirements
All requirements are part of the vulkan sdk. The only dynamically linked component is `DXC` for compiling DirectX shaders.
This is required to use the wgpu backend, or to compile DXIL code for the shader library. Therefore, users seeking this
functionality should plan to ship a copy of `dxc.dll` and `dxil.dll` with such applications (`dxil.dll` isn't part of the vulkan
sdk so must be downloaded separately, such as from DXC github releases).

If you choose to use the vulkan sdk, you must set the environment variable `VULKAN_SDK` to the path where it was installed to. It is
your responsibility to make sure that `dxc.dll` and `dxil.dll` are on the system library search path if you use them. A build configured
to use these that cannot find them may not function properly.

Otherwise, the full list of dependencies are as follows:
* Slang
* SPIRV-Cross (for most build configsl, and for stable compilation to MSL)
* SPIRV-Tools (for optimizing and extra validation)
* DXC and DXIL (for aforementioned use cases)

Follow the setup instructions in the repositories corresponding to the used rust bindings for each of the above.

### Features
Features can be enabled by passing a command line argument of this form: `--features="feature1,feature2"`. The available features for supasim are
* `wgpu` - enables the `wgpu` backend as well as `wgpu` interop support (such as for memory sharing)
* `vulkan` - enables the raw vulkan backend for maximum performance
* `cuda` - currently unused, in the future will be used for `cuda` backend

Features for `supasim-kernels` are:
* `msl-stable-out` - enables writing stable MSL output. Requires SPIRV-Cross
* `wgsl-out` - enables writing `wgsl` for WebGPU
* `dxil-out` - enables writing `dxil` output. Note that HLSL output is always supported
* `opt-valid` - enables extra optimizations or validation. Requires SPIRV-Tools

Note that `supasim-kernels` always supports writing SPIR-V. Also note that as `wgpu` and `vulkan` backends both only take
SPIR-V at the moment, no features need to be enabled for use with `supasim`.