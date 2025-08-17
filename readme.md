# SupaSim

[![CI](https://github.com/supasim/supasim/actions/workflows/ci.yml/badge.svg)](https://github.com/supasim/supasim/actions/workflows/ci.yml)
[![Lines of code](https://tokei.rs/b1/github/supasim/supasim)](https://github.com/supasim/supasim)

SupaSim is a general purpose GPU and simulation toolkit, made in Rust, with kernels written in [Slang](https://shader-slang.org/).

## Repo Overview
Supasim encompasses a large set of libraries. The following libraries live on this repo:
* [![Crates.io](https://img.shields.io/crates/v/supasim.svg?label=supasim)](https://crates.io/crates/supasim)
* [![Crates.io](https://img.shields.io/crates/v/supasim-hal.svg?label=supasim-hal)](https://crates.io/crates/supasim-hal)
* [![Crates.io](https://img.shields.io/crates/v/supasim-kernels.svg?label=supasim-kernels)](https://crates.io/crates/supasim-kernels)
* [![Crates.io](https://img.shields.io/crates/v/supasim-dev-utils.svg?label=supasim-dev-utils)](https://crates.io/crates/supasim-dev-utils)
* [![Crates.io](https://img.shields.io/crates/v/supasim-types.svg?label=supasim-types)](https://crates.io/crates/supasim-types)

The following additional libraries live on the [supasim github organization](https://github.com/supasim):
* [![Crates.io](https://img.shields.io/crates/v/supasim-spirv-cross-sys.svg?label=supasim-spirv-cross-sys)](https://crates.io/crates/supasim-spirv-cross-sys)
* [![Crates.io](https://img.shields.io/crates/v/supasim-spirv-tools-sys.svg?label=supasim-spirv-tools-sys)](https://crates.io/crates/supasim-spirv-tools-sys)

## Shaders

Shaders are written in [Slang](https://shader-slang.org/). Supasim provides some advice on how to avoid unintended behavior (aka bugs) in Slang code, which can be seen [here](./kernels/readme.md). When compiling kernels, they may not be directly transpiled to the target language. For example, many languages first go through SPIR-V, enabling SPIRV-Opt to optimize code, as well as allowing use of SPIRV-Cross, which may be more mature and stable than Slang's direct output into the target language.

## Compiling

### Requirements
All requirements are part of the vulkan sdk. The only dynamically linked component is `DXC` for compiling DirectX shaders.
This is required to use the wgpu backend, or to compile DXIL code for the shader library. Therefore, users seeking this
functionality should plan to ship a copy of `dxc.dll`. If you wish to use older versions of DXC for whatever reason, you should also ship `dxil.dll`, which isn't part of the vulkan
sdk and must be downloaded separately, such as from DXC github releases.

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
Features can be enabled by passing a command line argument of this form: `--features="feature1,feature2"`. The available features for supasim are:
* `wgpu` - enables the `wgpu` backend, which is the most cross platform backend, and also provides more error checking
* `vulkan` - enables the raw vulkan backend for higher performance
* `metal` - enables the metal backend

Features for `supasim-kernels` are:
* `msl-out` - enables writing MSL output. Requires SPIRV-Cross
* `wgsl-out` - enables writing `wgsl` for WebGPU
* `dxil-out` - enables writing `dxil` output. Note that HLSL output is always supported
* `opt-valid` - enables extra optimizations or validation. Requires SPIRV-Tools and SPIRV-Cross.

Note that `supasim-kernels` always supports writing SPIR-V. Also note that as `wgpu` and `vulkan` backends both only take
SPIR-V at the moment, no features need to be enabled for use with `supasim`.

## License

Licensed under either of

* Apache License, Version 2.0
  ([LICENSE-APACHE](LICENSE-APACHE) or <http://www.apache.org/licenses/LICENSE-2.0>)
* MIT license
  ([LICENSE-MIT](LICENSE-MIT) or <http://opensource.org/licenses/MIT>)

at your option.

## Contribution

Unless you explicitly state otherwise, any contribution intentionally submitted
for inclusion in the work by you, as defined in the Apache-2.0 license, shall be
dual licensed as above, without any additional terms or conditions.