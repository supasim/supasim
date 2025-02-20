# SupaSim

## Planned features

### Backends
* Cuda
* Vulkan
* OpenCL(long term)
* HIP(long term)
* Metal(long term) 2.3 or 2.4

### Graphics
* WGPU powered
* Graphing with plotters

### GUI with hot reloading
* Context and device stuff will be handled by the GUI, to preserve it across runs
* GUI will handle reloading shaders, rebuilding and reloading simulations themselves
* GUI will have settings that are saved across simulations. GUI can be installed itself independent of any project.

### Shaders
* Shaders will be written in slang, then transpiled at runtime