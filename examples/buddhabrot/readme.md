# Buddhabrot demo

This demo renders the [Buddhabrot](https://en.wikipedia.org/wiki/Buddhabrot) fractal.
The fractal is stretched to fit the window size.
Rendering is an iterative process, where each frame a certain number of new paths are rendered on top of previous runs.
Previous paths are dropped when the window is resized, resetting the render.

Most of the rendering code is taken almost verbatim from <https://sotrh.github.io/learn-wgpu/>.

## Options

Options can be configured by setting environment variables before running the example.
* `BACKEND` - if this is exactly "vulkan", then the vulkan backend is used. Otherwise the wgpu backend is used
* `WORKGROUP_DIM` - the number of workgroups in each dimension. The number of paths per frame is proportional
  to the cube of this value. Defaults to 4 if not provided or not a valid integer. Clamped between 1 and 16.
  Each workgroup tracks 64 paths per frame.
* `ITERATION_SETS` - each iteration set is 512 iterations. Therefore, the number of iterations rendered is 512*ITERATION_SETS. Defaults to 4, must be at least 1.
* `SKIP_LAST_POINTS` - when rendering with fewer iterations, more of the points rendered end up on the outside,
  providing an ugly and distracting background. Setting this to e.g. 10 prevents this from happening, but may bias the result.
  Defaults to 0.