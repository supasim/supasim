/* BEGIN LICENSE
  SupaSim, a GPGPU and simulation toolkit.
  Copyright (C) 2025 SupaMaggie70 (Magnus Larsson)


  SupaSim is free software; you can redistribute it and/or
  modify it under the terms of the GNU General Public License
  as published by the Free Software Foundation; either version 3
  of the License, or (at your option) any later version.

  SupaSim is distributed in the hope that it will be useful,
  but WITHOUT ANY WARRANTY; without even the implied warranty of
  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
  GNU General Public License for more details.

  You should have received a copy of the GNU General Public License
  along with this program.  If not, see <http://www.gnu.org/licenses/>.
END LICENSE */
// Courtesy of ChatGPT

struct OutputSize {
    width: u32,
    height: u32,
}

@group(0) @binding(0)
var<uniform> size: OutputSize;

@group(0) @binding(1)
var<uniform> max_value: u32;

@group(1) @binding(0)
var<storage, read> buffer: array<u32>;

// Vertex positions for a full-screen quad made of two triangles (CCW winding)
var<private> full_screen_triangle_vertices: array<vec2<f32>, 6> = array<vec2<f32>, 6>(
    vec2<f32>(-1.0, -1.0), // bottom-left
    vec2<f32>(-1.0,  1.0), // top-left
    vec2<f32>( 1.0,  1.0), // top-right

    vec2<f32>(-1.0, -1.0), // bottom-left
    vec2<f32>( 1.0,  1.0), // top-right
    vec2<f32>( 1.0, -1.0)  // bottom-right
);

@vertex
fn render_vs(@builtin(vertex_index) in_vertex_index: u32) -> @builtin(position) vec4<f32> {
    let pos = full_screen_triangle_vertices[in_vertex_index];
    return vec4<f32>(pos, 0.0, 1.0);
}

@fragment
fn render_fs(@builtin(position) position: vec4<f32>) -> @location(0) vec4<f32> {
    let x = u32(position.x);
    let y = u32(position.y);
    let index = y * size.width + x;
    let value = buffer[index];
    //let scaled = f32(value) / f32(max_value) * 3.0;
    return vec4<f32>(0.0, 0.0, scaled, 1.0);
}
