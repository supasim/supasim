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
struct Parameters
{
    RWStructuredBuffer<uint32_t> buffer;
}

[shader("compute")]
[numthreads(1, 1, 1)]
void double(uint3 threadId: SV_DispatchThreadID, ParameterBlock<Parameters> parameters)
{
    uint index = threadId.x;
    parameters.buffer[index] = parameters.buffer[index] * 2;
}
