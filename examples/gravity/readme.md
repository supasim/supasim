# Gravity Demo
This is a demo of the capabilities of SupaSim that runs a simulation of many particles under the influence of gravity.

### Steps
* All particles are stored in a contiguous buffer
* Whenever a particle escapes a given radius it is swapped with a particle at the end of the buffer and the length of the buffer is reduced
* There is a chunk tracker buffer that keeps track of the number of particles in each buffer
* There is a chunk data buffer that stores the indices to actual particles sorted by chunk, in bins. If too many particles are in
one chunk, it can point to another bin in the buffer.
* For each chunk, the total influence felt by all farther chunks is summed for the center of the chunk and added to each particle
* Then each particle calculates its influence felt by all particles in its chunk or neighboring chunks
* Then these influences are added and applied to each particle
* Bins are completely recalculated very rarely. Most iterations, no bin work is done. Some iterations, particles are checked
to ensure they haven't changed bins. Bins are only recalculated when most bins have been allocated (as bins can't be deallocated)