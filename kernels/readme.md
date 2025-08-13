# All about kernels

## What is this folder?
This folder is for kernels that may be used in examples or tests across multiple folders in this project. As a user, you can probably ignore this folder.

## Known issues with slang
**SLANG IS WEIRD.** Lots of thing won't behave as expected. It is highly recommended that you use assertions to verify the correctness of generated reflection information, which reflects what slang outputs.

**Always use the `[binding(0)]` attribute.** This prevents *soo* many bugs before they occur.

Anyway, on with the list:

* Always try to put read-write buffers before readonly buffer. On Metal, buffers may be automatically reordered behind the scenes without supasim-kernels catching it.
* Non-static const's are reflected in the shader's interface, which is *never* intentional
* Slang tries to determine which global variables are used, but depending on optimization level this can sometimes have surprising results.
  * Multiple bindings in the same struct/block get optimized away together or not at all. If you group all of the bindings for a function together in a struct, you can guarantee they will all be reflected in the interface.
* Supasim only uses a single descriptor set/binding space/whatever. Sometimes, if you have some inputs declared globally and some declared as function parameters, or multiple global variable blocks, or any of a number of other setups, slang will automatically set some of them to have different binding spaces. `supasim-kernels` automatically panics if it detects a nonzero binding space.
* Whether a buffer is read/write or readonly is very important - supasim uses this information for synchronization and such.
* Non buffer and non builtin inputs are unsupported. We don't support textures, argument buffers, push constants, uniform buffers, or any other advanced features. Slang sometimes represents these as constant buffers or the like.
