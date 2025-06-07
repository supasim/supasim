### Vulkan synchronization issue
There is a vulkan sync issue that makes the add_numbers_vulkan test fail on my machine but not on the cloud, suggesting invalid usage of vulkan. While the validation layers don't catch this, it is suspected to be a synchronization issue due to the nature of the issue and the seeming correctness of other aspects.

This issue has been present for quite some time.

###  MappedBuffer drop order issue
A MappedBuffer object existing doesn't prevent a buffer from being destroyed manually, which can cause a whole host of issues, including writing to memory that is no longer valid, error during unwrap on drop, etc.

### No checking of valid usage/handling
For example, mapping has certain requirements such as offset being a multiple of 8/16. SupaSim should automatically check for invalid offsets and handle them behind the scenes without throwing errors. In general, everything regarding size/offset should be multiples of 16.

### No checking of physical device limits
Physical device limits are necessary to check to ensure proper usage.