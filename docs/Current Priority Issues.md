### No checking of valid usage/handling
For example, mapping has certain requirements such as offset being a multiple of 8/16. SupaSim should automatically check for invalid offsets and handle them behind the scenes without throwing errors. In general, everything regarding size/offset should be multiples of 16.

### No checking of physical device limits
Physical device limits are necessary to check to ensure proper usage.