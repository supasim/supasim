# Sync & resource tracking overview (Out ouf date)

* The instance tracks all command recorders submitted by index and associated semaphore
* The instance occasionally checks for completed semaphores
* Whenever a buffer is used for commands it gains details about how it is used and the semaphore
* When a buffer is destroyed/dropped, it is added to a list of buffers to be destroyed when all users finish
* Whenever a command recorder is submitted, it is destroyed and a semaphore is returned. The command recorders used get
  added to the list of command recorders and semaphores

# Avoiding deadlocks
* Locks are acquired in a specified order across all functions
  * First, the instance is locked
  * Then any other resources, in alphabetical order of their type names
* This isn't currently followed unfortunately