# Sync & resource tracking overview

* The instance tracks all command recorders submitted by index and associated semaphore
* The instance occasionally checks for completed semaphores
* Whenever a buffer is used for commands it gains details about how it is used and the semaphore
* When a buffer is destroyed/dropped, it is added to a list of buffers to be destroyed when all users finish
* Whenever a command recorder is submitted, it is destroyed and a semaphore is returned. The command recorders used get
  added to the list of command recorders and semaphores