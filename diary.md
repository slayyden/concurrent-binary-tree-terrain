# TODO
 - [x] Create window
 - [x] Create instance
 - [x] Create physical device
 - [x] Create queue
 - [x] Create Logical device
 - [x] Create pipeline
 - [x] Create pipeline layout
 - [x] Create shader modules
 - [x] Write basic shaders
 - [x] Compile basic shaders
 - [x] Design synchronization system
 - [x] Load correct features
 - [x] Fix synchronization validation errors
 - [ ] Enable depth test
 - [x] use multiple command buffers
 - [x] refactor into multiple files
 - [x] use mesh buffers
 - [ ] create projection matrices
 - [ ] create more interesting vertex buffer

# Design
## Vulkan Features
Vulkan 1.3 that also works on Apple Silicon. Bindless architecture.
- Dynamic state
- Dynamic rendering
- Buffer device address (TODO)
- No synchronization 2
## Synchronization
- triple buffering
- !!! CPU waits for previous frame to finish
  - maybe use multiple command buffers but keep CPU writing to one command buffer at a time
- binary semaphores used within a single frame
- timeline semaphores for inter-frame synchronization
- fences for command buffer reuse

# Diary
## Day 4
### Bugs
- Validation error about the number of timeline semaphores in queue wait semaphores argument
  - wait semaphores must either all be binary or all be timeline
  - we choose to put timeline semaphore wait into command buffer because you can't do that for binary semaphores
- No clear color
  - have to explicitly write clear commands
- Green screen :(
  - forgot to set color write mask in color attachment state
