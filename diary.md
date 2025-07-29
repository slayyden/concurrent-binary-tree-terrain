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
 - [x] create CBT implementation on CPU
 - [x] test CBT implementation on CPU

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

memory can either be contiguous in temp buffers
or fragmented in the CBT memory pool

how do we know the depth of a bisector?
- store in upper bits of heapid?
- heapid is a NUMBER that INDEXES the heap
- root bisectors lie on the highest level that can fit them
  - edit: NO THEY FUCKING DONT U IDIOT
    - ROOT BISECTORS TAKE UP MEMORY BLOCKS IN THE CBT LIKE ALL OTHER BISECTORS
    - THEY ARE ON THE HIGHEST LEVEL OF THE BISECTION TREE, BUT NO THE CBT
  - binary tree vs prefix sum array:
    - pros of binary tree
      - half the work compared to reduction prefix sum
      - theorhetically even better (on non Mac) if the array cannot fit in shared memory, but that will not happen here
    - pros of prefix sum array
      - no empty slots when number of root bisectors is not a power of 2
      - memory usage (for 2048 leaf elements = 2^17 bits)
        - optimized binary tree: 26,592 bits
        - naive prefix sum array: 65,504 bits (still fits in 32 kb, occupies about 1/4)
          - doubled up prefix sum array: (32,736 bits)
          - doubled up u16 prefix sum array + microtree (3 u32): (16,464 bits)
  - new invariant: contiguity
    - we HAVE to use contiguous buffers anyway for rendering
    - problem: merges create holes
      - split using mergeless budget
      - then determine merges
      - then free memory using merges
      - then binary search AGAIN and fill in splits
        - problem: more merges than splits can still lead to fragmentation
        - there MUST be a partitioning step no matter what
  - defragmentation step:
    - CBT reduction (prefix sum)
    - all threads read number of memory blocks, n
      - if i < n do return
    - thread i finds memory block i
    - thread i binary searches for location of memory block i
    - thread i writes the contents of block i into index i of a new array (writes can be coalesced)
      - requires another swap chain thingy
      - OR we could store things temporarily in Shared Memory
- binary tree
  -
- prefix sum array w/ binary search
- heapid: 000...00 BASE_SECTION SUBDIVISION_SECTION
  - SUBDIVISION SECTION goes from highest to lowest level of the tree


do we need to map heapids to memory blocks?
- neighbors are identified their heapids
- allocation indices in the memory block array are SEPARATE

Okay so what data to we need and where?
CLASSIFICATION:
- bisector verts
  - heapids (traverse CBT) and write to contiguous heapid buffer
  - root bisectors (first chunk of memory)
SPLIT COMMAND WRITING:
- bisectors (from heapid buffer)
  - classification (contiguous)
  - neighbors (contiguous?)
ALLOCATION:
- bisectors
  - list of heapids with allocations
  - list of empty CBT slots
SPLITTING NEIGHBOR UPDATES:
- bisectors
  - neighbors
  - command
MERGE COMMAND WRITING:
MERGING:
- parent data has to come from somewhere
- could be recomputed
- newest vertex must be replaced by parent vertex
MERGING NEIGHBOR UPDATES:
CBT RESUMMATION:
- CBT bitfield
DRAWING:
- number of bisectors
- vertex buffer (must be contiguous)

CBT MEMORY POOL:
- heapid
- neighbor pointers

## Day 4
### Bugs
- Validation error about the number of timeline semaphores in queue wait semaphores argument
  - wait semaphores must either all be binary or all be timeline
  - we choose to put timeline semaphore wait into command buffer because you can't do that for binary semaphores
- No clear color
  - have to explicitly write clear commands
- Green screen :(
  - forgot to set color write mask in color attachment state

# Day 27
Warp scans are SLOW compared to multi warp scans, but can have lower latency.
