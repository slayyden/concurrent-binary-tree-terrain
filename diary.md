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
 - [ ] put together pipeline
 - [ ] test pipeline
 - [ ] visualize pipeline
 - [ ] write classification function
 - [ ] port to GPU

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

# Day 29
OKAY

Step 1: Each bisector is classified.
- includes backface, frustum, and size based "culling"
- reset some buffers
- split and merge (only even) bisector memoryblock indices are atomically appended to "dynamic" arrays with length S and M
- record remaining # of slots as R
Step 2: Splitting
- dispatch S threads, one for each splitting bisector
- split commands are written to bisectors, ensuring not to exceed R
- atomically append memblk indices to allocation array A (one for each allocation)
Step 3: Allocation
- dispatch len(A) threads
- each thread looks up its subdivision pattern and binary searches for up to 3 free memory blocks and marks them as occupied
- the indices of these memory blocks are written to the bisector
Step 4: Splitting Pointer updates
- dispatch len(A) threads, one for each split bisector
- update all neighbor pointers of the children
- update neighbor pointers of the parent's unsplit neighbors
- write new heapids
Step 5: Enqueue merges
- dispatch M threads
- thread reads bisector data
- split command => return early
- write merge commands in bisector data
- 2 kinds of merge command
  - lower heapid: merge with next
  - higher heapid: merge with prev
- write lower pairs to buffer M'
Step 6: Merging
- dispatch len(M') threads

- pointer updates:
if upper->twin->command is "merge with prev"
  merged->prev = upper->twin->prev->id
else
  merged->prev = upper->twin->id
// symmetric case for lower->twin
merged->twin = upper->next

- update heapid
- free upper

Step 8: CBT update
- store number of bisectors in B
Step 9: Write and displace vertex buffer
- dispatch B threads
Step 10: Render (vertex and fragment shaders)

# Day 34
leaf: 0b1111110110
heapid:
| -1  | 1.  | 2.  | -1   | 4.  | 5.  | 0.1  | 0.0  | 3.1  | 3.0

leaf: 0b1111100100
| -1  | 1.  | 2.  | -1   | 4.  | 5.  | 0    | -1   | 3.   | -1


# FINAL STRETCH
- CAMERA CONTROL
- VIEWPROJECT MATRICES
- FINISH CLASSIFICATION
- WRITE TO THE VERTEX BUFFER
- VERIFY SHADER INPUTS AND OUTPUTS
- WRITE INDIRECT DISPATCHES
  - GLOBAL BARRIERS BETWEEN THEM
- PERLIN NOISE
  - IN THE COMPUTE SHADER FUCK YOU
- FANCY FRAGMENTS
