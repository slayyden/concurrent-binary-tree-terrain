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

# GPU Debugging Capture 1
scene buffer is 0x157776700
cbt interior is buffer 0x14770d2e0
cbt leaves is probably buffer 0x158b14dd0
indirect buffer: 0x1477131e0


- frame n-1
  - 14269: cbt reduce, 12888 bisectors
    - scene is 0x10007e40000
    - cbt_interior is 0x100000f00000
    - cbt_leaves is 0x100000e38000
- frame n
  - 14735: cbt reduce, 24576 bisectors
    - scene is 0x10007e40000
    - cbt_interior is 0x100000f00000 = 0b100000000000000000000111100000000000000000000
    - cbt_leaves is 0x100000e38000 = 0100000000000000000000111000111000000000000000
  - 14781: vertex compute
    - pc cbt interior[0] is 24576
      - but so are other elements??
- frame n+1
- 15202: cbt reduce, everything is zero
  - push constants
    - scene is 0x10007e40000
    - cbt_interior is 0x7003_00007002 = 123158187241474 = 0b11100000000001100000000000000000111000000000010
      - 0x7002 = 28674
      - 0x7003 = 28675
    - cbt_leaves is 0x7005_00007004 = 123166777176068 = 0b11100000000010100000000000000000111000000000100

# GPU Debugging Capture 2
dispatch buffer: 0x151811990
- root_bisector vertices is not (offset 0)
- cbt_interior is saved from overrun  (offset 8)
-

# Well we fixed it
I don't even know what the cause was. Fuck my life man

# GPU Debugging Capture 3
- i wonder if we're facing the same bug somehow

buffer view unreliable??

cbt interior buffer: 0x1212347f0

(cbt_reduce) → 12
1173 (reset) -> 12
1397 (prepare merge dispatch) -> 12
1475 (cbt_reduce) -> 18
1586 (cbt_reduce) -> 18

1810 (prepare merge dispatch) -> 18
1888 (cbt_reduce) cbt_interior	uint*	0x100005b8000 → 30
1999 (reset) cbt_interior	uint*	0x100005b8000 → 30

2223 (prepare merge dispatch) -> 30
2301 (cbt_reduce) cbt_interior	uint*	0x100005b8000 → 54
2412 (reset) cbt_interior	uint*	0x100005b8000 → 54

2636 (prepare merge dispatch) cbt_interior	uint*	0x100005b8000 → 54
2714 (cbt_reduce) cbt_interior	uint*	0x100005b8000 → 85

2893 (classify)cbt_interior	uint*	0x100005b8000 → 85
3049 (prepare merge dispatch) cbt_interior	uint*	0x100005b8000 → 85
3127 (cbt_reduce) cbt_interior	uint*	0x100005b8000 → 146
3127 (cbt_reduce 2) cbt_interior	uint*	0x100005b8000 → 146

wtf is happening here??
3153 (vertex compute 0) cbt_interior	uint*	0x100005b8000 → 146
3153 (vertex compute 0 2) cbt_interior	uint*	0x100005b8000 → 146
3153 (vertex compute 64) cbt_interior	uint*	0x100005b8000 → 146
3153 (vertex compute 64 2) cbt_interior	uint*	0x100005b8000 → 144
3153 (vertex compute 96) cbt_interior	uint*	0x100005b8000 → 146
3153 (vertex compute 96 2) cbt_interior	uint*	0x100005b8000 → 146
3153 (vertex compute 128) cbt_interior	uint*	0x100005b8000 → 144
3153 (vertex compute 128 2) cbt_interior	uint*	0x100005b8000 → 146
3153 (vertex compute 128 3) cbt_interior	uint*	0x100005b8000 → 144
3153 (vertex compute 160) cbt_interior	uint*	0x100005b8000 → 146

3153 (vertex compute 0 a) 146
3153 (vertex compute 32 a) 146
3153 (vertex compute 64 a) 144
3153 (vertex compute 96 a) 146
3153 (vertex compute 128 a) 144
3153 (vertex compute 160 a) 144

3153 (vertex compute 32 b) 146
3153 (vertex compute 64 b) 146

3238 (reset) cbt_interior	uint*	0x100005b8000 → 144
3238 (reset 2) cbt_interior	uint*	0x100005b8000 → 144

3306 (classify) cbt_interior	uint*	0x100005b8000 → 146
3450 (cbt_reduce) cbt_interior	uint*	0x100005b8000 → 220
3651 (reset) cbt_interior	uint*	0x100005b8000 → 220

3719 (classify) cbt_interior	uint*	0x100005b8000 → 212
3719 (classify 2) cbt_interior	uint*	0x100005b8000 → 220

3719 (classify 0 a) cbt_interior	uint*	0x100005b8000 → 213
3719 (classify 0 a) cbt_interior	uint*	0x100005b8000 → 213

3745 (prepare split dispatch) cbt_interior	uint*	0x100005b8000 → 220
3875 (prepare merge dispatch) cbt_interior	uint*	0x100005b8000 → 223
3953 (cbt_reduce) cbt_interior	uint*	0x100005b8000 → 341
4064 (reset) cbt_interior	uint*	0x100005b8000 → 332

4132 (classify) cbt_interior	uint*	0x100005b8000 → 332
4288 (prepare merge dispatch) cbt_interior	uint*	0x100005b8000 → 333
4366 (cbt_reduce) cbt_interior	uint*	0x100005b8000 → 492
4477 (reset) cbt_interior	uint*	0x100005b8000 → 470

6844 (cbt_reduce)
  cbt_interior	uint*	0x100005b8000 → 4230
6870 (reset)
  cbt_interior	uint*	0x100005b8000 → 3318

18821 (cbt_reduce)
  cbt_interior	uint*	0x100005b8000 → 124628
18847 (reset)
  cbt_interior	uint*	0x100005b8000 → 116407

19647 (cbt reduce):
  cbt_interior	uint*	0x100005b8000 → 127170
19673 (vertex compute):
  cbt_interior	uint*	0x100005b8000 → 126971 (-199)
19758 (reset):
  cbt_interior	uint*	0x100005b8000 → 126620 (-351)
19826 (classify):
  cbt_interior	uint*	0x100005b8000 → 126681 (+21)
19852 (prepare split dispatch)
  cbt_interior	uint*	0x100005b8000 → 127006 (+25)

# GPU Debugging Capture 4
cbt interior: 0x12a738190


no consistency between the SAME thread in the SAME compute shader
shader view:
  reduce: 29622
  vertex compute: 32669
buffer view:
  classify: 34267
  prepare_merge: 34267
  reduce: 34267
  vertex compute: 34267
  reset: 34267


how about leaves:
- 3758093807
- 3758093807
- 3758093807
leaves are consistent

# Session 5
atomic loads are inconsistent between runs of the debugger
binary view is BACKWARDS
heapid buffer: 0x13082a660
also heapid buffer: 0x1308339c0
- THEY'RE FUCKING ALIASED FMLLLLL
cbt_leaves buffer: 0x13090d4e0

want split buffer: 0x
Frame k-4 (Present 3764)
  Geometry is fine
    Tri A is 30
    Tri B is 65
  Draw call:
    heapid[43] = 0b100110 = 38
      - 38 % 8 != 0 => will not be selected to split next time
Frame k-3 (Present 4174)
  2 Nontriangular faces in geometry
    Verts: 87, 88, 89, 189, 190, 191
    Tris: 29 (A), 63 (B)
      Tri A should have been split by Tri 56 (from last geo)
      Tri B should have been split by Tri 69 (from last geo)

  Classify:
    heapid[56] = INVALID_INDEX
    heapid[69] = 0b100010111 = 279
    leaves[56] = 0
  Vertex Compute
    Thread 29:
      curr_id = 43
      heapid[43] = 0b100110 = 38
        this heapid reflects the geometry
        so why are the heapids wrong??
    Thread 63:

Frame k-2 (Present 4584)
  Hole in geometry after draw call
  Attachment looks fine??
Frame k-1 (Present 4994)
non triangular grid

Frame n-1 (Present ??):
  Draw call: 392802 verts
    Dispatch:
      Draw call: 392802 verts
      Dispatch Split: 200
      Dispatch Allocate: 1
      Dispatch Vertex Compute: 2046
      Remaining Memory: 138
      Allocation Counter: 17
      Want Split Buffer Count: 12739
      Splitting Buffer Count: 7
      Want Merge Buffer Count: 0
      Merging Bisector Count: 0
      Num Allocated Blocks:  130934
Frame n (Present 25084):
Vertex Compute:
  Dispatch:
    Draw call: 392919 verts
    Dispatch Split: 200
    Dispatch Allocate: 2
    Dispatch Vertex Compute: 2047
    Remaining Memory: 80
    Allocation Counter: 30
    Want Split Buffer Count: 12734
    Splitting Buffer Count: 99
    Want Merge Buffer Count: 0
    Merging Bisector Count: 0
    Num Allocated Blocks:  130973
Draw call:
  Push constants:
  Dispatch:
    Draw call: 392976 verts
    Dispatch Split: 199
    Dispatch Allocate: 1
    Dispatch Vertex Compute: 2047
    Remaining Memory: 80
    Allocation Counter: 30
    Want Split Buffer Count: 12734
    Splitting Buffer Count: 11
    Want Merge Buffer Count: 0
    Merging Bisector Count: 0
    Num Allocated Blocks 130992



  Draw call: 392976 verts
  Vertex buffer: 0x130910660
  Observations:
  - Vertices 392937 to 392975 are part of triangles that share a vertex with a hole
  - 39 Vertices or 13 triangles

lots of holes is not from not using allocated children
holes are consistent between frames

```c++
inline T spvFindUMSB(T x)
{
    return select(clz(T(0)) - (clz(x) + T(1)), T(-1), x == T(0));
}

inline uint spvFindUMSB(uint x)
{
    return select(clz(0) - (clz(x) + 1), UINT_MAX, x == T(0));
}

inline uint spvFindUMSB(uint x)
{
    return select(32 - clz(x) + 1, UINT_MAX, x == T(0));
}
```
- this is a dead end. seems like a bug related to writing to a uniform buffer


# Session 6
heapid buffer: 0x147e545a0
Vertex Compute 4635
- Thread 43
  - Split code: 6 = 00110
  - Verts: [0, 66.66, 0], [-12.5, 54.1, 0], [0, 50, 0]


Present 4750
- all good
- Triangle ID 68 is present w/ vtxes 129, 130, 131
  - vid/3 = 129/3 = 43
  - vertex_buffer[129] = [75, 25, 0]
  - vertex_buffer[130] = [16, 0, -25]
  - vertex_buffer[131] = [0, -12.5, -70.8] WHAT THE FUCK IS THISSS

Classify 4788
- Thread 43
  - curr_id is 68
- heap_id_buffer[68] = 1_000_0000110
  - should be 1_100_0010
  - wait now it's 1_010_00110

Present 5173
- missing triangles
- nontrianglular faces
- previous triangle id 68 is absent


Okay let's think through this.
Symptoms:
- SOMETIMES, black triangles appear in the RENDERED VIEW AFTER we are at SUBDIVISION EQUILIBRIUM
- EVERY TIME, nontriangular faces and black triangles appear in the DEBUG VIEW AFTER ~12 levels of subdivision
- triangle vertices represent heapid buffer
- longer time between frames reduces the chance of the bug appearing?
  - seems to still happen when manually stepping through
- bug increases in likelihood when all bisectors wants to split at once
- from our validation, it seems that all faces are triangular
  - furthermore, these neighbors must either exist in the vertex buffer or be invalid
  - increasing vertex draw count does not seem to solve anything
  - therefore, the neighbors are probably invalid
    - let's check using the fragment shader
    - the neighbors are INVALID
How many iterations?
- 30, 19, 30, 34, 20, 26, 34, 29


# Double buffering
                    - Classify
                    - Split
                    - Allocate
bisector lifetime |- Update pointers
                  |- CBT reduce
                  |- Vertex compute ---| Vertex Lifetime
                  |- Render            |
                  |- Reset             |
                  |- Classify          |
                  |- Split             |
                  |- Allocate          |
                    - Update pointers  |
                    - CBT reduce ------|
                    - Vertex compute
                    - Render
                    - Reset
Goal: Do not mutate any data in place.

Should we use the bisector lifetime or the vertex lifetime?
- vertices depend on the bisectors, but not vice versa
- we should use bisector lifetime, vertices are DOWNSTREAM of bisectors

- there is only 1 vertex buffer, which we recompute IF the bisectors change
  - BUT we compute curr_id buffer at the same stage which we need to store
  - easier to just store both
- then we use the curr_id buffer


Some buffers cannot be reconstructed statelessly and have to be copied over
- neighbors buffer
- heapid buffer

Is there an easy way to swap between single and multi buffering?
- we define the length of the bisector data buffer at comptime
- only copy buffers if length > 1

# Ring Buffer

```
[a, b, c]
 ^num
 ^curr

[a, b, c]
    ^num
    ^curr

[a, b, c]
       ^num
       ^curr
```
WAIT IM STUPID
```odin
// BisectorData is a smallish struct holding array pointers but is >16 bytes
bisector_data : [NUM_STORED]BisectorData
bisector_data[0] = initialize_bisector_data()
num_iters := 0 // number of iterations that have been COMPLETED
curr_iter := 0 // which iter to render
for {
    advance_iter := process_input(&curr_iter) or_break
    if (advance_iter) {
        prev_idx := num_iters % NUM_STORED
        next_idx := (num_iters + 1) % NUM_STORED
        defer {
            num_iters = next_idx
            curr_iter = next_idx
        }

        prev_bisectors := &bisector_data[prev_idx]
        next_bisectors := &bisector_data[next_idx]
        bisector_deep_copy(src=prev_bisectors, dst=next_bisectors)
        compute_next_iter_in_place(next_bisectors)
    }
    render(bisector_data[curr_iter])
}
```
GOD I LOVE ODIN

Okay so we want to view at some data from the previous generation of bisectors
- which bisectors wanna split?

```
bisectors[0].want_split
bisectors[0].vertices <----
                          | is only valid with this data
                          |
bisectors[1].want_split---|
bisectors[1].vertices
```
implementation options:
- pass in a future pointer (may be invalid)
- pass in a past pointer (almost never invalid)
- implement a ring buffer

```odin
// ring buffer
CBTScene :: struct {...}
Scenes :: [NUM_STORED]CBTScene
SceneRefs :: [NUM_STORED]^CBTScene
scenes : Buffer(Scenes) = allocate_buffer(Scenes)
scene_refs : Buffer(SceneRefs) = allocate_buffer(SceneRefs)

for scene, i in scenes {
    scene_refs[i] = get_buffer_address(scene)
}
```

hmmm there seems to be two different notions here:
- data required to construct the vertices/cbt/neighbors of a bisector generation
- data derived from the vertices/cbt/neighbors of a bisector generation

```odin
BisectorData :: struct {
    vertices : [3]f32
    neighbors : [3]u32
    heapid : HeapID
}
BisectorCBT :: struct {
    interior : ..
    leaves : ..
    bisectors : #soa[MAX_BISECTORS]BisectorData
}
BisectorClassifications {
   want_split : #dyanamic[]u32
   splitting: #dyanamic[]u32
   want_merge: #dyanamic[]u32
   merging : #dynamic[]u32
}
BisectorIterationState :: struct {
    command : u32,
    allocation_indices : [4]u32
}
BisectorIterationData {
    state : #soa[MAX_BISECTORS]BisectorIterationState
    classifications : BisectorClassifications
}

iteration_data : [NUM_STORED]BisectorIterationData
bisector_data : [NUM_STORED]BisectorCBT

bisector_data[0] = initialize_bisector_data()
num_iters := 0 // number of iterations that have been COMPLETED
curr_iter := 0 // which iter to render
for {
    advance_iter := process_input(&curr_iter) or_break
    if (advance_iter) {
        prev_idx := num_iters % NUM_STORED
        next_idx := (num_iters + 1) % NUM_STORED
        defer {
            num_iters = next_idx
            curr_iter = next_idx
        }

        prev_bisectors := &bisector_data[prev_idx]
        next_bisectors := &bisector_data[next_idx]
        next_iterdata := reset_iter_data(&iteration_data[next_idx])
        if NUM_STORED > 1 do bisector_deep_copy(src=prev_bisectors, dst=next_bisectors)
        // wait what happens to neighbors?
        // fuck
        compute_next_iterdata(next_iterdata, next_bisectors^)
        compute_next_iter_in_place(next_bisectors, next_iterdata^)
    }
    render(bisector_data[curr_iter])
}
```
ok nevermind
it seems difficult to disentangle "stable" and "unstable" data in the pipeline
the update pointers method happens in the middle, requiring changing the neighbors
then, we only need to mutate next_iterdata for deciding what to merge
then, we update `next_bisectors` mutably again
what if we separate splitting and merging queues
- that makes data very "fine" in granularity

```odin
prev_bisectors := &bisector_data[prev_idx]
next_bisectors := &bisector_data[next_idx]
if NUM_STORED > 1 do bisector_deep_copy(src=prev_bisectors, dst=next_bisectors)
command_iterdata := command_iterdata(next_bisectors)
split_iterdata := split_iterdata(next_bisectors)
split_bisectors(&next_bisectors, split_iterdata)
merge_iterdata := merge_iterdata(next_bisectors)
merge_bisectors(&next_bisectors, merge_iterdata)
cbt_reduce(&next_bisectors)
```
this shit is too complicated
we have a record anyway, let's just point to it
this only matters when rendering

```odin
RenderingMode :: enum {
    DEFAULT,
    DEBUG_WANT_SPLIT,
}
// BisectorData is a smallish struct holding array pointers but is >16 bytes
bisector_data : [NUM_STORED]BisectorData
bisector_data[0] = initialize_bisector_data()
num_iters := 0 // number of iterations that have been COMPLETED
curr_iter := 0 // which iter to render
for {
    advance_iter, rendering_mode := process_input(&curr_iter) or_break
    if NUM_STORED == 1 do rendering_mode = .DEFAULT
    if (advance_iter) {
        prev_idx := num_iters % NUM_STORED
        next_idx := (num_iters + 1) % NUM_STORED
        defer {
            num_iters = next_idx
            curr_iter = next_idx
        }

        prev_bisectors := &bisector_data[prev_idx]
        next_bisectors := &bisector_data[next_idx]
        bisector_deep_copy(src=prev_bisectors, dst=next_bisectors)
        compute_next_iter_in_place(next_bisectors)
    }
    prev_iter := max(0, curr_iter - 1) % NUM_STORED
    render(bisector_data[curr_iter], bisector_data[prev_iter], rendering_mode)
}
```
we're out of push constants
okay
it's time for uniform buffers
jk no we're not
now the pc buffer is full

i think we can simplify this further
there is a specific field that we need to display
only one im pretty sure
so what if we pass that in as a pointer?
alternatively, we pass in the old vertex buffer as a pointer
- this is more likely to be valid
- also would need a pointer to the old curr_id buffer and old dispatch buffer for the correct draw call

# Scalar Layout vs C Layout
Very similar
| Scalar | C |
| Scalars of size N have alignment N | Scalars of size N<=16 typically have alignment N on 64-bit systems but this is not spec-garaunteed |
| Matrices have the alignment of their elements | Matrices in common LA libraries are 16-byte aligned for SIMD |

WARNING: Matrices in LA libraries and matrices in Slang have the same in-memory representation, but may have DIFFERENT alignment.

# Finally we have double buffering correct
Okay so
It seems like the "holes" are triangles that don't allocate.
Their neighbors always seem to allocate.
They always seem to be placed such that either
1. They are in the compatibility chain of a neighbor that splits.
2. They are in the compatibility chain of a "hole" that satisfies (1)
3. They are in the compatibility chain of a "hole" that satisfies (2)

Let's visualize their commands to see what else we can gather.

Okay so it seems like they're not even being told to split.
- wait no sometimes they are, but they don't allocate?
  - maybe an issue with the find_zeros function
seems like we're getting contradictory info so let's keep testing

AHA it seems like we're overflowing the memory available.
But why?
This should have been prevented. HMMM

IT WAS THIS LINE
```c++
int max_required_memory = (twin_id == INVALID_INDEX) ? 2 // boundary
      : (neighbors_buffer[twin_id][TWIN] == curr_id) ? 4 // perfect matching
      : 4 * (current_depth - base_depth) + 3;            // triple split for each entry in compatibility chain
    //  ^ THIS WAS 3
```
It needs to be 4 because a triple split is 4 memory allocations
NEVERMIND a bisector never needs to triple split something
at most, it must double split

AHA I FOUND THE ACTUAL BUG FUCK MY LIFE

```c++
// split_element.slang
...
while(true) {
    if (...) {
       ...
    } else {
       ...
    }
    // SHOULD BE OUTSIDE THE LOOP
    pc.dispatch_next.remaining_memory_count.add(max_required_memory - used_memory);
}
```
