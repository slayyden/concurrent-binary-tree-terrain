# About
This is an implementation of [Concurrent Binary Trees for Large Scale Game Components](https://ggx-research.github.io/publication/2024/07/27/publication-cbt.html) using Rust, Vulkan 1.3, and Slang. By using a concurrent binary tree as a memory manager, we can allow for granular per-triangle memory allocation with stable pointers. Using atomics, this allows triangles to split, merge, and update neighbors independently for extremely granular per-triangle level of detail for large scale game components, which allows for massive scene scales at the cost of additional synchronization compared to tesselation.

Please see `stress_test.mp4` for a test with constant triangle splitting and merging.

# Building
Shaders are already compiled into spirv.
Build with `cargo run`.
MacOS systems need to install MoltenVK.
You may need to remove Vulkan compatibility extensions for this to work outside of Molten-VK.

You may need to remove references to `layer_names_raw` in `main.rs` if you do not have Vulkan Validation layers.

# Controls
- Mouse to look around.
- WASD to move
- R to begin splitting
