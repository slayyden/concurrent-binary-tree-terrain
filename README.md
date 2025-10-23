
Please see `stress_test.mp4` for a test with constant triangle splitting and merging.

Shaders are already compiled into spirv.
Build with `cargo run`.

You may need to remove references to `layer_names_raw` in `main.rs` if you do not have Vulkan Validation layers.

Controls:
WASD to move
R to begin splitting
