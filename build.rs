// build.rs – Copy specific Vulkan dylibs into the target profile directory.
// This script runs before compilation and is executed by Cargo for every build.
//
// Prerequisites:
//   * The environment variable `VULKAN_SDK` must be set to the root of your Vulkan SDK installation.
//   * The three libraries you want to ship are:
//
//       libvulkan.1.4.321.dylib
//       libvulkan.1.dylib
//       libvulkan.dylib
//
// The script will copy each file (if present) from `$VULKAN_SDK/lib` into
// `target/<profile>/`, where `<profile>` is `debug` or `release`.
//
// Note: If any of the files are missing, Cargo will fail with a clear message.

use std::{
    env,
    fs::{self, copy},
    path::PathBuf,
};

fn main() {
    // add executable directory to rpath
    // from: https://github.com/Rust-SDL2/rust-sdl2
    #[cfg(target_os = "macos")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,@loader_path");

    #[cfg(target_os = "linux")]
    println!("cargo:rustc-link-arg=-Wl,-rpath,$ORIGIN");

    // Resolve target output directory
    let target_dir =
        PathBuf::from(env::var("CARGO_TARGET_DIR").unwrap_or_else(|_| "target".into()));
    let profile = env::var("PROFILE").expect("`PROFILE` environment variable not set");
    let dest_dir = target_dir.join(&profile);

    // Locate Vulkan SDK directory
    let vk_sdk = env::var("VULKAN_SDK")
        .expect("$VULKAN_SDK must be set to the root of your Vulkan SDK installation");

    let lib_dir = PathBuf::from(vk_sdk).join("lib");
    if !lib_dir.is_dir() {
        panic!(
            "Expected VULKAN_SDK/lib directory at `{}` but it does not exist",
            lib_dir.display()
        );
    }

    // List the specific dylibs we want to copy
    let files_to_copy = [
        "libvulkan.1.4.321.dylib",
        "libvulkan.1.dylib",
        "libvulkan.dylib",
    ];

    // Ensure destination directory exists
    fs::create_dir_all(&dest_dir).expect("Failed to create target profile directory");

    for file_name in &files_to_copy {
        let src_path = lib_dir.join(file_name);
        if !src_path.is_file() {
            panic!(
                "Requested Vulkan library `{}` not found at `{}`",
                file_name,
                src_path.display()
            );
        }

        // Instruct Cargo to re-run this script if the source file changes
        println!("cargo:rerun-if-changed={}", src_path.display());

        let dst_path = dest_dir.join(file_name);
        copy(&src_path, &dst_path).unwrap_or_else(|e| {
            panic!(
                "Failed to copy `{}` -> `{}`: {}",
                file_name,
                dst_path.display(),
                e
            )
        });

        println!(
            "Copied {} → {}",
            src_path.file_name().unwrap().to_string_lossy(),
            dst_path.to_str().unwrap()
        );
    }
}
