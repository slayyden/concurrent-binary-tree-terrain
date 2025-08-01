SLANGC=$VULKAN_SDK/bin/slangc

$SLANGC src/shader/test.slang -fvk-use-scalar-layout -matrix-layout-column-major -target spirv -entry vertexMain -o src/shader/vert.spv
$SLANGC src/shader/allocate.slang -fvk-use-scalar-layout -target spirv -entry allocate -o src/shader/allocate.spv
$SLANGC src/shader/cbt_reduce.slang -fvk-use-scalar-layout -target spirv -entry reduce -o src/shader/reduce.spv
$SLANGC src/shader/split_element.slang -fvk-use-scalar-layout -target spirv -entry split_element -o src/shader/split_element.spv
$SLANGC src/shader/update_pointers.slang -fvk-use-scalar-layout -target spirv -entry update_pointers -o src/shader/update_pointers.spv
$SLANGC src/shader/prepare_merge.slang -fvk-use-scalar-layout -target spirv -entry prepare_merge -o src/shader/prepare_merge.spv
$SLANGC src/shader/merge.slang -fvk-use-scalar-layout -target spirv -entry merge -o src/shader/merge.spv
