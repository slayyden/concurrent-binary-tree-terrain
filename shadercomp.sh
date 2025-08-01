SLANGC=$VULKAN_SDK/bin/slangc
FLAGS="-fvk-use-scalar-layout -matrix-layout-row-major -target spirv"

$SLANGC src/shader/test.slang $FLAGS -entry vertexMain -o src/shader/vert.spv
$SLANGC src/shader/classify.slang $FLAGS -entry classify -o src/shader/classify.spv
$SLANGC src/shader/allocate.slang $FLAGS -entry allocate -o src/shader/allocate.spv
$SLANGC src/shader/cbt_reduce.slang $FLAGS -entry reduce -o src/shader/reduce.spv
$SLANGC src/shader/split_element.slang $FLAGS -entry split_element -o src/shader/split_element.spv
$SLANGC src/shader/update_pointers.slang $FLAGS -entry update_pointers -o src/shader/update_pointers.spv
$SLANGC src/shader/prepare_merge.slang $FLAGS -entry prepare_merge -o src/shader/prepare_merge.spv
$SLANGC src/shader/merge.slang $FLAGS -entry merge -o src/shader/merge.spv
$SLANGC src/shader/reset.slang $FLAGS -entry reset -o src/shader/reset.spv
