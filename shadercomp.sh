SLANGC=$VULKAN_SDK/bin/slangc

$SLANGC src/shader/test.slang -target spirv -entry vertexMain -o src/shader/vert.spv
$SLANGC src/shader/cbt_reduce.slang -target spirv -entry reduce -o src/shader/reduce.spv
