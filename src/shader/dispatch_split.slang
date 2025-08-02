#include "common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void dispatch_split()
{
    pc.dispatch.dispatch_split_command.x = div_ceil(
        pc.dispatch.want_split_buffer_count.load(),
        WORKGROUP_SIZE);
}
