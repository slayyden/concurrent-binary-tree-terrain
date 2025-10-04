#include "common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void dispatch_prepare_merge()
{
    pc.dispatch_next.dispatch_prepare_merge_command.x = div_ceil(
        pc.dispatch_next.want_merge_buffer_count.load(),
        WORKGROUP_SIZE);
}
