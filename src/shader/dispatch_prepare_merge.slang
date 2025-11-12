#include "common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void dispatch_prepare_merge()
{
    let dispatch = pc.uniform_buffer.dispatch_next;
    dispatch.dispatch_prepare_merge_command.x = div_ceil(
        dispatch.want_merge_buffer_count.load(),
        WORKGROUP_SIZE);
}
