#include "common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void dispatch_allocate()
{
    let dispatch = pc.uniform_buffer.dispatch_next;
    dispatch.dispatch_allocate_command.x = div_ceil(
        dispatch.splitting_buffer_count.load(),
        WORKGROUP_SIZE);
}
