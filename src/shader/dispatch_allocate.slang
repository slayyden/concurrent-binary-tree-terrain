#include "common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void dispatch_allocate()
{
    pc.dispatch.dispatch_allocate_command.x = div_ceil(
        pc.dispatch.splitting_buffer_count.load(),
        WORKGROUP_SIZE);
}
