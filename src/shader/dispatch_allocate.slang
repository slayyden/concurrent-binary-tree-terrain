#include "common.slang"

[shader("compute")]
[numthreads(1, 1, 1)]
void dispatch_allocate()
{
    pc.dispatch_next.dispatch_allocate_command.x = div_ceil(
        pc.dispatch_next.splitting_buffer_count.load(),
        WORKGROUP_SIZE);
}
