#include "cbt.slang"

// Vertex shader to pixel shader
struct v2p
{
    float3 color : COLOR;
    float4 sv_position : SV_Position; // Specify the position of a vertex in screen space (after projection).
    uint bisector_id : BISECTOR_ID;
    uint4 user_data : USER_DATA;
};

[shader("vertex")]
v2p vertexMain(uint vid: SV_VertexID)
{
    let scene = (pc.rendering_mode == RenderingMode::DEFAULT) ? pc.scene_next : pc.scene_prev;
    let vertex_buffer = scene.vertex_buffer;
    const float3 colors[3] = float3[3](
        float3(1.0f, 0.0f, 0.0f), // red
        float3(0.0f, 1.0f, 0.0f), // green
        float3(0.0f, 0.0f, 1.0f)  // blue
    );
    v2p output;

    var color = float3(1.0, 0.0, 0.0);
    color = colors[vid % 3];
    float4 pos = float4(vertex_buffer[vid], 1);
    pos = mul(pos, pc.view_project);

    let curr_id = scene.curr_id_buffer[vid / 3];
    let neighbors = scene.neighbors_buffer[curr_id];

    if ((pc.rendering_mode == RenderingMode::DEFAULT) || (pc.rendering_mode == RenderingMode::ROLLBACK_DEFAULT)) {
        var num_valid_neighbors = 0;
        for (int i = 0; i < 3; ++i) {
            if (neighbors[i] != INVALID_INDEX) {
                num_valid_neighbors += 1;
            }
        }
        output.user_data[0] = num_valid_neighbors;
    } else if (pc.rendering_mode == RenderingMode::DBG_WANT_SPLIT) {
        output.user_data = scene.allocation_indices_buffer[curr_id];
    }

    // output.sv_position = mul(pos, pc.view_project);
    output.sv_position = pos;
    output.color = color;
    output.bisector_id = pc.scene_next.curr_id_buffer[vid / 3];



    return output;
}

// Pixel shader
[shader("pixel")]
float4 fragmentMain(v2p input) : SV_Target
{
    if (pc.rendering_mode == RenderingMode::DEFAULT) {
        return float4(0.0, 0.0, 0.0, 1.0);
    } else {
        return float4(1.0, 1.0, 1.0, 1.0);
    }

    /*
    switch (pc.rendering_mode) {
        case RenderingMode::ROLLBACK_DEFAULT: 
            return float4(1.0, 0.0, 1.0, 1.0);
        case RenderingMode::DEFAULT : {
            return float4(1.0, 1.0, 1.0, 1.0);
            let num_valid_neighbors = input.user_data[0];
            if (num_valid_neighbors == 3)
            {
                return float4(float3((float)input.bisector_id / (1 << 17)), 1.0);
            }
            else if (num_valid_neighbors == 2)
            {
                return float4(0.0, 1.0, 1.0, 1.0);
            }
            else if (num_valid_neighbors == 1)
            {
                return float4(0.0, 1.0, 0.0, 1.0);
            }
            else if (num_valid_neighbors == 0)
            {
                return float4(0.0, 0.0, 1.0, 1.0);
            }
            return float4(0.0, 0.0, 0.0, 1.0);
        }
        case RenderingMode::DBG_WANT_SPLIT : {
            // return float4(1.0, 1.0, 1.0, 1.0);
            var num_allocation_indices = 0;
            for (int i = 0; i < 4; i += 1) {
                if (input.user_data[i] != INVALID_INDEX) {
                    num_allocation_indices += 1;
                }
            }
            if (num_allocation_indices == 4) {
                return float4(1.0, 0.0, 0.0, 1.0);
            } else
            if (num_allocation_indices == 3) {
                return float4(0.0, 1.0, 0.0, 1.0);
            } else
            if (num_allocation_indices == 2) {
                return float4(0.0, 0.0, 1.0, 1.0);
            } else
            if (num_allocation_indices == 1) {
                return float4(0.5, 0.0, 0.0, 1.0);
            } else {
                return float4(1.0, 1.0, 1.0, 1.0);
            }
        }
    }
    return float4(1.0, 1.0, 0.0, 1.0);*/
}

// Pixel shader
[shader("pixel")]
float4 fragmentWire(v2p input) : SV_Target
{
    return float4(.10, .10, 0.0, 1.0);
}
