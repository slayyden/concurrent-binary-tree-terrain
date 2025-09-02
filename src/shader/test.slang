#include "cbt.slang"

// Vertex shader to pixel shader
struct v2p
{
    float3 color : COLOR;
    float4 sv_position : SV_Position; // Specify the position of a vertex in screen space (after projection).
    uint bisector_id : BISECTOR_ID;
    uint num_valid_neighbors : NUM_NEIGHBORS;
};

[shader("vertex")]
v2p vertexMain(uint vid: SV_VertexID)
{
    let vertex_buffer = pc.vertex_buffer;
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

    let curr_id = pc.scene.curr_id_buffer[vid / 3];
    let neighbors = pc.scene.neighbors_buffer[curr_id];
    var num_valid_neighbors = 0;
    for (int i = 0; i < 3; ++i)
    {
        let neighbor = neighbors[i];
        if (neighbor != INVALID_INDEX)
        {
            num_valid_neighbors += 1;
        }
    }

    // output.sv_position = mul(pos, pc.view_project);
    output.sv_position = pos;
    output.color = color;
    output.bisector_id = pc.scene.curr_id_buffer[vid / 3];
    output.num_valid_neighbors = num_valid_neighbors;

    return output;
}

// Pixel shader
[shader("pixel")]
float4 fragmentMain(v2p input) : SV_Target
{
    if (input.num_valid_neighbors == 3)
    {
        return float4(float3((float)input.bisector_id / (1 << 17)), 1.0);
    }
    else if (input.num_valid_neighbors == 2)
    {
        return float4(0.0, 1.0, 1.0, 1.0);
    }
    else if (input.num_valid_neighbors == 1)
    {
        return float4(0.0, 1.0, 0.0, 1.0);
    }
    else if (input.num_valid_neighbors == 0)
    {
        return float4(0.0, 0.0, 1.0, 1.0);
    }
    return float4(0.0, 0.0, 0.0, 1.0);
}

// Pixel shader
[shader("pixel")]
float4 fragmentWire(v2p input) : SV_Target
{
    return float4(.10, .10, 0.0, 1.0);
}
