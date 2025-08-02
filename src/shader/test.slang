#include "common.slang"

// Vertex shader to pixel shader
struct v2p
{
    float3 color : COLOR;
    float4 sv_position : SV_Position; // Specify the position of a vertex in screen space (after projection).
};

[shader("vertex")]
v2p vertexMain(uint vid: SV_VertexID)
{
    const float3 colors[3] = float3[3](
        float3(1.0f, 0.0f, 0.0f), // red
        float3(0.0f, 1.0f, 0.0f), // green
        float3(0.0f, 0.0f, 1.0f)  // blue
    );
    v2p output;
    // uint32_t channel = pushConst.curr_id[vid / 3] / 8;
    // float3 color = float3(0.0, 0.0, 0.0);
    // color[channel] = ((pushConst.curr_id[vid / 3] % 8) + 1) / 8.0;

    let curr_id = pc.scene.currid_buffer[vid / 3];
    // let curr_id = vid / 3;
    let state = pc.scene.bisector_state_buffer[curr_id];
    var color = float3(1.0, 0.0, 0.0);
    if (state == TriState::BISECT_ELEMENT)
    {
        color = float3(0.0, 1.0, 0.0);
    }
    else if (state == TriState::SIMPLIFY_ELEMENT)
    {
        color = float3(0.0, 0.0, 1.0);
    }
    else if (state == TriState::BACK_FACE_CULLED)
    {

        color = float3(1.0, 0.0, 1.0);
    }
    float4 pos = float4(pc.scene.vertex_buffer[vid], 1);
    // pos.y = -pos.y;
    // pos = mul(pos, pc.view_project);
    pos = mul(pos, pc.view_project);
    // output.sv_position = mul(pos, pc.view_project);
    output.sv_position = pos;
    output.color = color;

    return output;
}

// Pixel shader
[shader("pixel")]
float4 fragmentMain(v2p input) : SV_Target
{
    return float4(input.color, 1.0);
}
