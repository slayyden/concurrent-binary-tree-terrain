struct PushConstant
{
    float4x4 view_project;
    float *positions;
}

[[vk::push_constant]]
ConstantBuffer<PushConstant> pushConst;

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
    output.color = colors[vid % 3];
    output.sv_position = float4(
        pushConst.positions[vid * 3],
        pushConst.positions[vid * 3 + 1],
        pushConst.positions[vid * 3 + 2],
        1.0);

    return output;
}

// Pixel shader
[shader("pixel")]
float4 fragmentMain(v2p input) : SV_Target
{
    return float4(input.color, 1.0);
}
