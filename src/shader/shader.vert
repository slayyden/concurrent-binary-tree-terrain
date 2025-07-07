#version 450
#extension GL_EXT_buffer_reference : require

layout(std430, buffer_reference, buffer_reference_align = 8) readonly buffer Position {
    vec2 positions[];
};

layout(push_constant) uniform Registers {
    mat4 view_project;
    Position positions;
} registers;

layout(location = 0) flat out vec3 out_color;

void main()
{
    //const array of positions for the triangle
    const vec3 positions[3] = vec3[3](
            vec3(1.f, 1.f, 0.5f),
            vec3(-1.f, 1.f, 0.5f),
            vec3(0.f, -1.f, 0.5f)
        );

    //const array of colors for the triangle
    const vec3 colors[3] = vec3[3](
            vec3(1.0f, 0.0f, 0.0f), //red
            vec3(0.0f, 1.0f, 0.0f), //green
            vec3(00.f, 0.0f, 1.0f) //blue
        );

    //output the position of each vertex
    gl_Position = vec4(positions[gl_VertexIndex], 1.0f);
    out_color = colors[gl_VertexIndex];

    // restrict Position positions = registers.positions;
    // vec2 pos = positions[gl_VertexIndex];
    // gl_Position = vec4(pos, 0, 1);
    // out_color = vec4(1, 0, 0, 1);
}
