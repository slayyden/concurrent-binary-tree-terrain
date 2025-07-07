#version 450

//shader input
layout(location = 0) in vec3 in_color;

//output write
layout(location = 0) out vec4 out_color;

void main() {
    out_color = vec4(in_color, 1);
}
