#include "cbt.slang"

// Vertex shader to pixel shader
struct v2p
{
    float3 color : COLOR;
    float4 sv_position : SV_Position; // Specify the position of a vertex in screen space (after projection).
    uint bisector_id : BISECTOR_ID;
    uint4 user_data : USER_DATA;
    float tri_area : TRI_AREA;
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

    switch (pc.rendering_mode) {
        case RenderingMode::DEFAULT:
        case RenderingMode::ROLLBACK_DEFAULT: {
            var num_valid_neighbors = 0;
            for (int i = 0; i < 3; ++i) {
                if (neighbors[i] != INVALID_INDEX) {
                    num_valid_neighbors += 1;
                }
            }
            output.user_data[0] = num_valid_neighbors;
            break;
        }
        case RenderingMode::DBG_WANT_SPLIT: {
            output.user_data = pc.scene_next.allocation_indices_buffer[curr_id];
            break;
        }
        case RenderingMode::DBG_SPLIT_COMMAND: {
            output.user_data[0] = pc.scene_next.bisector_split_command_buffer[curr_id].load();
            break;
        }
    }

    let verts : float3[3] = {
        vertex_buffer[vid/3 * 3],
        vertex_buffer[vid/3 * 3 + 1],
        vertex_buffer[vid/3 * 3 + 2],
    };
    let normal = cross(verts[1] - verts[0], verts[2] - verts[1]);
    let normal_normalized = normalize(normal);
    let VdotN = dot(normal_normalized, pc.lookdir);
    output.tri_area = tri_area(verts, VdotN);


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
    return float4(
        float3(input.tri_area) * 100, 1
    );
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
            var num_allocation_indices = 0;
            for (int i = 0; i < 4; i += 1) {
                if (input.user_data[i] == INVALID_INDEX) {break;}
                num_allocation_indices += 1;
            }
            if (num_allocation_indices == 4) {
                return float4(0.0, 1.0, 0.0, 1.0);
            } else if (num_allocation_indices == 3) {
                return float4(0.0, 0.0, 1.0, 1.0);
            } else if (num_allocation_indices == 2) {
                return float4(0.5, 0.5, 0.0, 1.0);
            } else if (num_allocation_indices == 1) {
                return float4(0.0, 0.5, 0.5, 1.0);
            } else if (num_allocation_indices == 0) {
                return float4(1.0, 1.0, 1.0, 1.0);
            } else {
                return float4(1.0, 0.0, 1.0, 1.0);
            }
        }
        case RenderingMode::DBG_SPLIT_COMMAND : {
            let split_command = input.user_data[0];
            switch (split_command) {
                case TRIPLE_SPLIT: return float4(0.0, 1.0, 0.0, 1.0);
                case LEFT_DOUBLE_SPLIT: return float4(0.75, 0.75, 0.0, 1.0);
                case RIGHT_DOUBLE_SPLIT: return float4(0.0, 0.75, 0.75, 1.0);
                case CENTER_SPLIT: return float4(0.5, 0.0, 0.5, 1.0);
                case NO_SPLIT: return float4(1.0, 1.0, 1.0, 1.0);
            }
            return float4(0.0, 0.0, 0.0, 1.0);
        }
    }
    return float4(1.0, 1.0, 0.0, 1.0);
}

// Pixel shader
[shader("pixel")]
float4 fragmentWire(v2p input) : SV_Target
{
    return float4(.10, .10, 0.0, 1.0);
}
