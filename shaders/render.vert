#version 330 core
layout(location=0) in vec3 pos;
layout(location=1) in float energy;
uniform mat4 mvp;
out vec3 vColor;
void main(){
    gl_Position = mvp * vec4(pos,1.0);
    float e = clamp(energy, 0.0, 1.0);
    vColor = vec3(e, 0.0, 1.0 - e);
}
