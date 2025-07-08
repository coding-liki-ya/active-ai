#version 330 core
layout(location=0) in vec3 pos;
layout(location=1) in float energy;
uniform mat4 mvp;
uniform vec3 baseColor;
uniform float alpha;
uniform float pointSize;
out vec4 vColor;
void main(){
    gl_Position = mvp * vec4(pos,1.0);
    gl_PointSize = pointSize;
    float e = clamp(energy, 0.0, 5.0) / 5.0;
    vColor = vec4(baseColor * e, alpha * e);
}
