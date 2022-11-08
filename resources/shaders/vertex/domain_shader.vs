#version 330 core

layout (location = 0) in vec3 aPos_domain;
layout (location = 1) in vec4 aColor_domain;
out vec4 aColor;

uniform mat4 model_domain;
uniform mat4 view_domain;
uniform mat4 projection_domain;

void main()
{
   gl_Position =  projection_domain * view_domain * model_domain * vec4(aPos_domain.x, aPos_domain.y, aPos_domain.z, 1.0);
   aColor = aColor_domain;
}