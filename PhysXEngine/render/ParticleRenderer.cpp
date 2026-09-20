#include "ParticleRenderer.hpp"
#include "Shader.hpp"

#include <epoxy/gl.h>
#include <glm/gtc/type_ptr.hpp>

namespace metis
{
namespace
{
    const char* kVertexSource = R"(#version 460 core
layout(location = 0) in vec4 inPosMass;

uniform mat4 uViewProjection;
uniform float uPointScale;

out float vMass;

void main()
{
    gl_Position = uViewProjection * vec4(inPosMass.xyz, 1.0);
    gl_PointSize = max(uPointScale / max(gl_Position.w, 1e-6), 1.0);
    vMass = inPosMass.w;
}
)";

    const char* kFragmentSource = R"(#version 460 core
in float vMass;
out vec4 fragColor;

void main()
{
    // Round sprite with a soft edge.
    vec2 d = gl_PointCoord - vec2(0.5);
    float r2 = dot(d, d);
    if (r2 > 0.25) discard;

    float falloff = 1.0 - smoothstep(0.0, 0.25, r2);
    vec3 tint = mix(vec3(0.35, 0.55, 1.0), vec3(1.0, 0.85, 0.55),
                    clamp(log(vMass + 1.0) * 0.02, 0.0, 1.0));
    fragColor = vec4(tint * falloff, falloff);
}
)";
}

ParticleRenderer::ParticleRenderer(std::size_t capacity)
{
    glGenVertexArrays(1, &vao_);
    glGenBuffers(1, &vbo_);

    glBindVertexArray(vao_);
    glBindBuffer(GL_ARRAY_BUFFER, vbo_);
    glBufferData(GL_ARRAY_BUFFER,
                 static_cast<GLsizeiptr>(capacity * 4 * sizeof(float)),
                 nullptr, GL_DYNAMIC_DRAW);

    glEnableVertexAttribArray(0);
    glVertexAttribPointer(0, 4, GL_FLOAT, GL_FALSE, 4 * sizeof(float), nullptr);

    glBindVertexArray(0);
    glBindBuffer(GL_ARRAY_BUFFER, 0);

    shader_ = std::make_unique<Shader>(kVertexSource, kFragmentSource);
}

ParticleRenderer::~ParticleRenderer()
{
    if (vbo_) glDeleteBuffers(1, &vbo_);
    if (vao_) glDeleteVertexArrays(1, &vao_);
}

void ParticleRenderer::draw(const Camera& camera, std::size_t count, float pointScale)
{
    glEnable(GL_PROGRAM_POINT_SIZE);
    glEnable(GL_BLEND);
    glBlendFunc(GL_SRC_ALPHA, GL_ONE);   // additive
    glDepthMask(GL_FALSE);

    shader_->use();
    const glm::mat4 vp = camera.viewProjection();
    glUniformMatrix4fv(shader_->uniform("uViewProjection"), 1, GL_FALSE, glm::value_ptr(vp));
    glUniform1f(shader_->uniform("uPointScale"), pointScale);

    glBindVertexArray(vao_);
    glDrawArrays(GL_POINTS, 0, static_cast<GLsizei>(count));
    glBindVertexArray(0);

    glDepthMask(GL_TRUE);
    glDisable(GL_BLEND);
}

} // namespace metis
