#include "Shader.hpp"

#include <epoxy/gl.h>
#include <stdexcept>
#include <vector>

namespace metis
{
namespace
{
    unsigned compile(unsigned type, const char* source)
    {
        const unsigned shader = glCreateShader(type);
        glShaderSource(shader, 1, &source, nullptr);
        glCompileShader(shader);

        int ok = 0;
        glGetShaderiv(shader, GL_COMPILE_STATUS, &ok);
        if (!ok) {
            int len = 0;
            glGetShaderiv(shader, GL_INFO_LOG_LENGTH, &len);
            std::vector<char> log(len > 1 ? len : 1);
            glGetShaderInfoLog(shader, len, nullptr, log.data());
            throw std::runtime_error(std::string("shader compile failed: ") + log.data());
        }
        return shader;
    }
}

Shader::Shader(const char* vertexSource, const char* fragmentSource)
{
    const unsigned vs = compile(GL_VERTEX_SHADER, vertexSource);
    const unsigned fs = compile(GL_FRAGMENT_SHADER, fragmentSource);

    program_ = glCreateProgram();
    glAttachShader(program_, vs);
    glAttachShader(program_, fs);
    glLinkProgram(program_);

    int ok = 0;
    glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        int len = 0;
        glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? len : 1);
        glGetProgramInfoLog(program_, len, nullptr, log.data());
        throw std::runtime_error(std::string("program link failed: ") + log.data());
    }

    glDeleteShader(vs);
    glDeleteShader(fs);
}

Shader::~Shader() { if (program_) glDeleteProgram(program_); }
void Shader::use() const { glUseProgram(program_); }
int Shader::uniform(const char* name) const { return glGetUniformLocation(program_, name); }

} // namespace metis
