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

namespace
{
    // Deletes the shader on every exit path out of the constructor below,
    // including a throw from the fragment compile or the link check -- vs
    // and fs would otherwise leak on those paths (the constructor never
    // completing means ~Shader never gets to clean them up either).
    struct ShaderGuard
    {
        unsigned id;
        ~ShaderGuard() { if (id) glDeleteShader(id); }
    };
}

Shader::Shader(const char* vertexSource, const char* fragmentSource)
{
    ShaderGuard vs{compile(GL_VERTEX_SHADER, vertexSource)};
    ShaderGuard fs{compile(GL_FRAGMENT_SHADER, fragmentSource)};

    program_ = glCreateProgram();
    glAttachShader(program_, vs.id);
    glAttachShader(program_, fs.id);
    glLinkProgram(program_);

    int ok = 0;
    glGetProgramiv(program_, GL_LINK_STATUS, &ok);
    if (!ok) {
        int len = 0;
        glGetProgramiv(program_, GL_INFO_LOG_LENGTH, &len);
        std::vector<char> log(len > 1 ? len : 1);
        glGetProgramInfoLog(program_, len, nullptr, log.data());
        glDeleteProgram(program_);
        program_ = 0;
        throw std::runtime_error(std::string("program link failed: ") + log.data());
    }
}

Shader::~Shader() { if (program_) glDeleteProgram(program_); }
void Shader::use() const { glUseProgram(program_); }
int Shader::uniform(const char* name) const { return glGetUniformLocation(program_, name); }

} // namespace metis
