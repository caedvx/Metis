#pragma once
#include <string>

namespace metis
{
    class Shader
    {
    public:
        Shader(const char* vertexSource, const char* fragmentSource);
        ~Shader();

        Shader(const Shader&) = delete;
        Shader& operator=(const Shader&) = delete;

        void use() const;
        unsigned id() const { return program_; }
        int uniform(const char* name) const;

    private:
        unsigned program_ = 0;
    };
}
