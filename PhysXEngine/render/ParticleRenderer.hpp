#pragma once
#include <cstddef>
#include <memory>
#include "Camera.hpp"

namespace metis
{
    class Shader;

    class ParticleRenderer
    {
    public:
        explicit ParticleRenderer(std::size_t capacity);
        ~ParticleRenderer();

        ParticleRenderer(const ParticleRenderer&) = delete;
        ParticleRenderer& operator=(const ParticleRenderer&) = delete;

        // The VBO CUDA writes into. Layout is float4 per particle: xyz + mass.
        unsigned vbo() const { return vbo_; }

        void draw(const Camera& camera, std::size_t count, float pointScale);

    private:
        unsigned vbo_ = 0;
        unsigned vao_ = 0;
        std::unique_ptr<Shader> shader_;
        int uViewProjectionLoc_ = -1;
        int uPointScaleLoc_ = -1;
    };
}
