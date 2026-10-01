#pragma once
#include <glm/glm.hpp>

namespace metis
{
    // Orbit camera: azimuth/elevation around a target at a given distance.
    class Camera
    {
    public:
        void orbit(float deltaAzimuth, float deltaElevation);
        void zoom(float factor);
        void pan(float dx, float dy);

        void setAspect(float aspect) { aspect_ = aspect; }
        void setDistance(float d) { distance_ = d; }

        glm::mat4 viewProjection() const;
        glm::vec3 position() const;
        float distance() const { return distance_; }

    private:
        glm::vec3 target_{0.0f};
        float distance_ = 1.0f;
        float azimuth_ = 0.0f;
        float elevation_ = 0.3f;
        float aspect_ = 16.0f / 9.0f;
        float fov_ = 1.0472f;   // 60 degrees
    };
}
