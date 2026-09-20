#include "Camera.hpp"

#include <algorithm>
#include <cmath>
#include <glm/gtc/matrix_transform.hpp>

namespace metis
{
    void Camera::orbit(float deltaAzimuth, float deltaElevation)
    {
        azimuth_ += deltaAzimuth;
        elevation_ = std::clamp(elevation_ + deltaElevation, -1.55f, 1.55f);
    }

    void Camera::zoom(float factor)
    {
        distance_ = std::max(distance_ * factor, 1e-6f);
    }

    void Camera::pan(float dx, float dy)
    {
        const glm::vec3 forward = glm::normalize(target_ - position());
        const glm::vec3 right = glm::normalize(glm::cross(forward, glm::vec3(0, 1, 0)));
        const glm::vec3 up = glm::cross(right, forward);
        target_ += (-right * dx + up * dy) * distance_;
    }

    glm::vec3 Camera::position() const
    {
        const float ce = std::cos(elevation_);
        return target_ + distance_ * glm::vec3(ce * std::cos(azimuth_),
                                               std::sin(elevation_),
                                               ce * std::sin(azimuth_));
    }

    glm::mat4 Camera::viewProjection() const
    {
        const glm::mat4 view = glm::lookAt(position(), target_, glm::vec3(0, 1, 0));
        const glm::mat4 proj = glm::perspective(fov_, aspect_,
                                                distance_ * 1e-3f, distance_ * 1e3f);
        return proj * view;
    }
}
