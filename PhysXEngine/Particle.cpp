#include "Particle.hpp"
#include <cmath>

Vector3 calculateGravitationalForce(const Particle& p1, const Particle& p2)
{
    double dx = p2.position.x - p1.position.x;
    double dy = p2.position.y - p1.position.y;
    double dz = p2.position.z - p1.position.z;

    double r2 = dx * dx + dy * dy + dz * dz;
    double eps = physics::softening;

    // Plummer: F = G*m1*m2 * d / (r^2 + eps^2)^(3/2). Pairs exactly with the
    // potential used in totalEnergy, so energy drift measures the integrator
    // and nothing else. Also removes the r == 0 singularity outright.
    double soft2 = r2 + eps * eps;
    double denom = soft2 * std::sqrt(soft2);      // (r^2+eps^2)^(3/2), no pow()
    double scale = (physics::G * p1.mass * p2.mass) / denom;

    return { dx * scale, dy * scale, dz * scale };
}

Vector3 applyForce(const Particle& particle, const Vector3& force, double dt)
{
    Vector3 acceleration = force * (1.0 / particle.mass);
    Vector3 deltav = acceleration * dt;
    return deltav;
}

double totalEnergy(const std::vector<Particle>& particles)
{
    double kinetic = 0.0;
    double potential = 0.0;
    double eps = physics::softening;

    for (size_t i = 0; i < particles.size(); ++i)
    {
        if (!particles[i].isActive) continue;

        const Vector3& v = particles[i].velocity;
        kinetic += 0.5 * particles[i].mass * (v.x * v.x + v.y * v.y + v.z * v.z);

        for (size_t j = i + 1; j < particles.size(); ++j)
        {
            if (!particles[j].isActive) continue;

            double dx = particles[j].position.x - particles[i].position.x;
            double dy = particles[j].position.y - particles[i].position.y;
            double dz = particles[j].position.z - particles[i].position.z;

            double soft = std::sqrt(dx * dx + dy * dy + dz * dz + eps * eps);
            potential -= (physics::G * particles[i].mass * particles[j].mass) / soft;
        }
    }

    return kinetic + potential;
}

Vector3 totalMomentum(const std::vector<Particle>& particles)
{
    Vector3 p = {0.0, 0.0, 0.0};

    for (const Particle& particle : particles)
    {
        if (!particle.isActive) continue;
        p += particle.velocity * particle.mass;
    }

    return p;
}
