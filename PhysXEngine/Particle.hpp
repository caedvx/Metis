#pragma once
#include <ostream>
#include <vector>

struct Vector3
{
    double x, y, z;

    Vector3& operator*=(double scalar) 
    {
        x *= scalar;
        y *= scalar;
        z *= scalar;
        return *this;
    }

    Vector3& operator+=(const Vector3& other) 
    {
        x += other.x;
        y += other.y;
        z += other.z;
        return *this;
    }
};

inline Vector3 operator+(Vector3 lhs, const Vector3& rhs) 
{
    lhs += rhs;
    return lhs;
}

inline Vector3 operator*(Vector3 vec, double scalar) 
{
    vec *= scalar;
    return vec;
}

inline Vector3 operator*(double scalar, Vector3 vec) 
{
    vec *= scalar;
    return vec;
}

inline std::ostream& operator<<(std::ostream& os, const Vector3& vec) 
{
    // Format: (x, y, z) - clean and scannable for debugging logs
    os << "(" << vec.x << ", " << vec.y << ", " << vec.z << ")";
    return os;
}

struct Particle
{
    int id;

    Vector3 position;
    Vector3 velocity;

    double mass;
    
    bool isActive;

    bool operator<(const Particle& other) const
    {
        return id < other.id;
    }
};

namespace physics
{
    constexpr double G = 6.674e-11;

    // Plummer softening length (m). Caps the force at close approach so a fixed
    // dt cannot inject energy during a near-collision.
    constexpr double softening = 1.0e4;
}

Vector3 calculateGravitationalForce(const Particle& p1, const Particle& p2);
Vector3 applyForce(const Particle& particle, const Vector3& force, double dt);

// Diagnostics: both must stay flat if the integration is sound.
double totalEnergy(const std::vector<Particle>& particles);
Vector3 totalMomentum(const std::vector<Particle>& particles);
