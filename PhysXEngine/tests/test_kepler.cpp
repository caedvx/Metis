#include <gtest/gtest.h>
#include <cmath>
#include <vector>
#include "Particle.hpp"
#include "metis/Physics.hpp"

namespace {

constexpr double kEarthMass = 5.97e24;
constexpr double kMoonMass  = 7.34e22;
constexpr double kSeparation = 3.844e8;

// Analytic two-body circular solution, barycentre at rest.
void analyticMoon(double t, double& outX, double& outY)
{
    const double M = kEarthMass + kMoonMass;
    const double vRel = std::sqrt(metis::physics::G * M / kSeparation);
    const double omega = vRel / kSeparation;
    const double comY = kMoonMass * kSeparation / M;
    outX = (kEarthMass / M) * kSeparation * std::sin(omega * t);
    outY = comY + (kEarthMass / M) * kSeparation * std::cos(omega * t);
}

std::vector<Particle> makeEarthMoon()
{
    const double M = kEarthMass + kMoonMass;
    const double vRel = std::sqrt(metis::physics::G * M / kSeparation);
    std::vector<Particle> p;
    p.push_back({1, {0.0, 0.0, 0.0}, {-vRel * kMoonMass / M, 0.0, 0.0}, kEarthMass, true});
    p.push_back({2, {0.0, kSeparation, 0.0}, { vRel * kEarthMass / M, 0.0, 0.0}, kMoonMass, true});
    return p;
}

double runAndMeasureMoonError(double dt, double totalTime)
{
    std::vector<Particle> ps = makeEarthMoon();
    std::vector<Vector3> forces(ps.size());
    const long steps = std::lround(totalTime / dt);

    for (long s = 0; s < steps; ++s) {
        for (size_t i = 0; i < ps.size(); ++i) {
            forces[i] = {0.0, 0.0, 0.0};
            for (size_t j = 0; j < ps.size(); ++j) {
                if (i == j) continue;
                forces[i] += calculateGravitationalForce(ps[i], ps[j]);
            }
        }
        for (size_t i = 0; i < ps.size(); ++i) {
            ps[i].velocity += applyForce(ps[i], forces[i], dt);
            ps[i].position += ps[i].velocity * dt;
        }
    }

    double ax, ay;
    analyticMoon(steps * dt, ax, ay);
    return std::abs(ps[1].position.y - ay);
}

} // namespace

TEST(KeplerOracle, MatchesAnalyticSolutionAtMillimetreScale)
{
    const double err = runAndMeasureMoonError(0.001, 3600.0);
    EXPECT_NEAR(err, 0.004836, 1e-5)
        << "Kepler oracle moved. Measured baseline is 0.004836 m at dt=0.001.";
}

TEST(KeplerOracle, ConvergesFirstOrder)
{
    const double coarse = runAndMeasureMoonError(0.01, 3600.0);
    const double fine   = runAndMeasureMoonError(0.001, 3600.0);
    EXPECT_NEAR(coarse / fine, 10.0, 0.5)
        << "Semi-implicit Euler must converge as O(dt).";
}
