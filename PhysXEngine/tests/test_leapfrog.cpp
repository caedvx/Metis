#include <gtest/gtest.h>
#include <cmath>
#include "metis/GravityCpu.hpp"
#include "metis/LeapfrogKdk.hpp"
#include "metis/Physics.hpp"
#include "metis/Scenario.hpp"

namespace {

constexpr double kEarthMass = 5.97e24;
constexpr double kMoonMass  = 7.34e22;
constexpr double kSeparation = 3.844e8;

double analyticMoonY(double t)
{
    const double M = kEarthMass + kMoonMass;
    const double vRel = std::sqrt(metis::physics::G * M / kSeparation);
    const double omega = vRel / kSeparation;
    const double comY = kMoonMass * kSeparation / M;
    return comY + (kEarthMass / M) * kSeparation * std::cos(omega * t);
}

double leapfrogError(double dt, double totalTime)
{
    metis::SimState s = metis::scenario::earthMoon();
    s.softening = 1.0e4;
    metis::GravityCpu forces;
    metis::LeapfrogKdk integrator;

    const long steps = std::lround(totalTime / dt);
    for (long i = 0; i < steps; ++i) integrator.step(s, forces, dt);

    return std::abs(s.py[1] - analyticMoonY(steps * dt));
}

} // namespace

TEST(LeapfrogKdk, ConvergesSecondOrder)
{
    const double coarse = leapfrogError(0.1, 3600.0);
    const double fine   = leapfrogError(0.01, 3600.0);
    // Second order: 10x smaller dt gives ~100x smaller error.
    EXPECT_GT(coarse / fine, 50.0)
        << "Leapfrog KDK must converge as O(dt^2), not O(dt).";
}

TEST(LeapfrogKdk, BeatsEulerAtSameTimestep)
{
    // At dt=0.1 Euler measured 0.485360 m. Leapfrog must do far better.
    EXPECT_LT(leapfrogError(0.1, 3600.0), 0.01);
}
