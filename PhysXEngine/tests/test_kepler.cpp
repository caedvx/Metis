#include <gtest/gtest.h>
#include <cmath>
#include "metis/GravityCpu.hpp"
#include "metis/Physics.hpp"
#include "metis/Scenario.hpp"
#include "metis/SemiImplicitEuler.hpp"

namespace {

constexpr double kEarthMass = 5.97e24;
constexpr double kMoonMass  = 7.34e22;
constexpr double kSeparation = 3.844e8;

void analyticMoon(double t, double& outX, double& outY)
{
    const double M = kEarthMass + kMoonMass;
    const double vRel = std::sqrt(metis::physics::G * M / kSeparation);
    const double omega = vRel / kSeparation;
    const double comY = kMoonMass * kSeparation / M;
    outX = (kEarthMass / M) * kSeparation * std::sin(omega * t);
    outY = comY + (kEarthMass / M) * kSeparation * std::cos(omega * t);
}

double runAndMeasureMoonError(double dt, double totalTime)
{
    metis::SimState s = metis::scenario::earthMoon();
    s.softening = 1.0e4;
    metis::GravityCpu forces;
    metis::SemiImplicitEuler integrator;

    const long steps = std::lround(totalTime / dt);
    for (long i = 0; i < steps; ++i) integrator.step(s, forces, dt);

    double ax, ay;
    analyticMoon(steps * dt, ax, ay);
    return std::abs(s.py[1] - ay);
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
