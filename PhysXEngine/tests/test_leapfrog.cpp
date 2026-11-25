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

metis::SimState runFreshEarthMoon(metis::LeapfrogKdk& integrator, metis::IForceModel& forces,
                                   double dt, long steps)
{
    metis::SimState s = metis::scenario::earthMoon();
    s.softening = 1.0e4;
    for (long i = 0; i < steps; ++i) integrator.step(s, forces, dt);
    return s;
}

} // namespace

TEST(LeapfrogKdk, ConvergesSecondOrder)
{
    // 4x smaller dt gives ~19x less error: second order predicts 16x, first
    // order 4x. Keep dt above 50, below that the error hits the FP64 floor
    // (~1.8e-5 m) and the ratio measures rounding.
    const double coarse = leapfrogError(400.0, 3600.0);
    const double fine   = leapfrogError(100.0, 3600.0);
    EXPECT_GT(coarse / fine, 10.0)
        << "Leapfrog KDK must converge as O(dt^2), not O(dt).";
}

TEST(LeapfrogKdk, BeatsEulerAtSameTimestep)
{
    // At dt=0.1 Euler measured 0.485360 m. Leapfrog must do far better.
    EXPECT_LT(leapfrogError(0.1, 3600.0), 0.01);
}

TEST(LeapfrogKdk, ReusedInstanceMatchesFreshInstance)
{
    // A reused integrator has to re-prime on a new state. primedFor_ compares
    // addresses, so both states stay alive to keep them at distinct addresses.
    const double dt = 1.0;
    const long steps = 100;
    metis::GravityCpu forces;

    metis::LeapfrogKdk reused;
    const metis::SimState firstRun = runFreshEarthMoon(reused, forces, dt, steps);
    const metis::SimState reusedResult = runFreshEarthMoon(reused, forces, dt, steps);

    metis::LeapfrogKdk fresh;
    const metis::SimState freshResult = runFreshEarthMoon(fresh, forces, dt, steps);

    for (std::size_t i = 0; i < reusedResult.size(); ++i) {
        EXPECT_DOUBLE_EQ(reusedResult.px[i], freshResult.px[i]);
        EXPECT_DOUBLE_EQ(reusedResult.py[i], freshResult.py[i]);
        EXPECT_DOUBLE_EQ(reusedResult.pz[i], freshResult.pz[i]);
        EXPECT_DOUBLE_EQ(reusedResult.vx[i], freshResult.vx[i]);
        EXPECT_DOUBLE_EQ(reusedResult.vy[i], freshResult.vy[i]);
        EXPECT_DOUBLE_EQ(reusedResult.vz[i], freshResult.vz[i]);
    }
}
