#include <cmath>
#include <cstdio>
#include "metis/GravityCpu.hpp"
#include "metis/LeapfrogKdk.hpp"
#include "metis/Physics.hpp"
#include "metis/Scenario.hpp"
#include "metis/SemiImplicitEuler.hpp"

// Prints the convergence table in docs/design/01-gpu-engine.md: Earth-Moon, t=3600 s,
// error in the Moon's y position against the analytic orbit.
namespace
{
    constexpr double kEarthMass = 5.97e24;
    constexpr double kMoonMass  = 7.34e22;
    constexpr double kSeparation = 3.844e8;

    double analyticMoonY(double t)
    {
        const double M = kEarthMass + kMoonMass;
        const double omega = std::sqrt(metis::physics::G * M / kSeparation) / kSeparation;
        return kMoonMass * kSeparation / M + (kEarthMass / M) * kSeparation * std::cos(omega * t);
    }

    template <class Integrator>
    double moonError(double dt, double totalTime)
    {
        metis::SimState s = metis::scenario::earthMoon();
        s.softening = 1.0e4;
        metis::GravityCpu forces;
        Integrator integrator;
        const long steps = std::lround(totalTime / dt);
        for (long i = 0; i < steps; ++i) integrator.step(s, forces, dt);
        return std::abs(s.py[1] - analyticMoonY(steps * dt));
    }
}

int main()
{
    std::printf("dt        euler_m         leapfrog_m\n");
    for (double dt : {400.0, 200.0, 100.0, 50.0, 25.0, 10.0, 1.0, 0.1, 0.01}) {
        std::printf("%-8g  %-14.7g  %.7g\n", dt,
                    moonError<metis::SemiImplicitEuler>(dt, 3600.0),
                    moonError<metis::LeapfrogKdk>(dt, 3600.0));
    }
}
