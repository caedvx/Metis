#include <gtest/gtest.h>
#include <cmath>
#include "metis/Diagnostics.hpp"
#include "metis/GravityCpu.hpp"
#include "metis/LeapfrogKdk.hpp"
#include "metis/Scenario.hpp"

TEST(Diagnostics, EnergyBoundedOverOrbitArc)
{
    metis::SimState s = metis::scenario::earthMoon();
    s.softening = 1.0e4;
    metis::GravityCpu forces;
    metis::LeapfrogKdk integrator;

    const double e0 = metis::diagnostics::totalEnergy(s);
    ASSERT_LT(e0, 0.0) << "Earth-Moon must start bound.";

    for (int i = 0; i < 36000; ++i) integrator.step(s, forces, 0.1);

    const double e1 = metis::diagnostics::totalEnergy(s);
    EXPECT_LT(std::abs((e1 - e0) / e0), 1e-9) << "Symplectic integrator must not drift energy.";
}

TEST(Diagnostics, MomentumConserved)
{
    metis::SimState s = metis::scenario::earthMoon();
    s.softening = 1.0e4;
    metis::GravityCpu forces;
    metis::LeapfrogKdk integrator;

    double px0, py0, pz0;
    metis::diagnostics::totalMomentum(s, px0, py0, pz0);

    for (int i = 0; i < 36000; ++i) integrator.step(s, forces, 0.1);

    double px1, py1, pz1;
    metis::diagnostics::totalMomentum(s, px1, py1, pz1);

    // Scale is set by the individual terms, which are ~1e26 and cancel.
    const double scale = s.mass[0] * std::abs(s.vx[0]);
    EXPECT_LT(std::abs(px1 - px0) / scale, 1e-12);
    EXPECT_LT(std::abs(py1 - py0) / scale, 1e-12);
}
