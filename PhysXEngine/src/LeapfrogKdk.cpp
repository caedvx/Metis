#include "metis/LeapfrogKdk.hpp"

namespace metis
{
    void LeapfrogKdk::step(SimState& s, IForceModel& forces, double dt)
    {
        const std::size_t n = s.size();
        const double half = 0.5 * dt;

        // Re-prime when the state or particle count changes, so a reused integrator
        // never kicks with stale accelerations.
        if (primedFor_ != &s || primedSize_ != n) {
            forces.computeAccelerations(s);
            primedFor_ = &s;
            primedSize_ = n;
        }

        // Kick (half)
        for (std::size_t i = 0; i < n; ++i) {
            s.vx[i] += s.ax[i] * half;
            s.vy[i] += s.ay[i] * half;
            s.vz[i] += s.az[i] * half;
        }

        // Drift (full)
        for (std::size_t i = 0; i < n; ++i) {
            s.px[i] += s.vx[i] * dt;
            s.py[i] += s.vy[i] * dt;
            s.pz[i] += s.vz[i] * dt;
        }

        // Recompute at the new positions
        forces.computeAccelerations(s);

        // Kick (half)
        for (std::size_t i = 0; i < n; ++i) {
            s.vx[i] += s.ax[i] * half;
            s.vy[i] += s.ay[i] * half;
            s.vz[i] += s.az[i] * half;
        }
    }
}
