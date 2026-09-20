#include "metis/SemiImplicitEuler.hpp"

namespace metis
{
    void SemiImplicitEuler::step(SimState& s, IForceModel& forces, double dt)
    {
        forces.computeAccelerations(s);
        const std::size_t n = s.size();
        for (std::size_t i = 0; i < n; ++i) {
            s.vx[i] += s.ax[i] * dt;
            s.vy[i] += s.ay[i] * dt;
            s.vz[i] += s.az[i] * dt;
            s.px[i] += s.vx[i] * dt;
            s.py[i] += s.vy[i] * dt;
            s.pz[i] += s.vz[i] * dt;
        }
    }
}
