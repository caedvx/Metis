#include "metis/Diagnostics.hpp"
#include "metis/Physics.hpp"
#include <cmath>

namespace metis::diagnostics
{
    double totalEnergy(const SimState& s)
    {
        const std::size_t n = s.size();
        const double eps2 = s.softening * s.softening;
        double kinetic = 0.0;
        double potential = 0.0;

        for (std::size_t i = 0; i < n; ++i) {
            kinetic += 0.5 * s.mass[i] *
                (s.vx[i]*s.vx[i] + s.vy[i]*s.vy[i] + s.vz[i]*s.vz[i]);

            for (std::size_t j = i + 1; j < n; ++j) {
                const double dx = s.px[j] - s.px[i];
                const double dy = s.py[j] - s.py[i];
                const double dz = s.pz[j] - s.pz[i];
                const double soft = std::sqrt(dx*dx + dy*dy + dz*dz + eps2);
                potential -= physics::G * s.mass[i] * s.mass[j] / soft;
            }
        }
        return kinetic + potential;
    }

    void totalMomentum(const SimState& s, double& px, double& py, double& pz)
    {
        px = py = pz = 0.0;
        for (std::size_t i = 0; i < s.size(); ++i) {
            px += s.mass[i] * s.vx[i];
            py += s.mass[i] * s.vy[i];
            pz += s.mass[i] * s.vz[i];
        }
    }
}
