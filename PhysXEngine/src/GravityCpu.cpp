#include "metis/GravityCpu.hpp"
#include "metis/Physics.hpp"
#include <cmath>

namespace metis
{
    void GravityCpu::computeAccelerations(SimState& s)
    {
        const std::size_t n = s.size();
        const double eps2 = s.softening * s.softening;
        s.clearAccelerations();

        for (std::size_t i = 0; i < n; ++i) {
            double axi = 0.0, ayi = 0.0, azi = 0.0;
            for (std::size_t j = 0; j < n; ++j) {
                if (i == j) continue;
                const double dx = s.px[j] - s.px[i];
                const double dy = s.py[j] - s.py[i];
                const double dz = s.pz[j] - s.pz[i];
                const double soft2 = dx*dx + dy*dy + dz*dz + eps2;
                const double inv = 1.0 / (soft2 * std::sqrt(soft2));
                const double k = physics::G * s.mass[j] * inv;
                axi += dx * k; ayi += dy * k; azi += dz * k;
            }
            s.ax[i] = axi; s.ay[i] = ayi; s.az[i] = azi;
        }
    }
}
