#include "metis/SimState.hpp"
#include <algorithm>

namespace metis
{
    void SimState::resize(std::size_t n)
    {
        px.assign(n, 0.0); py.assign(n, 0.0); pz.assign(n, 0.0);
        vx.assign(n, 0.0); vy.assign(n, 0.0); vz.assign(n, 0.0);
        ax.assign(n, 0.0); ay.assign(n, 0.0); az.assign(n, 0.0);
        mass.assign(n, 0.0);
    }

    void SimState::clearAccelerations()
    {
        std::fill(ax.begin(), ax.end(), 0.0);
        std::fill(ay.begin(), ay.end(), 0.0);
        std::fill(az.begin(), az.end(), 0.0);
    }
}
