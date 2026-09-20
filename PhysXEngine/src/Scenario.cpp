#include "metis/Scenario.hpp"
#include "metis/Physics.hpp"
#include <cmath>
#include <random>

namespace metis::scenario
{
    SimState earthMoon()
    {
        constexpr double earthMass = 5.97e24;
        constexpr double moonMass  = 7.34e22;
        constexpr double separation = 3.844e8;
        const double M = earthMass + moonMass;
        const double vRel = std::sqrt(physics::G * M / separation);

        SimState s;
        s.resize(2);
        s.mass[0] = earthMass;
        s.mass[1] = moonMass;
        s.py[1] = separation;
        s.vx[0] = -vRel * moonMass / M;
        s.vx[1] =  vRel * earthMass / M;
        return s;
    }

    SimState randomCloud(std::size_t n, double radius, double totalMass, std::uint32_t seed)
    {
        std::mt19937 rng(seed);
        std::uniform_real_distribution<double> uni(-1.0, 1.0);

        SimState s;
        s.resize(n);
        s.softening = radius * 1e-3;
        for (std::size_t i = 0; i < n; ++i) {
            double x, y, z, r2;
            do {
                x = uni(rng); y = uni(rng); z = uni(rng);
                r2 = x*x + y*y + z*z;
            } while (r2 > 1.0 || r2 < 1e-12);
            s.px[i] = x * radius;
            s.py[i] = y * radius;
            s.pz[i] = z * radius;
            s.mass[i] = totalMass / static_cast<double>(n);
        }
        return s;
    }
}
