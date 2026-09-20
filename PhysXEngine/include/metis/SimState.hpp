#pragma once
#include <cstddef>
#include <vector>

namespace metis
{
    // Structure-of-arrays host state. Canonical FP64 representation.
    struct SimState
    {
        std::vector<double> px, py, pz;
        std::vector<double> vx, vy, vz;
        std::vector<double> ax, ay, az;
        std::vector<double> mass;

        double softening = 1.0e4;

        std::size_t size() const { return px.size(); }
        void resize(std::size_t n);
        void clearAccelerations();
    };
}
