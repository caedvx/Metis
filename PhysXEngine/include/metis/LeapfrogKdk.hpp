#pragma once
#include <cstddef>
#include "metis/IIntegrator.hpp"

namespace metis
{
    // Kick-drift-kick leapfrog. Second-order symplectic.
    // Caches accelerations between steps so each step after the first costs
    // one force evaluation (the first costs two: the priming call plus the
    // post-drift recompute).
    class LeapfrogKdk final : public IIntegrator
    {
    public:
        void step(SimState& state, IForceModel& forces, double dt) override;
        const char* name() const override { return "LeapfrogKDK"; }

    private:
        const SimState* primedFor_ = nullptr;
        std::size_t primedSize_ = 0;
    };
}
