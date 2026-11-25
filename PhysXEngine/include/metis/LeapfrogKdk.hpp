#pragma once
#include <cstddef>
#include "metis/IIntegrator.hpp"

namespace metis
{
    // Kick-drift-kick leapfrog. Second-order symplectic.
    // Accelerations carry over between steps: one force evaluation per step,
    // two on the first.
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
