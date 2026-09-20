#pragma once
#include "metis/IIntegrator.hpp"

namespace metis
{
    // Kick-drift-kick leapfrog. Second-order symplectic.
    // Caches accelerations between steps so each step costs one force evaluation.
    class LeapfrogKdk final : public IIntegrator
    {
    public:
        void step(SimState& state, IForceModel& forces, double dt) override;
        const char* name() const override { return "LeapfrogKDK"; }

    private:
        bool primed_ = false;
    };
}
