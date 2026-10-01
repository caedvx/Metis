#pragma once
#include "metis/IIntegrator.hpp"

namespace metis
{
    // First-order symplectic. Retained as the convergence-order baseline.
    class SemiImplicitEuler final : public IIntegrator
    {
    public:
        void step(SimState& state, IForceModel& forces, double dt) override;
        const char* name() const override { return "SemiImplicitEuler"; }
    };
}
