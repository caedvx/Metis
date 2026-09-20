#pragma once
#include "metis/IForceModel.hpp"

namespace metis
{
    // FP64 direct summation with Plummer softening. The validation oracle.
    class GravityCpu final : public IForceModel
    {
    public:
        void computeAccelerations(SimState& state) override;
        const char* name() const override { return "GravityCpu(FP64)"; }
    };
}
