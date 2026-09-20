#pragma once
#include "metis/SimState.hpp"

namespace metis::diagnostics
{
    // Always FP64, even when the simulation runs FP32 on device.
    double totalEnergy(const SimState& state);
    void totalMomentum(const SimState& state, double& px, double& py, double& pz);
}
