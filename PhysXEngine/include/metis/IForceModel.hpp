#pragma once
#include "metis/SimState.hpp"

namespace metis
{
    // Operates on whole buffers. One virtual call per step, never per particle.
    class IForceModel
    {
    public:
        virtual ~IForceModel() = default;
        virtual void computeAccelerations(SimState& state) = 0;
        virtual const char* name() const = 0;
    };
}
