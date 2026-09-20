#pragma once
#include "metis/IForceModel.hpp"
#include "metis/SimState.hpp"

namespace metis
{
    class IIntegrator
    {
    public:
        virtual ~IIntegrator() = default;
        virtual void step(SimState& state, IForceModel& forces, double dt) = 0;
        virtual const char* name() const = 0;
    };
}
