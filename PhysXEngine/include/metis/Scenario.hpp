#pragma once
#include <cstddef>
#include <cstdint>
#include "metis/SimState.hpp"

namespace metis::scenario
{
    SimState earthMoon();
    SimState randomCloud(std::size_t n, double radius, double totalMass, std::uint32_t seed);
}
