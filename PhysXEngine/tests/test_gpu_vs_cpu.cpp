#include <gtest/gtest.h>
#include <cmath>
#include "metis/GravityCpu.hpp"
#include "metis/GravityGpu.hpp"
#include "metis/Scenario.hpp"

namespace
{
    double worstRelativeError(const metis::SimState& cpuState, const metis::SimState& gpuState)
    {
        double worst = 0.0;
        for (std::size_t i = 0; i < cpuState.size(); ++i) {
            const double rx = cpuState.ax[i], ry = cpuState.ay[i], rz = cpuState.az[i];
            const double refMag = std::sqrt(rx*rx + ry*ry + rz*rz);
            const double dx = gpuState.ax[i] - rx;
            const double dy = gpuState.ay[i] - ry;
            const double dz = gpuState.az[i] - rz;
            const double errMag = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (refMag > 0.0) worst = std::max(worst, errMag / refMag);
        }
        return worst;
    }
}

TEST(GravityGpu, MatchesCpuOracleWithinFp32Tolerance)
{
    metis::SimState cpuState = metis::scenario::randomCloud(1024, 1.0e9, 1.0e24, 1234u);
    metis::SimState gpuState = cpuState;

    metis::GravityCpu cpu;
    cpu.computeAccelerations(cpuState);

    metis::GravityGpu gpu;
    gpu.uploadFrom(gpuState);
    gpu.computeAccelerations(gpuState);
    gpu.downloadTo(gpuState);

    const double worst = worstRelativeError(cpuState, gpuState);

    EXPECT_LT(worst, 1e-4)
        << "FP32 kernel deviates from the FP64 oracle beyond accumulated-rounding tolerance. "
        << "worst=" << worst;
}

// N=1000 is not a multiple of the 256-thread block size, so padded != n.
// This is the regression test for the tile-padding defect: a kernel that
// clamps padded tile reads to particle n-1 (instead of using zero-mass
// padding) counts the last particle (padded-n+1) times in every force sum.
// N=1024 above is an exact multiple of 256 (padded == n) and cannot detect
// this bug at all.
TEST(GravityGpu, MatchesCpuOracleWithNonMultipleOfBlockSize)
{
    metis::SimState cpuState = metis::scenario::randomCloud(1000, 1.0e9, 1.0e24, 5678u);
    metis::SimState gpuState = cpuState;

    metis::GravityCpu cpu;
    cpu.computeAccelerations(cpuState);

    metis::GravityGpu gpu;
    gpu.uploadFrom(gpuState);
    gpu.computeAccelerations(gpuState);
    gpu.downloadTo(gpuState);

    const double worst = worstRelativeError(cpuState, gpuState);

    EXPECT_LT(worst, 1e-4)
        << "FP32 kernel deviates from the FP64 oracle beyond accumulated-rounding tolerance "
        << "at a particle count that is not a multiple of the block size. worst=" << worst;
}
