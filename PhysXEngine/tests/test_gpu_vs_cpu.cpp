#include <gtest/gtest.h>
#include <cmath>
#include "metis/GravityCpu.hpp"
#include "metis/GravityGpu.hpp"
#include "metis/Scenario.hpp"

namespace
{
    // ASSERT_TRUE requires a void-returning function; the fatal isfinite
    // check must happen before any use of the value in std::max, since
    // std::max(0.0, NaN) == 0.0 (NaN compares false against everything) and
    // would otherwise let a fully NaN result score worst == 0 and pass.
    void expectMatchesOracle(const metis::SimState& cpuState, const metis::SimState& gpuState,
                              double tolerance, const char* context)
    {
        double worst = 0.0;
        for (std::size_t i = 0; i < cpuState.size(); ++i) {
            ASSERT_TRUE(std::isfinite(gpuState.ax[i]) && std::isfinite(gpuState.ay[i])
                     && std::isfinite(gpuState.az[i]))
                << context << ": non-finite acceleration at particle " << i;

            const double rx = cpuState.ax[i], ry = cpuState.ay[i], rz = cpuState.az[i];
            const double refMag = std::sqrt(rx*rx + ry*ry + rz*rz);
            const double dx = gpuState.ax[i] - rx;
            const double dy = gpuState.ay[i] - ry;
            const double dz = gpuState.az[i] - rz;
            const double errMag = std::sqrt(dx*dx + dy*dy + dz*dz);
            if (refMag > 0.0) worst = std::max(worst, errMag / refMag);
        }
        EXPECT_LT(worst, tolerance)
            << context << ": FP32 kernel deviates from the FP64 oracle beyond tolerance. worst=" << worst;
    }

    metis::SimState computeOnGpu(metis::GravityGpu& gpu, const metis::SimState& reference)
    {
        metis::SimState gpuState = reference;
        gpu.uploadFrom(gpuState);
        gpu.computeAccelerations(gpuState);
        gpu.downloadTo(gpuState);
        return gpuState;
    }
}

TEST(GravityGpu, MatchesCpuOracleWithinFp32Tolerance)
{
    metis::SimState cpuState = metis::scenario::randomCloud(1024, 1.0e9, 1.0e24, 1234u);
    metis::GravityCpu cpu;
    cpu.computeAccelerations(cpuState);

    metis::GravityGpu gpu;
    const metis::SimState gpuState = computeOnGpu(gpu, cpuState);

    expectMatchesOracle(cpuState, gpuState, 1e-4, "N=1024");
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
    metis::GravityCpu cpu;
    cpu.computeAccelerations(cpuState);

    metis::GravityGpu gpu;
    const metis::SimState gpuState = computeOnGpu(gpu, cpuState);

    expectMatchesOracle(cpuState, gpuState, 1e-4, "N=1000");
}

// Every frame of the integration loop reuses one GravityGpu instance across
// varying particle counts. 1024 and 1000 both pad to 1024 (4 blocks of 256),
// so 1024 -> 1000 keeps padded unchanged and exercises Impl::allocate's
// pad == padded early-return branch; 1000 -> 1024 then exercises it again
// on the way back.
TEST(GravityGpu, ReusedInstanceAcrossParticleCounts)
{
    metis::GravityGpu gpu;
    metis::GravityCpu cpu;

    {
        metis::SimState cpuState = metis::scenario::randomCloud(1024, 1.0e9, 1.0e24, 1111u);
        cpu.computeAccelerations(cpuState);
        const metis::SimState gpuState = computeOnGpu(gpu, cpuState);
        expectMatchesOracle(cpuState, gpuState, 1e-4, "N=1024 (first pass)");
    }
    {
        metis::SimState cpuState = metis::scenario::randomCloud(1000, 1.0e9, 1.0e24, 2222u);
        cpu.computeAccelerations(cpuState);
        const metis::SimState gpuState = computeOnGpu(gpu, cpuState);
        expectMatchesOracle(cpuState, gpuState, 1e-4, "N=1000 (reused instance)");
    }
    {
        metis::SimState cpuState = metis::scenario::randomCloud(1024, 1.0e9, 1.0e24, 3333u);
        cpu.computeAccelerations(cpuState);
        const metis::SimState gpuState = computeOnGpu(gpu, cpuState);
        expectMatchesOracle(cpuState, gpuState, 1e-4, "N=1024 (second pass)");
    }
}

TEST(GravityGpu, VelocitiesRoundTrip)
{
    metis::SimState s = metis::scenario::randomCloud(1000, 1.0e9, 1.0e24, 4242u);
    for (std::size_t i = 0; i < s.size(); ++i) {
        s.vx[i] = static_cast<double>(i) * 1.5;
        s.vy[i] = static_cast<double>(i) * -2.25;
        s.vz[i] = 3.0;
    }

    metis::GravityGpu gpu;
    gpu.uploadFrom(s);

    metis::SimState out = s;
    for (std::size_t i = 0; i < out.size(); ++i) { out.vx[i] = 0.0; out.vy[i] = 0.0; out.vz[i] = 0.0; }
    gpu.downloadVelocitiesTo(out);

    double worst = 0.0;
    for (std::size_t i = 0; i < s.size(); ++i) {
        worst = std::max(worst, std::abs(out.vx[i] - s.vx[i]));
        worst = std::max(worst, std::abs(out.vy[i] - s.vy[i]));
        worst = std::max(worst, std::abs(out.vz[i] - s.vz[i]));
    }
    EXPECT_LT(worst, 1e-3) << "worst absolute velocity round-trip error=" << worst;
}
