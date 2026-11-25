#include <gtest/gtest.h>
#include <cmath>
#include "metis/GravityCpu.hpp"
#include "metis/GravityGpu.hpp"
#include "metis/LeapfrogKdk.hpp"
#include "metis/Scenario.hpp"

namespace
{
    // std::max(0.0, NaN) == 0.0, so check isfinite first or an all-NaN result
    // scores 0 and passes.
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
        // Clear the oracle's accelerations so a pass means downloadTo really wrote
        // the GPU's results.
        gpuState.clearAccelerations();
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

// N=1000 isn't a multiple of 256, so the tiles pad. Clamping padded reads to
// particle n-1 instead of zero-mass padding over-counts the last particle;
// N=1024 can't see that.
TEST(GravityGpu, MatchesCpuOracleWithNonMultipleOfBlockSize)
{
    metis::SimState cpuState = metis::scenario::randomCloud(1000, 1.0e9, 1.0e24, 5678u);
    metis::GravityCpu cpu;
    cpu.computeAccelerations(cpuState);

    metis::GravityGpu gpu;
    const metis::SimState gpuState = computeOnGpu(gpu, cpuState);

    expectMatchesOracle(cpuState, gpuState, 1e-4, "N=1000");
}

// 1024 and 1000 both pad to 1024, so switching between them takes allocate()'s
// no-realloc path in both directions.
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
        ASSERT_TRUE(std::isfinite(out.vx[i]) && std::isfinite(out.vy[i]) && std::isfinite(out.vz[i]))
            << "non-finite velocity at particle " << i;
        worst = std::max(worst, std::abs(out.vx[i] - s.vx[i]));
        worst = std::max(worst, std::abs(out.vy[i] - s.vy[i]));
        worst = std::max(worst, std::abs(out.vz[i] - s.vz[i]));
    }
    EXPECT_LT(worst, 1e-3) << "worst absolute velocity round-trip error=" << worst;
}

// Full KDK sequence against the CPU reference at a padded N. If stepLeapfrog
// skipped its initial force pass, kickDrift would read uninitialised devAcc
// and the isfinite checks would fail.
//
// Mass and dt match main.cpp (1e30, 2.0). With smaller values the per-step
// drift (~1e-3 m) is under one FP32 ulp at ~1e8 m and positions never change.
// Compare displacement from t=0, not absolute position.
TEST(GravityGpu, StepLeapfrogMatchesCpuLeapfrogOverMultipleSteps)
{
    constexpr int kSteps = 20;
    constexpr float kDt = 2.0f;

    metis::SimState cpuState = metis::scenario::randomCloud(1000, 1.0e9, 1.0e30, 8181u);
    const metis::SimState cpuInit = cpuState;
    metis::GravityCpu cpuForces;
    metis::LeapfrogKdk cpuIntegrator;
    for (int i = 0; i < kSteps; ++i)
        cpuIntegrator.step(cpuState, cpuForces, static_cast<double>(kDt));

    metis::SimState gpuState = metis::scenario::randomCloud(1000, 1.0e9, 1.0e30, 8181u);
    const metis::SimState gpuInit = gpuState;
    metis::GravityGpu gpu;
    gpu.uploadFrom(gpuState);
    for (int i = 0; i < kSteps; ++i)
        gpu.stepLeapfrog(gpuState, kDt);
    gpu.downloadPositionsTo(gpuState);
    gpu.downloadVelocitiesTo(gpuState);

    double worstDispErr = 0.0, worstVelRel = 0.0, maxCpuDisp = 0.0, maxGpuDisp = 0.0;
    for (std::size_t i = 0; i < cpuState.size(); ++i) {
        ASSERT_TRUE(std::isfinite(gpuState.px[i]) && std::isfinite(gpuState.py[i])
                 && std::isfinite(gpuState.pz[i]))
            << "non-finite position at particle " << i;
        ASSERT_TRUE(std::isfinite(gpuState.vx[i]) && std::isfinite(gpuState.vy[i])
                 && std::isfinite(gpuState.vz[i]))
            << "non-finite velocity at particle " << i;

        const double cdx = cpuState.px[i] - cpuInit.px[i];
        const double cdy = cpuState.py[i] - cpuInit.py[i];
        const double cdz = cpuState.pz[i] - cpuInit.pz[i];
        const double cpuDisp = std::sqrt(cdx*cdx + cdy*cdy + cdz*cdz);

        const double gdx = gpuState.px[i] - gpuInit.px[i];
        const double gdy = gpuState.py[i] - gpuInit.py[i];
        const double gdz = gpuState.pz[i] - gpuInit.pz[i];
        const double gpuDisp = std::sqrt(gdx*gdx + gdy*gdy + gdz*gdz);

        maxCpuDisp = std::max(maxCpuDisp, cpuDisp);
        maxGpuDisp = std::max(maxGpuDisp, gpuDisp);

        const double ddx = gdx - cdx, ddy = gdy - cdy, ddz = gdz - cdz;
        // Normalize by the ensemble's displacement scale, not per particle:
        // displacements span ~5e3-2.4e5 m, and per-particle division inflates the
        // ones that barely moved.
        worstDispErr = std::max(worstDispErr, std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz));

        const double vdx = gpuState.vx[i] - cpuState.vx[i];
        const double vdy = gpuState.vy[i] - cpuState.vy[i];
        const double vdz = gpuState.vz[i] - cpuState.vz[i];
        const double velErr = std::sqrt(vdx*vdx + vdy*vdy + vdz*vdz);
        const double velMag = std::sqrt(cpuState.vx[i]*cpuState.vx[i] + cpuState.vy[i]*cpuState.vy[i]
                                       + cpuState.vz[i]*cpuState.vz[i]);
        if (velMag > 0.0) worstVelRel = std::max(worstVelRel, velErr / velMag);
    }

    // Normalized by the ensemble scale, see above.
    const double worstPosRel = maxCpuDisp > 0.0 ? worstDispErr / maxCpuDisp : 0.0;

    std::printf("StepLeapfrogMatchesCpuLeapfrogOverMultipleSteps: worst pos rel=%.3e  worst vel rel=%.3e  "
                "max cpu disp=%.3e m  max gpu disp=%.3e m\n",
                worstPosRel, worstVelRel, maxCpuDisp, maxGpuDisp);

    // The sim has to actually move. A frozen buffer still reads ~46 m from FP32
    // round-trip noise; real motion here is hundreds of km.
    ASSERT_GT(maxCpuDisp, 1000.0) << "CPU oracle itself barely moved; scenario/dt regressed";
    ASSERT_GT(maxGpuDisp, 1000.0) << "device integrator produced no meaningful displacement: frozen simulation";

    // Measured ~2.7e-3 (FP32 round-off over 20 steps, 1000 particles). 1e-2
    // leaves margin; real bugs move this by orders of magnitude.
    EXPECT_LT(worstPosRel, 1e-2) << "device leapfrog displacement trajectory diverges from CPU oracle";
    EXPECT_LT(worstVelRel, 1e-3) << "device leapfrog velocity trajectory diverges from CPU oracle";
}

// Shrinking to zero: computeAccelerations checks the size against count, so
// this throws if uploadFrom(empty) didn't reset it. A fresh instance already
// has count 0 and couldn't catch that. writePositionsToGlBuffer needs a real
// GL context, so it isn't covered here.
TEST(GravityGpu, UploadEmptyStateAfterNonEmptyResetsCount)
{
    metis::GravityGpu gpu;

    metis::SimState populated = metis::scenario::randomCloud(1000, 1.0e9, 1.0e24, 9090u);
    ASSERT_NO_THROW(gpu.uploadFrom(populated));

    metis::SimState empty;
    ASSERT_NO_THROW(gpu.uploadFrom(empty));
    EXPECT_NO_THROW(gpu.computeAccelerations(empty));
}
