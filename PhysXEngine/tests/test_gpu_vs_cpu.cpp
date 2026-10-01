#include <gtest/gtest.h>
#include <cmath>
#include "metis/GravityCpu.hpp"
#include "metis/GravityGpu.hpp"
#include "metis/LeapfrogKdk.hpp"
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
        // reference already carries the CPU oracle's ax/ay/az (every caller
        // runs the CPU pass first). Clear them so a comparison can only pass
        // because downloadTo actually wrote the GPU's own result, not because
        // the oracle silently compared against itself.
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
        ASSERT_TRUE(std::isfinite(out.vx[i]) && std::isfinite(out.vy[i]) && std::isfinite(out.vz[i]))
            << "non-finite velocity at particle " << i;
        worst = std::max(worst, std::abs(out.vx[i] - s.vx[i]));
        worst = std::max(worst, std::abs(out.vy[i] - s.vy[i]));
        worst = std::max(worst, std::abs(out.vz[i] - s.vz[i]));
    }
    EXPECT_LT(worst, 1e-3) << "worst absolute velocity round-trip error=" << worst;
}

// Validates the device integrator against the CPU reference over a full
// KDK sequence, not just a single force evaluation: N=1000 pads (padded !=
// n), so this also exercises the zero-mass padding path. Also pins
// stepLeapfrog's prime-exactly-once behaviour: if the initial force
// evaluation were skipped, the first kickDrift would read devAcc straight
// out of cudaMalloc (never written), and the isfinite assertions below
// would fail immediately rather than this silently passing.
//
// Mass and dt match main.cpp's scenario (1e30, 2.0), not the smaller
// values this test used to carry: at 1e24/dt=1.0 the real per-step drift
// is ~1e-3 m against an ~8 m FP32 ULP at this position scale (~1e8 m), so
// kickDrift's position update rounded back to the same float every step
// and the test passed on a frozen simulation. Positions are compared as
// displacement from each path's own t=0 rather than as absolute values, so
// a tiny genuine drift can no longer cancel favourably against the
// ~1e8-1e9 m absolute position scale the way it did before.
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
        // Worst absolute displacement mismatch, normalized once below by
        // the ensemble's own displacement scale (maxCpuDisp) rather than
        // per-particle: individual displacements here span ~5e3-2.4e5 m, so
        // dividing each particle's error by its own (possibly modest)
        // displacement inflates particles that simply moved less into
        // spurious "worst" outliers, the same near-zero-denominator
        // problem a per-particle metric has at any scale.
        worstDispErr = std::max(worstDispErr, std::sqrt(ddx*ddx + ddy*ddy + ddz*ddz));

        const double vdx = gpuState.vx[i] - cpuState.vx[i];
        const double vdy = gpuState.vy[i] - cpuState.vy[i];
        const double vdz = gpuState.vz[i] - cpuState.vz[i];
        const double velErr = std::sqrt(vdx*vdx + vdy*vdy + vdz*vdz);
        const double velMag = std::sqrt(cpuState.vx[i]*cpuState.vx[i] + cpuState.vy[i]*cpuState.vy[i]
                                       + cpuState.vz[i]*cpuState.vz[i]);
        if (velMag > 0.0) worstVelRel = std::max(worstVelRel, velErr / velMag);
    }

    // Worst per-particle displacement mismatch, normalized once by the
    // ensemble's own displacement scale rather than per-particle (see the
    // comment above worstDispErr's accumulation for why).
    const double worstPosRel = maxCpuDisp > 0.0 ? worstDispErr / maxCpuDisp : 0.0;

    std::printf("StepLeapfrogMatchesCpuLeapfrogOverMultipleSteps: worst pos rel=%.3e  worst vel rel=%.3e  "
                "max cpu disp=%.3e m  max gpu disp=%.3e m\n",
                worstPosRel, worstVelRel, maxCpuDisp, maxGpuDisp);

    // Proves the simulation actually advanced, independent of the relative
    // checks below: a disabled kickDrift measured here at 46 m of "motion"
    // on a genuinely frozen position buffer -- not 0, because downloading
    // float32 positions and differencing against the float64 init state
    // carries its own upload/download round-trip quantization noise at this
    // position scale (~1e8-1e9 m). 1000 m clears that noise floor by >10x
    // while staying >100x below the hundreds-of-thousands-of-metres real
    // displacement this configuration produces when the kernel is correct.
    ASSERT_GT(maxCpuDisp, 1000.0) << "CPU oracle itself barely moved; scenario/dt regressed";
    ASSERT_GT(maxGpuDisp, 1000.0) << "device integrator produced no meaningful displacement: frozen simulation";

    // Measured worstPosRel at this seed/config is ~2.7e-3 (FP32 force-sum
    // round-off compounding over 20 KDK steps across 1000 interacting
    // particles); 1e-2 keeps a real margin above that without being loose
    // enough to miss a genuine regression (e.g. a reintroduced clamping or
    // scaling defect, which moved this metric by orders of magnitude in
    // earlier rounds of this test, not a few x).
    EXPECT_LT(worstPosRel, 1e-2) << "device leapfrog displacement trajectory diverges from CPU oracle";
    EXPECT_LT(worstVelRel, 1e-3) << "device leapfrog velocity trajectory diverges from CPU oracle";
}

// Regression coverage for the "particle count shrinks to zero" transition:
// computeAccelerations only accepts a state whose size matches impl_->count,
// so this fails loudly (a size-mismatch throw) if uploadFrom(empty) ever
// left count at its previous nonzero value instead of resetting it to 0.
// A test that only ever uploads an empty state to a fresh instance cannot
// exercise this: impl_->count defaults to 0, so it would pass whether or
// not uploadFrom updates anything at all. (writePositionsToGlBuffer's
// upload-tracking cannot be covered by this suite: it requires a real GL
// context/VBO to reach past attachGlBuffer, which this test harness does
// not have.)
TEST(GravityGpu, UploadEmptyStateAfterNonEmptyResetsCount)
{
    metis::GravityGpu gpu;

    metis::SimState populated = metis::scenario::randomCloud(1000, 1.0e9, 1.0e24, 9090u);
    ASSERT_NO_THROW(gpu.uploadFrom(populated));

    metis::SimState empty;
    ASSERT_NO_THROW(gpu.uploadFrom(empty));
    EXPECT_NO_THROW(gpu.computeAccelerations(empty));
}
