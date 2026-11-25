#pragma once
#include <memory>
#include "metis/SimState.hpp"

namespace metis
{
    // CUDA tiled direct summation, FP32 on device. CUDA headers stay behind the
    // pimpl.
    //
    // Not an IForceModel: computeAccelerations needs a prior uploadFrom and
    // doesn't write state.ax/ay/az, so LeapfrogKdk would run on stale
    // accelerations. Use stepLeapfrog.
    class GravityGpu final
    {
    public:
        GravityGpu();
        ~GravityGpu();

        GravityGpu(const GravityGpu&) = delete;
        GravityGpu& operator=(const GravityGpu&) = delete;

        void computeAccelerations(SimState& state);
        const char* name() const { return "GravityGpu(FP32)"; }

        void uploadFrom(const SimState& state);
        void downloadTo(SimState& state);
        void downloadVelocitiesTo(SimState& state);
        void downloadPositionsTo(SimState& state);

        // One KDK leapfrog step entirely on device. No host round-trip.
        void stepLeapfrog(const SimState& state, float dt);

        // Registers an OpenGL VBO as the position source for zero-copy rendering.
        // The VBO must hold at least as many vec4 (16-byte) elements as the last
        // uploaded particle count: each element is (x, y, z, mass).
        void attachGlBuffer(unsigned vbo);
        // Unregisters the attached GL buffer. Call this before deleting the buffer;
        // attachGlBuffer calls it before registering a new one.
        void detachGlBuffer();
        void writePositionsToGlBuffer();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
