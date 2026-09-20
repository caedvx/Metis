#pragma once
#include <memory>
#include "metis/IForceModel.hpp"

namespace metis
{
    // CUDA tiled direct summation, FP32 on device.
    // No CUDA headers here: device state lives behind the pimpl.
    //
    // Unlike GravityCpu, this is not a drop-in IForceModel: computeAccelerations
    // requires a prior uploadFrom(state) call with a state of the same particle
    // count, and throws otherwise. Results must be retrieved with downloadTo /
    // downloadVelocitiesTo; state.ax/ay/az are not touched by computeAccelerations
    // itself.
    class GravityGpu final : public IForceModel
    {
    public:
        GravityGpu();
        ~GravityGpu() override;

        GravityGpu(const GravityGpu&) = delete;
        GravityGpu& operator=(const GravityGpu&) = delete;

        void computeAccelerations(SimState& state) override;
        const char* name() const override { return "GravityGpu(FP32)"; }

        void uploadFrom(const SimState& state);
        void downloadTo(SimState& state);
        void downloadVelocitiesTo(SimState& state);

        // Registers an OpenGL VBO as the position source for zero-copy rendering.
        // The VBO must hold at least as many vec4 (16-byte) elements as the last
        // uploaded particle count: each element is (x, y, z, mass).
        void attachGlBuffer(unsigned vbo);
        void writePositionsToGlBuffer();

    private:
        struct Impl;
        std::unique_ptr<Impl> impl_;
    };
}
