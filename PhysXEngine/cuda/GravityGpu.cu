#include "metis/GravityGpu.hpp"
#include "metis/Physics.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
#include <cstdio>
#include <stdexcept>
#include <string>
#include <vector>

namespace metis
{
namespace
{
    constexpr int kThreadsPerBlock = 256;

    void check(cudaError_t e, const char* what)
    {
        if (e != cudaSuccess)
            throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
    }

    __global__ void nbodyTiled(const float4* __restrict__ pos, float4* acc,
                               int n, float eps2, float gravity)
    {
        extern __shared__ float4 tile[];

        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        const float4 p = pos[i];
        float ax = 0.0f, ay = 0.0f, az = 0.0f;

        for (int t = 0; t < gridDim.x; ++t) {
            const int idx = t * blockDim.x + threadIdx.x;
            tile[threadIdx.x] = pos[idx];
            __syncthreads();

            #pragma unroll 8
            for (int j = 0; j < blockDim.x; ++j) {
                const float4 q = tile[j];
                const float dx = q.x - p.x;
                const float dy = q.y - p.y;
                const float dz = q.z - p.z;
                const float inv = rsqrtf(dx*dx + dy*dy + dz*dz + eps2);
                const float k = gravity * q.w * inv * inv * inv;
                ax += dx * k; ay += dy * k; az += dz * k;
            }
            __syncthreads();
        }

        if (i < n) { acc[i] = make_float4(ax, ay, az, 0.0f); }
    }

    __global__ void copyPositionsToVbo(const float4* __restrict__ pos, float4* vbo, int n)
    {
        const int i = blockIdx.x * blockDim.x + threadIdx.x;
        if (i < n) vbo[i] = pos[i];
    }
}

struct GravityGpu::Impl
{
    float4* devPos = nullptr;
    float4* devVel = nullptr;
    float4* devAcc = nullptr;
    int count = 0;
    int padded = 0;
    cudaGraphicsResource* glResource = nullptr;
    // Tracked explicitly rather than inferred from devPos being non-null:
    // allocate(0) on a fresh instance takes the pad == padded early return
    // (both are 0) without ever touching devPos, so an upload of a
    // genuinely empty state leaves devPos null despite having completed.
    bool everUploaded = false;

    ~Impl()
    {
        // Never throws from a destructor, but a discarded failure here is
        // the one diagnostic that would reveal a stale/deleted GL buffer
        // still being registered at teardown, so report it instead.
        if (glResource) {
            const cudaError_t err = cudaGraphicsUnregisterResource(glResource);
            if (err != cudaSuccess) {
                std::fprintf(stderr, "GravityGpu: cudaGraphicsUnregisterResource failed in ~Impl: %s\n",
                             cudaGetErrorString(err));
            }
        }
        cudaFree(devPos);
        cudaFree(devVel);
        cudaFree(devAcc);
    }

    void allocate(int n)
    {
        const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const int pad = blocks * kThreadsPerBlock;
        if (pad == padded) { count = n; return; }

        // padded is dropped to 0 for the duration of reallocation: if any
        // cudaMalloc below throws, a subsequent allocate() call sees no
        // stale padded value that could otherwise match a retry's pad and
        // early-return onto freed/dangling device pointers.
        cudaFree(devPos); devPos = nullptr;
        cudaFree(devVel); devVel = nullptr;
        cudaFree(devAcc); devAcc = nullptr;
        padded = 0;
        count = 0;

        check(cudaMalloc(&devPos, pad * sizeof(float4)), "cudaMalloc pos");
        check(cudaMalloc(&devVel, pad * sizeof(float4)), "cudaMalloc vel");
        check(cudaMalloc(&devAcc, pad * sizeof(float4)), "cudaMalloc acc");
        padded = pad;
        count = n;
    }
};

GravityGpu::GravityGpu() : impl_(std::make_unique<Impl>()) {}
GravityGpu::~GravityGpu() = default;

void GravityGpu::uploadFrom(const SimState& s)
{
    // Brackets the whole upload precisely: allocate() commits count/padded
    // before either memcpy below runs, so if either memcpy throws, a stale
    // or uninitialised devPos would otherwise still read as a completed
    // upload. Clearing the flag here and only setting it true once both
    // memcpys have succeeded ensures a partial upload cannot read as one.
    impl_->everUploaded = false;

    const int n = static_cast<int>(s.size());
    impl_->allocate(n);

    std::vector<float4> hostPos(impl_->padded);
    for (int i = 0; i < n; ++i) {
        hostPos[i] = make_float4(static_cast<float>(s.px[i]), static_cast<float>(s.py[i]),
                                 static_cast<float>(s.pz[i]), static_cast<float>(s.mass[i]));
    }
    for (int i = n; i < impl_->padded; ++i) hostPos[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    check(cudaMemcpy(impl_->devPos, hostPos.data(), impl_->padded * sizeof(float4),
                     cudaMemcpyHostToDevice), "upload positions");

    std::vector<float4> hostVel(impl_->padded);
    for (int i = 0; i < n; ++i) {
        hostVel[i] = make_float4(static_cast<float>(s.vx[i]), static_cast<float>(s.vy[i]),
                                 static_cast<float>(s.vz[i]), 0.0f);
    }
    for (int i = n; i < impl_->padded; ++i) hostVel[i] = make_float4(0.0f, 0.0f, 0.0f, 0.0f);

    check(cudaMemcpy(impl_->devVel, hostVel.data(), impl_->padded * sizeof(float4),
                     cudaMemcpyHostToDevice), "upload velocities");

    impl_->everUploaded = true;
}

void GravityGpu::computeAccelerations(SimState& s)
{
    if (static_cast<int>(s.size()) != impl_->count)
        throw std::runtime_error("GravityGpu: uploadFrom(state) required before computeAccelerations");
    // An empty state uploaded and then computed is a legitimate no-op; a
    // forgotten upload is caught above by the size mismatch instead.
    if (impl_->count == 0) return;

    const int n = impl_->count;
    const int blocks = impl_->padded / kThreadsPerBlock;
    const std::size_t shmem = kThreadsPerBlock * sizeof(float4);
    const float eps2 = static_cast<float>(s.softening * s.softening);
    if (!(eps2 > 0.0f))
        throw std::runtime_error("GravityGpu requires SimState::softening > 0");

    nbodyTiled<<<blocks, kThreadsPerBlock, shmem>>>(
        impl_->devPos, impl_->devAcc, n, eps2, static_cast<float>(physics::G));
    check(cudaGetLastError(), "nbodyTiled launch");
#ifndef NDEBUG
    check(cudaDeviceSynchronize(), "nbodyTiled execution");
#endif
}

void GravityGpu::downloadTo(SimState& s)
{
    if (s.size() != static_cast<std::size_t>(impl_->count))
        throw std::runtime_error("GravityGpu::downloadTo: state size does not match the uploaded particle count");

    const int n = impl_->count;
    std::vector<float4> host(impl_->padded);
    check(cudaMemcpy(host.data(), impl_->devAcc, impl_->padded * sizeof(float4),
                     cudaMemcpyDeviceToHost), "download accelerations");
    for (int i = 0; i < n; ++i) {
        s.ax[i] = host[i].x;
        s.ay[i] = host[i].y;
        s.az[i] = host[i].z;
    }
}

void GravityGpu::downloadVelocitiesTo(SimState& s)
{
    if (s.size() != static_cast<std::size_t>(impl_->count))
        throw std::runtime_error("GravityGpu::downloadVelocitiesTo: state size does not match the uploaded particle count");

    const int n = impl_->count;
    std::vector<float4> host(impl_->padded);
    check(cudaMemcpy(host.data(), impl_->devVel, impl_->padded * sizeof(float4),
                     cudaMemcpyDeviceToHost), "download velocities");
    for (int i = 0; i < n; ++i) {
        s.vx[i] = host[i].x;
        s.vy[i] = host[i].y;
        s.vz[i] = host[i].z;
    }
}

void GravityGpu::attachGlBuffer(unsigned vbo)
{
    detachGlBuffer();

    cudaGraphicsResource* resource = nullptr;
    check(cudaGraphicsGLRegisterBuffer(&resource, vbo, cudaGraphicsRegisterFlagsWriteDiscard),
          "register GL buffer");
    impl_->glResource = resource;
}

void GravityGpu::detachGlBuffer()
{
    if (impl_->glResource) {
        cudaGraphicsUnregisterResource(impl_->glResource);
        impl_->glResource = nullptr;
    }
}

namespace
{
    // Guarantees the GL resource is unmapped on every exit path, including
    // an exception thrown from a check() call below. A leaked mapping wedges
    // the resource: every later call fails with cudaErrorAlreadyMapped and GL
    // can never touch the buffer again.
    struct UnmapGuard
    {
        cudaGraphicsResource** resource;
        ~UnmapGuard()
        {
            // Never throws from a destructor. Unmap, then unconditionally
            // drain the runtime's sticky last-error: either the unmap itself
            // just failed, or an earlier call in this function (e.g.
            // cudaGraphicsResourceGetMappedPointer) threw and left its error
            // sticky while the unmap succeeds. Either way, an undrained
            // sticky error would otherwise be misattributed to the next
            // unrelated check(cudaGetLastError(), ...) call (e.g. the next
            // kernel launch's error check). This does not mask a genuine
            // hard failure (device assert, illegal address, ECC) — those are
            // sticky-unrecoverable and cudaGetLastError cannot clear them.
            cudaGraphicsUnmapResources(1, resource);
            cudaGetLastError();
        }
    };
}

void GravityGpu::writePositionsToGlBuffer()
{
    if (!impl_->glResource) return;
    if (!impl_->everUploaded)
        throw std::runtime_error("GravityGpu: uploadFrom(state) required before writePositionsToGlBuffer");
    // A legitimate uploaded-empty state is a clean no-op; a 0-block launch
    // below would otherwise fail with "invalid configuration argument".
    if (impl_->count == 0) return;

    check(cudaGraphicsMapResources(1, &impl_->glResource), "map GL resource");
    UnmapGuard unmapGuard{&impl_->glResource};

    float4* mapped = nullptr;
    std::size_t bytes = 0;
    check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mapped),
                                               &bytes, impl_->glResource),
          "get mapped pointer");

    if (bytes < static_cast<std::size_t>(impl_->count) * sizeof(float4))
        throw std::runtime_error("GravityGpu::writePositionsToGlBuffer: VBO too small for particle count");

    const int blocks = (impl_->count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    copyPositionsToVbo<<<blocks, kThreadsPerBlock>>>(impl_->devPos, mapped, impl_->count);
    check(cudaGetLastError(), "copyPositionsToVbo launch");
}

} // namespace metis
