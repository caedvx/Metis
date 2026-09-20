#include "metis/GravityGpu.hpp"
#include "metis/Physics.hpp"

#include <cuda_runtime.h>
#include <cuda_gl_interop.h>
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
        const float4 p = pos[i < n ? i : n - 1];
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

    ~Impl()
    {
        if (glResource) cudaGraphicsUnregisterResource(glResource);
        if (devPos) cudaFree(devPos);
        if (devVel) cudaFree(devVel);
        if (devAcc) cudaFree(devAcc);
    }

    void allocate(int n)
    {
        const int blocks = (n + kThreadsPerBlock - 1) / kThreadsPerBlock;
        const int pad = blocks * kThreadsPerBlock;
        if (pad == padded) { count = n; return; }

        if (devPos) cudaFree(devPos);
        if (devVel) cudaFree(devVel);
        if (devAcc) cudaFree(devAcc);
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
}

void GravityGpu::computeAccelerations(SimState& s)
{
    const int n = impl_->count;
    if (n == 0) return;

    const int blocks = impl_->padded / kThreadsPerBlock;
    const std::size_t shmem = kThreadsPerBlock * sizeof(float4);
    const float eps2 = static_cast<float>(s.softening * s.softening);

    nbodyTiled<<<blocks, kThreadsPerBlock, shmem>>>(
        impl_->devPos, impl_->devAcc, n, eps2, static_cast<float>(physics::G));
    check(cudaGetLastError(), "nbodyTiled launch");
}

void GravityGpu::downloadTo(SimState& s)
{
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
    if (impl_->glResource) cudaGraphicsUnregisterResource(impl_->glResource);
    check(cudaGraphicsGLRegisterBuffer(&impl_->glResource, vbo,
                                       cudaGraphicsRegisterFlagsWriteDiscard),
          "register GL buffer");
}

void GravityGpu::writePositionsToGlBuffer()
{
    if (!impl_->glResource) return;

    check(cudaGraphicsMapResources(1, &impl_->glResource), "map GL resource");

    float4* mapped = nullptr;
    std::size_t bytes = 0;
    check(cudaGraphicsResourceGetMappedPointer(reinterpret_cast<void**>(&mapped),
                                               &bytes, impl_->glResource),
          "get mapped pointer");

    const int blocks = (impl_->count + kThreadsPerBlock - 1) / kThreadsPerBlock;
    copyPositionsToVbo<<<blocks, kThreadsPerBlock>>>(impl_->devPos, mapped, impl_->count);
    check(cudaGetLastError(), "copyPositionsToVbo launch");

    check(cudaGraphicsUnmapResources(1, &impl_->glResource), "unmap GL resource");
}

} // namespace metis
