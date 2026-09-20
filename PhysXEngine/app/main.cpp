#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <epoxy/gl.h>
#include "metis/GravityGpu.hpp"
#include "metis/Scenario.hpp"
#include "Camera.hpp"
#include "ParticleRenderer.hpp"
#include "Window.hpp"

namespace
{
    // Reads the framebuffer back and reports how many pixels differ from the
    // clear color. Only runs when METIS_VERIFY_PIXELS is set in the
    // environment, so it never touches the default frame loop.
    void verifyPixelsRendered(int width, int height)
    {
        std::vector<unsigned char> pixels(static_cast<std::size_t>(width) * height * 4);
        glReadPixels(0, 0, width, height, GL_RGBA, GL_UNSIGNED_BYTE, pixels.data());

        std::size_t nonBackground = 0;
        for (std::size_t i = 0; i < pixels.size(); i += 4) {
            if (pixels[i] > 10 || pixels[i + 1] > 10 || pixels[i + 2] > 25) ++nonBackground;
        }
        std::printf("METIS_VERIFY_PIXELS: %zu of %d pixels non-background\n",
                    nonBackground, width * height);
    }
}

// With no argv[1], runs interactively until Escape/close.
// With argv[1], exits after that many frames so this can run unattended
int main(int argc, char** argv)
{
    bool hasLimit = false;
    long maxFrames = 0;
    if (argc > 1) {
        char* end = nullptr;
        maxFrames = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || maxFrames < 0) {
            std::fprintf(stderr, "usage: metis_app [frame-count >= 0]\n");
            return 1;
        }
        hasLimit = true;
    }

    constexpr std::size_t kCount = 32768;

    metis::Window window("Metis", 1280, 720);
    std::printf("GL_VERSION  : %s\n", glGetString(GL_VERSION));
    std::printf("GL_RENDERER : %s\n", glGetString(GL_RENDERER));

    metis::SimState state = metis::scenario::randomCloud(kCount, 1.0e9, 1.0e30, 42u);

    // renderer is declared before gpu so that, on any exit path (normal
    // return or exception unwind out of the loop below), gpu's destructor
    // runs before renderer's: the CUDA graphics resource gets unregistered
    // before the VBO it points at is deleted. The explicit
    // gpu.detachGlBuffer() call after the loop does the same thing on the
    // normal path; this ordering is the backstop for the unwind path.
    metis::ParticleRenderer renderer(kCount);

    metis::GravityGpu gpu;
    gpu.uploadFrom(state);
    gpu.attachGlBuffer(renderer.vbo());

    metis::Camera camera;
    camera.setDistance(3.0e9f);

    // gl_PointSize = pointScale / gl_Position.w, where w is the particle's
    // camera-space depth. At camera distance 3e9 with the cloud's radius-1e9
    // spread, depth ranges roughly 2e9-4e9, averaging the camera distance
    // itself (3e9, since the cloud is centered on the look-at target). For
    // an on-screen sprite of ~4px at that average depth: 4 * 3e9 = 1.2e10.
    constexpr float kPointScale = 1.2e10f;

    const bool verifyPixels = std::getenv("METIS_VERIFY_PIXELS") != nullptr;

    long frame = 0;
    while (window.pollEvents()) {
        if (hasLimit && frame >= maxFrames) break;

        camera.setAspect(static_cast<float>(window.width()) / window.height());
        gpu.writePositionsToGlBuffer();

        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderer.draw(camera, kCount, kPointScale);

        if (verifyPixels && hasLimit && frame == maxFrames - 1)
            verifyPixelsRendered(window.width(), window.height());

        window.swap();
        ++frame;
    }
    gpu.detachGlBuffer();
    return 0;
}
