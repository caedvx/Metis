#include <cstddef>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <epoxy/gl.h>
#include "metis/GravityGpu.hpp"
#include "metis/Scenario.hpp"
#include "Camera.hpp"
#include "FrameTimer.hpp"
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
// An optional argv[2] overrides the particle count (default 32768), which
// the same measurement needs to sweep without disturbing the argv[1]
// contract above.
int main(int argc, char** argv)
{
    bool hasLimit = false;
    long maxFrames = 0;
    std::size_t count = 32768;

    if (argc > 1) {
        char* end = nullptr;
        maxFrames = std::strtol(argv[1], &end, 10);
        if (end == argv[1] || *end != '\0' || maxFrames < 0) {
            std::fprintf(stderr, "usage: metis_app [frame-count >= 0] [particle-count]\n");
            return 1;
        }
        hasLimit = true;
    }
    if (argc > 2) {
        char* end = nullptr;
        const long n = std::strtol(argv[2], &end, 10);
        if (end == argv[2] || *end != '\0' || n <= 0) {
            std::fprintf(stderr, "usage: metis_app [frame-count >= 0] [particle-count]\n");
            return 1;
        }
        count = static_cast<std::size_t>(n);
    }

    metis::Window window("Metis", 1280, 720);
    std::printf("GL_VERSION  : %s\n", glGetString(GL_VERSION));
    std::printf("GL_RENDERER : %s\n", glGetString(GL_RENDERER));
    std::printf("particles   : %zu\n", count);

    metis::SimState state = metis::scenario::randomCloud(count, 1.0e9, 1.0e30, 42u);

    // renderer is declared before gpu so that, on any exit path (normal
    // return or exception unwind out of the loop below), gpu's destructor
    // runs before renderer's: the CUDA graphics resource gets unregistered
    // before the VBO it points at is deleted. The explicit
    // gpu.detachGlBuffer() call after the loop does the same thing on the
    // normal path; this ordering is the backstop for the unwind path.
    metis::ParticleRenderer renderer(count);

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
    constexpr float kDt = 2.0f;

    const bool verifyPixels = std::getenv("METIS_VERIFY_PIXELS") != nullptr;

    // A non-interactive run (frame limit given) sizes the window to the
    // whole run so percentileMs sees the full population; FrameTimer's
    // 300-sample interactive default would make p99.9 meaningless.
    metis::FrameTimer timer(hasLimit ? static_cast<std::size_t>(maxFrames) : 300);
    long frame = 0;
    while (window.pollEvents()) {
        if (hasLimit && frame >= maxFrames) break;

        camera.setAspect(static_cast<float>(window.width()) / window.height());
        if (window.leftHeld())
            camera.orbit(window.mouseDeltaX() * 0.005f, window.mouseDeltaY() * 0.005f);
        if (window.rightHeld())
            camera.pan(window.mouseDeltaX() * 0.002f, window.mouseDeltaY() * 0.002f);
        if (window.scrollDelta() != 0.0f)
            camera.zoom(window.scrollDelta() > 0.0f ? 0.9f : 1.1f);

        gpu.stepLeapfrog(state, kDt);
        gpu.writePositionsToGlBuffer();

        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderer.draw(camera, count, kPointScale);

        if (verifyPixels && hasLimit && frame == maxFrames - 1)
            verifyPixelsRendered(window.width(), window.height());

        window.swap();
        timer.tick();
        ++frame;

        if (frame % 120 == 0) {
            std::printf("avg %.2f ms (%.1f fps)   p50 %.2f ms (%.1f fps)   p99 %.2f ms (%.1f fps)   "
                        "p99.9 %.2f ms (%.1f fps)   worst %.2f ms (%.1f fps)\n",
                        timer.averageMs(), timer.averageFps(),
                        timer.percentileMs(0.50), timer.percentileFps(0.50),
                        timer.percentileMs(0.99), timer.percentileFps(0.99),
                        timer.percentileMs(0.999), timer.percentileFps(0.999),
                        timer.worstMs(), timer.worstFps());
            std::fflush(stdout);
            timer.resetClock();
        }
    }

    std::printf("final: avg %.2f ms (%.1f fps)   p50 %.2f ms (%.1f fps)   p99 %.2f ms (%.1f fps)   "
                "p99.9 %.2f ms (%.1f fps)   worst %.2f ms (%.1f fps)\n",
                timer.averageMs(), timer.averageFps(),
                timer.percentileMs(0.50), timer.percentileFps(0.50),
                timer.percentileMs(0.99), timer.percentileFps(0.99),
                timer.percentileMs(0.999), timer.percentileFps(0.999),
                timer.worstMs(), timer.worstFps());

    gpu.detachGlBuffer();
    return 0;
}
