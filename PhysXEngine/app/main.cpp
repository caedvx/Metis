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
    // Counts framebuffer pixels that differ from the clear color. Only runs
    // with METIS_VERIFY_PIXELS set.
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

// No argv[1]: interactive until Escape or close. argv[1]: exit after that
// many frames. argv[2]: particle count (default 32768).
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

    // renderer is declared before gpu so gpu's destructor runs first on unwind,
    // unregistering the CUDA resource before its VBO is deleted. The explicit
    // detachGlBuffer() after the loop covers the normal path.
    metis::ParticleRenderer renderer(count);

    metis::GravityGpu gpu;
    gpu.uploadFrom(state);
    gpu.attachGlBuffer(renderer.vbo());

    metis::Camera camera;
    camera.setDistance(3.0e9f);

    // gl_PointSize = pointScale / w. The cloud sits ~3e9 from the camera, so
    // 4 px there needs 4 * 3e9 = 1.2e10.
    constexpr float kPointScale = 1.2e10f;
    constexpr float kDt = 2.0f;

    const bool verifyPixels = std::getenv("METIS_VERIFY_PIXELS") != nullptr;

    // With a frame limit, size the window to the whole run so p99.9 covers
    // every frame.
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
