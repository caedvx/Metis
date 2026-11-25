#include <cstdio>
#include <cstdlib>
#include <epoxy/gl.h>

#include "metis/GravityGpu.hpp"
#include "metis/Scenario.hpp"
#include "Camera.hpp"
#include "FrameTimer.hpp"
#include "ParticleRenderer.hpp"
#include "Window.hpp"

// Frame-timing sweep with no per-frame I/O. Produces the README's
// performance table.
int main(int argc, char** argv)
{
    const std::size_t count = (argc > 1) ? std::strtoul(argv[1], nullptr, 10) : 32768;
    const long total = (argc > 2) ? std::strtol(argv[2], nullptr, 10) : 3000;
    const long warmup = (argc > 3) ? std::strtol(argv[3], nullptr, 10) : 900;
    const float dt = 2.0f;
    const float kPointScale = 1.2e10f;

    metis::Window window("bench_noio", 1280, 720);
    metis::SimState state = metis::scenario::randomCloud(count, 1.0e9, 1.0e30, 42u);

    metis::ParticleRenderer renderer(count);
    metis::GravityGpu gpu;
    gpu.uploadFrom(state);
    gpu.attachGlBuffer(renderer.vbo());

    metis::Camera camera;
    camera.setDistance(3.0e9f);

    // Window sized to the whole run so p99.9 covers every frame.
    metis::FrameTimer timer(static_cast<std::size_t>(total > warmup ? total - warmup : 1));

    long frame = 0;
    while (frame < total && window.pollEvents()) {
        camera.setAspect(static_cast<float>(window.width()) / window.height());

        gpu.stepLeapfrog(state, dt);
        gpu.writePositionsToGlBuffer();

        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        renderer.draw(camera, count, kPointScale);
        window.swap();

        // Warm-up frames never reach the timer; resetClock() keeps their time out
        // of the first real sample.
        if (frame == warmup - 1) timer.resetClock();
        else if (frame >= warmup) timer.tick();
        ++frame;
    }

    gpu.detachGlBuffer();

    if (frame <= warmup) {
        std::printf("not enough frames past warmup\n");
        return 1;
    }

    const double avgMs = timer.averageMs();
    const double p50 = timer.percentileMs(0.50);
    const double p99 = timer.percentileMs(0.99);
    const double p999 = timer.percentileMs(0.999);
    const double worstMs = timer.worstMs();
    const long measured = frame - warmup;

    std::printf("frames measured (post-warmup) : %ld\n", measured);
    std::printf("avg   : %.3f ms (%.1f fps)\n", avgMs, 1000.0 / avgMs);
    std::printf("p50   : %.3f ms (%.1f fps)\n", p50, 1000.0 / p50);
    std::printf("p99   : %.3f ms (%.1f fps)\n", p99, 1000.0 / p99);
    std::printf("p99.9 : %.3f ms (%.1f fps)\n", p999, 1000.0 / p999);
    std::printf("worst : %.3f ms (%.1f fps)\n", worstMs, 1000.0 / worstMs);
    std::printf("CSV,%zu,%ld,%.3f,%.1f,%.3f,%.1f,%.3f,%.1f,%.3f,%.1f,%.3f,%.1f\n",
                count, measured,
                avgMs, 1000.0 / avgMs, p50, 1000.0 / p50,
                p99, 1000.0 / p99, p999, 1000.0 / p999, worstMs, 1000.0 / worstMs);
    return 0;
}
