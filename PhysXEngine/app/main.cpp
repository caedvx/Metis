#include <cstdio>
#include <cstdlib>
#include <epoxy/gl.h>
#include "Window.hpp"

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

    metis::Window window("Metis", 1280, 720);
    std::printf("GL_VERSION  : %s\n", glGetString(GL_VERSION));
    std::printf("GL_RENDERER : %s\n", glGetString(GL_RENDERER));

    long frame = 0;
    while (window.pollEvents()) {
        if (hasLimit && frame >= maxFrames) break;
        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        window.swap();
        ++frame;
    }
    return 0;
}
