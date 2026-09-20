#include <cstdio>
#include <cstdlib>
#include <epoxy/gl.h>
#include "Window.hpp"

// With no argv[1], runs interactively until Escape/close.
// With argv[1], exits after that many frames so this can run unattended
int main(int argc, char** argv)
{
    const long maxFrames = (argc > 1) ? std::strtol(argv[1], nullptr, 10) : -1;

    metis::Window window("Metis", 1280, 720);
    std::printf("GL_VERSION  : %s\n", glGetString(GL_VERSION));
    std::printf("GL_RENDERER : %s\n", glGetString(GL_RENDERER));

    long frame = 0;
    while (window.pollEvents()) {
        glClearColor(0.02f, 0.02f, 0.05f, 1.0f);
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);
        window.swap();
        if (maxFrames > 0 && ++frame >= maxFrames) break;
    }
    return 0;
}
