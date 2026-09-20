#include "Window.hpp"

#include <SDL3/SDL.h>
#include <epoxy/gl.h>
#include <stdexcept>
#include <string>

namespace metis
{
    Window::Window(const char* title, int width, int height)
        : width_(width), height_(height)
    {
        if (!SDL_Init(SDL_INIT_VIDEO))
            throw std::runtime_error(std::string("SDL_Init: ") + SDL_GetError());

        try {
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, 4);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, 6);
            SDL_GL_SetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, SDL_GL_CONTEXT_PROFILE_CORE);
            SDL_GL_SetAttribute(SDL_GL_DOUBLEBUFFER, 1);
            SDL_GL_SetAttribute(SDL_GL_DEPTH_SIZE, 24);

            window_ = SDL_CreateWindow(title, width, height,
                                       SDL_WINDOW_OPENGL | SDL_WINDOW_RESIZABLE);
            if (!window_)
                throw std::runtime_error(std::string("SDL_CreateWindow: ") + SDL_GetError());

            glContext_ = SDL_GL_CreateContext(window_);
            if (!glContext_)
                throw std::runtime_error(std::string("SDL_GL_CreateContext: ") + SDL_GetError());

            int major = 0, minor = 0, profile = 0;
            SDL_GL_GetAttribute(SDL_GL_CONTEXT_MAJOR_VERSION, &major);
            SDL_GL_GetAttribute(SDL_GL_CONTEXT_MINOR_VERSION, &minor);
            SDL_GL_GetAttribute(SDL_GL_CONTEXT_PROFILE_MASK, &profile);
            if (major < 4 || (major == 4 && minor < 6) || !(profile & SDL_GL_CONTEXT_PROFILE_CORE))
                throw std::runtime_error("requires an OpenGL 4.6 core profile context; got "
                                         + std::to_string(major) + "." + std::to_string(minor));

            // Vsync off: the 144 Hz target is measured, not clamped.
            SDL_GL_SetSwapInterval(0);
        } catch (...) {
            if (glContext_) SDL_GL_DestroyContext(static_cast<SDL_GLContext>(glContext_));
            if (window_) SDL_DestroyWindow(window_);
            SDL_Quit();
            throw;
        }
    }

    Window::~Window()
    {
        if (glContext_) SDL_GL_DestroyContext(static_cast<SDL_GLContext>(glContext_));
        if (window_) SDL_DestroyWindow(window_);
        SDL_Quit();
    }

    bool Window::pollEvents()
    {
        SDL_Event e;
        while (SDL_PollEvent(&e)) {
            if (e.type == SDL_EVENT_QUIT) return false;
            if (e.type == SDL_EVENT_KEY_DOWN && e.key.key == SDLK_ESCAPE) return false;
            if (e.type == SDL_EVENT_WINDOW_PIXEL_SIZE_CHANGED) {
                width_ = e.window.data1;
                height_ = e.window.data2;
                glViewport(0, 0, width_, height_);
            }
        }
        return true;
    }

    void Window::swap()
    {
        SDL_GL_SwapWindow(window_);
    }
}
