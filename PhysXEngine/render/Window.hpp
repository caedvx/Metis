#pragma once

struct SDL_Window;

namespace metis
{
    class Window
    {
    public:
        Window(const char* title, int width, int height);
        ~Window();

        Window(const Window&) = delete;
        Window& operator=(const Window&) = delete;

        // Returns false when the user has asked to quit.
        bool pollEvents();
        void swap();

        int width() const { return width_; }
        int height() const { return height_; }
        SDL_Window* handle() const { return window_; }

        float mouseDeltaX() const { return mouseDx_; }
        float mouseDeltaY() const { return mouseDy_; }
        float scrollDelta() const { return scroll_; }
        bool leftHeld() const { return leftHeld_; }
        bool rightHeld() const { return rightHeld_; }

    private:
        SDL_Window* window_ = nullptr;
        void* glContext_ = nullptr;
        int width_ = 0;
        int height_ = 0;
        float mouseDx_ = 0.0f, mouseDy_ = 0.0f, scroll_ = 0.0f;
        bool leftHeld_ = false, rightHeld_ = false;
    };
}
