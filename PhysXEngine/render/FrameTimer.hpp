#pragma once
#include <algorithm>
#include <chrono>
#include <vector>

namespace metis
{
    // Rolling frame-time statistics. Reports the worst frame, which is what
    // the 120 Hz floor is actually about.
    class FrameTimer
    {
    public:
        explicit FrameTimer(std::size_t window = 300) : window_(window)
        {
            samples_.reserve(window);
            last_ = std::chrono::steady_clock::now();
        }

        void tick()
        {
            const auto now = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - last_).count();
            last_ = now;
            if (samples_.size() < window_) samples_.push_back(ms);
            else samples_[cursor_ = (cursor_ + 1) % window_] = ms;
        }

        double averageMs() const
        {
            if (samples_.empty()) return 0.0;
            double sum = 0.0;
            for (double s : samples_) sum += s;
            return sum / static_cast<double>(samples_.size());
        }

        double worstMs() const
        {
            if (samples_.empty()) return 0.0;
            return *std::max_element(samples_.begin(), samples_.end());
        }

        double averageFps() const { const double m = averageMs(); return m > 0.0 ? 1000.0 / m : 0.0; }
        double worstFps() const { const double m = worstMs(); return m > 0.0 ? 1000.0 / m : 0.0; }

    private:
        std::size_t window_;
        std::size_t cursor_ = 0;
        std::vector<double> samples_;
        std::chrono::steady_clock::time_point last_;
    };
}
