#pragma once
#include <algorithm>
#include <chrono>
#include <vector>

namespace metis
{
    // Rolling frame-time stats: average, percentiles, worst. The 120 Hz floor
    // is a p99.9 threshold, because the worst frame keeps getting worse the
    // longer a run goes.
    class FrameTimer
    {
    public:
        explicit FrameTimer(std::size_t window = 300) : window_(std::max<std::size_t>(window, 1))
        {
            samples_.reserve(window_);
            last_ = std::chrono::steady_clock::now();
        }

        void tick()
        {
            const auto now = std::chrono::steady_clock::now();
            const double ms = std::chrono::duration<double, std::milli>(now - last_).count();
            last_ = now;
            if (samples_.size() < window_) {
                samples_.push_back(ms);
            } else {
                samples_[cursor_] = ms;
                cursor_ = (cursor_ + 1) % window_;
            }
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

        // Nearest-rank percentile over the retained samples, p in [0, 1]. The index
        // is clamped so p=1.0 stays in bounds.
        double percentileMs(double p) const
        {
            if (samples_.empty()) return 0.0;
            std::vector<double> sorted(samples_.begin(), samples_.end());
            std::sort(sorted.begin(), sorted.end());
            const std::size_t rank = static_cast<std::size_t>(p * static_cast<double>(sorted.size()));
            return sorted[std::min(rank, sorted.size() - 1)];
        }
        double percentileFps(double p) const { const double m = percentileMs(p); return m > 0.0 ? 1000.0 / m : 0.0; }

        double averageFps() const { const double m = averageMs(); return m > 0.0 ? 1000.0 / m : 0.0; }
        double worstFps() const { const double m = worstMs(); return m > 0.0 ? 1000.0 / m : 0.0; }

        // Keeps whatever ran since the last tick() (e.g. a printf) out of the next
        // frame's interval.
        void resetClock() { last_ = std::chrono::steady_clock::now(); }

    private:
        std::size_t window_;
        std::size_t cursor_ = 0;
        std::vector<double> samples_;
        std::chrono::steady_clock::time_point last_;
    };
}
