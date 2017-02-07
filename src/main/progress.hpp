#ifndef PROGRESS_HPP
#define PROGRESS_HPP

#include <mutex>
#include <iomanip>
#include <chrono>

namespace stdmMf {


template <class T>
class Progress {
protected:
    std::mutex mutex_;

    uint32_t num_done_;
    uint32_t total_;

    const uint32_t bar_width_;

    T * out_;

    std::chrono::time_point<std::chrono::system_clock> tick_;

    void print() {
        const std::chrono::time_point<std::chrono::system_clock> tock
            = std::chrono::system_clock::now();

        *this->out_ << "\r" << std::setw(5) << std::setfill(' ') << num_done_;
        if (total_ > 0) {
            *this->out_ << " / " << std::setw(5) << std::setfill(' ') << total_
                       << " [";
            const uint32_t num_blocks_ = num_done_ * bar_width_ / total_;
            for (uint32_t i = 0; i < bar_width_; ++i) {
                if (i < num_blocks_) {
                    *this->out_ << "#";
                } else {
                    *this->out_ << " ";
                }
            }
            *this->out_ << "]";
        }

        using FpHours =
            std::chrono::duration<double, std::chrono::hours::period>;
        FpHours elapsed = FpHours(tock - this->tick_);

        *this->out_ << " (" << std::setw(6) << std::setfill(' ')
                    << std::fixed << std::setprecision(2)
                    << elapsed.count() << " hours)";
        *this->out_ << std::flush;
    }

public:
    Progress(T * out)
        : Progress(0, out) {
    }

    Progress(const uint32_t & total, T * out)
        : num_done_(0), total_(total), bar_width_(20), out_(out),
          tick_(std::chrono::system_clock::now()) {
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->print();
    }

    void update() {
        std::lock_guard<std::mutex> lock(this->mutex_);
        ++num_done_;
        this->print();
    }

    void total(const uint32_t & total) {
        std::lock_guard<std::mutex> lock(this->mutex_);
        this->total_ = total;
        this->print();
    }

    uint32_t total() {
        return this->total_;
    };
};


} // namespace stdmMf


#endif // PROGRESS_HPP
