#include "optim.hpp"

namespace stdmMf {


Optim::Optim(const std::function<double(const std::vector<double> & ,
                void * const)> & f,
        const std::vector<double> & par,
        void * const data)
    : f_(f), par_(par), par_size_(par.size()), data_(data),
      completed_steps_(0), verbose_(false) {
}

void Optim::verbose(const bool & verbose) {
    this->verbose_ = verbose;
}

std::vector<double> Optim::par() const {
    return this->par_;
}

uint32_t Optim::completed_steps() const {
    return this->completed_steps_;
}


} // namespace stdmMf
