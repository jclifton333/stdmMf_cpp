#include "optim.hpp"

namespace stdmMf {


Optim::Optim(const std::function<double(std::vector<double>, void*)> & f,
        const std::vector<double> & par,
        void * const data)
    : f_(f), par_(par), par_size_(par.size()), data_(data),
      completed_steps_(0) {
}

std::vector<double> Optim::par() const {
    return this->par_;
}


} // namespace stdmMf
