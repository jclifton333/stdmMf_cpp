#include "simPerturb.hpp"

#include <iostream>
#include <cmath>

namespace stdmMf {


SimPerturb::SimPerturb(
        const std::function<double(const std::vector<double> & ,
                void * const)> & f,
        const std::vector<double> & par,
        void * const data,
        const double & c,
        const double & t,
        const double & a,
        const double & b,
        const double & ell,
        const double & min_step_size)
    : Optim(f, par, data), c_(c), t_(t), a_(a), b_(b), ell_(ell),
      min_step_size_(min_step_size) {
}

Optim::ErrorCode SimPerturb::step() {
    const double scale = this->c_ /
        std::pow(this->completed_steps_ + 1, this->t_);

    std::vector<double> perturb;
    perturb.reserve(this->par_size_);
    for (uint32_t i = 0; i < this->par_size_; ++i) {
        if (this->rng->runif_01() < 0.5)
            perturb.push_back(-scale);
        else
            perturb.push_back(scale);
    }

    const std::vector<double> par_plus = add_a_and_b(this->par(), perturb);

    const std::vector<double> par_minus = add_a_and_b(this->par(),
            mult_a_and_b(perturb, -1.));

    const double val_plus = this->f_(par_plus, this->data_);
    const double val_minus = this->f_(par_minus, this->data_);

    // estimate gradient
    const std::vector<double> grad_est = recip_of(mult_a_and_b(perturb,
                    2.0 / (val_plus - val_minus)));


    const double step_size = this->a_ /
        std::pow(this->b_ + this->completed_steps_, this->ell_);

    // negative step_size for minimization
    add_b_to_a(this->par_, mult_a_and_b(grad_est, - step_size));

    ++this->completed_steps_;

    if (this->verbose_) {
        // evaluate and print
        const double val = this->f_(this->par_, this->data_);
        std::cout << "iter: " << this->completed_steps_ << std::endl;
        std::cout << "par:";
        for (uint32_t i = 0; i < this->par_.size(); ++i) {
            std::cout << " " << this->par_.at(i);
        }
        std::cout << std::endl;
        std::cout << "value: " << val << std::endl;
    }

    if (std::any_of(this->par_.begin(), this->par_.end(),
                    [](const double & x) { return !std::isfinite(x);})) {
        return Optim::NON_FINITE_PARAMETER;
    } else if (step_size < this->min_step_size_) {
        return Optim::SUCCESS;
    } else {
        return Optim::CONTINUE;
    }
}


} // namespace stdmMf
