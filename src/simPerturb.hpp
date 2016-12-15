#ifndef SIM_PERTURB_HPP
#define SIM_PERTURB_HPP

#include "optim.hpp"

namespace stdmMf {


class SimPerturb : public Optim {
private:
    const double c_;
    const double t_;
    const double a_;
    const double b_;
    const double ell_;
    const double min_step_size_;

    double step_size() const;

public:
    SimPerturb(const std::function<double(std::vector<double>, void*)> & f,
            const std::vector<double> & par,
            void * const data,
            const double & c,
            const double & t,
            const double & a,
            const double & b,
            const double & ell,
            const double & min_step_size);

    Optim::ErrorCode step();
};


} // namespace stdmMf


#endif // SIM_PERTURB_HPP
