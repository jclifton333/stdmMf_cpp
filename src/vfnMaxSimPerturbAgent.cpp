#include "vfnMaxSimPerturbAgent.hpp"

#include <glog/logging.h>

#include "system.hpp"
#include "runner.hpp"

namespace stdmMf {


VfnMaxSimPerturbAgent::VfnMaxSimPerturbAgent(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features> & features,
        const std::shared_ptr<Model> & model,
        const uint32_t & num_reps,
        const uint32_t & final_t,
        const double & c,
        const double & t,
        const double & a,
        const double & b,
        const double & ell,
        const double & min_step_size)
    : Agent(network), features_(features), model_(model), num_reps_(num_reps),
      final_t_(final_t), c_(c), t_(t), a_(a), b_(b), ell_(ell),
      min_step_size_(min_step_size_) {
}

boost::dynamic_bitset<> VfnMaxSimPerturbAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    this->model_->est_par(inf_bits, history);

    const uint32_t num_points = this->final_t_ - history.size();


    auto f = [&](const std::vector<double> & par, void * const data) {
        SweepAgent a(this->network_, this->features_, par, 2);
        System s(this->network_, this->model_);
        double val = 0.0;
        for (uint32_t i = 0; i < this->num_reps_; ++i) {
            s.cleanse();
            s.wipe_trt();
            s.erase_history();
            s.inf_bits(inf_bits);

            val += runner(&s, &a, num_points);
        }
        val /= this->num_reps_;

        // return negative since optim minimizes functions
        return -val;
    };

    SimPerturb sp(f, std::vector<double>(this->features_->num_features(), 0.),
            NULL, this->c_, this->t_, this->a_, this->b_, this->ell_,
            this->min_step_size_);

    Optim::ErrorCode ec;
    do {
        ec = sp.step();
    } while (ec == Optim::ErrorCode::CONTINUE);

    CHECK_EQ(ec, Optim::ErrorCode::SUCCESS);

    SweepAgent a(this->network_, this->features_, sp.par(), 2);
    return a.apply_trt(inf_bits, history);
}


} // namespace stdmMf
