#include "brMinSimPerturbAgent.hpp"

#include "utilities.hpp"
#include "sweepAgent.hpp"
#include "objFns.hpp"
#include "simPerturb.hpp"

#include "proximalAgent.hpp"

#include <glog/logging.h>

namespace stdmMf {


BrMinSimPerturbAgent::BrMinSimPerturbAgent(
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

BrMinSimPerturbAgent::BrMinSimPerturbAgent(
        const BrMinSimPerturbAgent & other)
    : Agent(other), features_(other.features_->clone()),
      model_(other.model_->clone()), num_reps_(other.num_reps_),
      final_t_(other.final_t_), c_(other.c_), t_(other.t_), a_(other.a_),
      b_(other.b_), ell_(other.ell_), min_step_size_(other.min_step_size_){
}

std::shared_ptr<Agent> BrMinSimPerturbAgent::clone() const {
    return std::shared_ptr<Agent>(new BrMinSimPerturbAgent(*this));
}

boost::dynamic_bitset<> BrMinSimPerturbAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits) {
    LOG(FATAL) << "Needs history to apply treatment.";
}


boost::dynamic_bitset<> BrMinSimPerturbAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    if (history.size() <= 1) {
        ProximalAgent a(this->network_);
        return a.apply_trt(inf_bits, history);
    }

    this->model_->est_par(inf_bits, history);

    std::vector<BitsetPair> all_history = history;
    all_history.push_back(BitsetPair(inf_bits, boost::dynamic_bitset<>()));

    auto f = [&](const std::vector<double> & par, void * const data) {
        SweepAgent a(this->network_, this->features_, par, 2);
        a.set_rng(this->get_rng());

        auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits_t,
                const boost::dynamic_bitset<> & trt_bits_t) {
            return dot_a_and_b(par,
                    this->features_->get_features(inf_bits_t, trt_bits_t));
        };

        return bellman_residual_sq(all_history, &a, 1.0, q_fn);
    };

    SimPerturb sp(f, std::vector<double>(this->features_->num_features(), 0.),
            NULL, this->c_, this->t_, this->a_, this->b_, this->ell_,
            this->min_step_size_);
    sp.set_rng(this->get_rng());

    Optim::ErrorCode ec;
    do {
        ec = sp.step();
    } while (ec == Optim::ErrorCode::CONTINUE);

    CHECK_EQ(ec, Optim::ErrorCode::SUCCESS);

    SweepAgent a(this->network_, this->features_, sp.par(), 2);
    a.set_rng(this->get_rng());
    return a.apply_trt(inf_bits, history);
}



} // namespace stdmMf
