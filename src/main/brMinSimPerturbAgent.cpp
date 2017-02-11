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
        const double & c,
        const double & t,
        const double & a,
        const double & b,
        const double & ell,
        const double & min_step_size)
    : Agent(network), features_(features),
      c_(c), t_(t), a_(a), b_(b), ell_(ell),
      min_step_size_(min_step_size) {
}

BrMinSimPerturbAgent::BrMinSimPerturbAgent(
        const BrMinSimPerturbAgent & other)
    : Agent(other), RngClass(other), features_(other.features_->clone()),
      c_(other.c_), t_(other.t_), a_(other.a_),
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
        const std::vector<InfAndTrt> & history) {
    if (history.size() < 1) {
        ProximalAgent a(this->network_);
        return a.apply_trt(inf_bits, history);
    }

    std::vector<Transition> all_history(
            Transition::from_sequence(history, inf_bits));

    auto f = [&](const std::vector<double> & par, void * const data) {
        SweepAgent a(this->network_, this->features_, par, 2, false);
        a.rng(this->rng());

        auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits_t,
                const boost::dynamic_bitset<> & trt_bits_t) {
            return dot_a_and_b(par,
                    this->features_->get_features(inf_bits_t, trt_bits_t));
        };

        return bellman_residual_sq(all_history, &a, 0.9, q_fn);
    };

    SimPerturb sp(f, std::vector<double>(this->features_->num_features(), 0.),
            NULL, this->c_, this->t_, this->a_, this->b_, this->ell_,
            this->min_step_size_);
    sp.rng(this->rng());

    Optim::ErrorCode ec;
    do {
        ec = sp.step();
    } while (ec == Optim::ErrorCode::CONTINUE);

    const std::vector<double> par(sp.par());

    CHECK_EQ(ec, Optim::ErrorCode::SUCCESS)
        << std::endl
        << "seed: " << this->seed() << std::endl
        << "steps: " << sp.completed_steps() << std::endl
        << "range: [" << *std::min_element(par.begin(), par.end())
        << ", " << *std::max_element(par.begin(), par.end()) << "]"
        << std::endl
        << "c: " << this->c_ << std::endl
        << "t: " << this->t_ << std::endl
        << "a: " << this->a_ << std::endl
        << "b: " << this->b_ << std::endl
        << "ell: " << this->min_step_size_ << std::endl;

    SweepAgent a(this->network_, this->features_, sp.par(), 2, false);
    a.rng(this->rng());
    return a.apply_trt(inf_bits, history);
}



} // namespace stdmMf
