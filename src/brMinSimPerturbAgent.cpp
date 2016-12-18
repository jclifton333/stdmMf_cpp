#include "brMinSimPerturbAgent.hpp"

#include "utilities.hpp"
#include "sweepAgent.hpp"
#include "objFns.hpp"
#include "simPerturb.hpp"

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
    this->model_->est_par(inf_bits, history);

    const uint32_t num_points = this->final_t_ - history.size();

    std::vector<BitsetPair> all_history = history;
    all_history.push_back(BitsetPair(inf_bits, boost::dynamic_bitset<>()));


    auto f = [&](const std::vector<double> & par, void * const data) {
        SweepAgent a(this->network_, this->features_, par, 2);

        auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits,
                const boost::dynamic_bitset<> & trt_bits) {
            return dot_a_and_b(par,
                    this->features_->get_features(inf_bits, trt_bits));
        };

        return bellman_residual_sq(all_history, &a, 1.0, q_fn);
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
