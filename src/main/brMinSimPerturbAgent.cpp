#include "brMinSimPerturbAgent.hpp"

#include <njm_cpp/tools/random.hpp>
#include <njm_cpp/optim/simPerturb.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include "sweepAgent.hpp"
#include "objFns.hpp"

#include "proximalAgent.hpp"

#include <glog/logging.h>

namespace stdmMf {


template <typename State>
BrMinSimPerturbAgent<State>::BrMinSimPerturbAgent(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features<State> > & features,
        const double & c,
        const double & t,
        const double & a,
        const double & b,
        const double & ell,
        const double & min_step_size)
    : Agent<State>(network), features_(features),
    c_(c), t_(t), a_(a), b_(b), ell_(ell),
    min_step_size_(min_step_size) {
}


template <typename State>
BrMinSimPerturbAgent<State>::BrMinSimPerturbAgent(
        const BrMinSimPerturbAgent & other)
    : Agent<State>(other), RngClass(other), features_(other.features_->clone()),
    c_(other.c_), t_(other.t_), a_(other.a_),
    b_(other.b_), ell_(other.ell_), min_step_size_(other.min_step_size_){
}


template <typename State>
std::shared_ptr<Agent<State> > BrMinSimPerturbAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(
            new BrMinSimPerturbAgent<State>(*this));
}


template <typename State>
boost::dynamic_bitset<> BrMinSimPerturbAgent<State>::apply_trt(
        const State & curr_state,
        const std::vector<StateAndTrt<State> > & history) {
    if (history.size() < 1) {
        ProximalAgent<State> a(this->network_);
        return a.apply_trt(curr_state, history);
    }

    std::vector<Transition<State> > all_history(
            Transition<State>::from_sequence(history, curr_state));

    auto f = [&](const std::vector<double> & par, void * const data) {
        SweepAgent<State> a(this->network_, this->features_, par, 2, false);
        a.rng(this->rng());

        auto q_fn = [&](const State & state_t,
                const boost::dynamic_bitset<> & trt_bits_t) {
            return njm::linalg::dot_a_and_b(par,
                    this->features_->get_features(state_t, trt_bits_t));
        };

        return bellman_residual_sq<State>(all_history, &a, 0.9, q_fn);
    };

    njm::optim::SimPerturb sp(f,
            std::vector<double>(this->features_->num_features(), 0.),
            NULL, this->c_, this->t_, this->a_, this->b_, this->ell_,
            this->min_step_size_);
    sp.rng(this->rng());

    njm::optim::ErrorCode ec;
    do {
        ec = sp.step();
    } while (ec == njm::optim::ErrorCode::CONTINUE);

    CHECK_EQ(ec, njm::optim::ErrorCode::SUCCESS)
        << std::endl
        << "seed: " << this->seed() << std::endl
        << "steps: " << sp.completed_steps() << std::endl
        << "c: " << this->c_ << std::endl
        << "t: " << this->t_ << std::endl
        << "a: " << this->a_ << std::endl
        << "b: " << this->b_ << std::endl
        << "ell: " << this->min_step_size_ << std::endl;

    SweepAgent<State> a(this->network_, this->features_, sp.par(), 2, false);
    a.rng(this->rng());
    return a.apply_trt(curr_state, history);
}


template class BrMinSimPerturbAgent<InfState>;
template class BrMinSimPerturbAgent<InfShieldState>;


} // namespace stdmMf
