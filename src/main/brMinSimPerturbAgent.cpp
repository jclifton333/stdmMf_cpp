#include "brMinSimPerturbAgent.hpp"

#include <njm_cpp/optim/simPerturb.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include "sweepAgent.hpp"
#include "objFns.hpp"

#include "proximalAgent.hpp"

#include <glog/logging.h>

#include <iterator>

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
        const double & min_step_size,
        const bool & do_sweep,
        const bool & gs_step,
        const bool & sq_total_br)
    : Agent<State>(network), features_(features),
      c_(c), t_(t), a_(a), b_(b), ell_(ell), min_step_size_(min_step_size),
      do_sweep_(do_sweep), gs_step_(gs_step), sq_total_br_(sq_total_br),
      record_(false), train_history_() {
}


template <typename State>
BrMinSimPerturbAgent<State>::BrMinSimPerturbAgent(
        const BrMinSimPerturbAgent & other)
    : Agent<State>(other), features_(other.features_->clone()),
      c_(other.c_), t_(other.t_), a_(other.a_),
      b_(other.b_), ell_(other.ell_), min_step_size_(other.min_step_size_),
      do_sweep_(other.do_sweep_), gs_step_(other.gs_step_),
      sq_total_br_(other.sq_total_br_), record_(other.record_),
      train_history_(other.train_history_) {
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
        // use proximal agent when no data is available
        ProximalAgent<State> a(this->network_);
        a.rng(this->rng());
        return a.apply_trt(curr_state, history);
    }

    // use transition form
    const std::vector<Transition<State> > all_history(
            Transition<State>::from_sequence(history, curr_state));

    const std::vector<double> optim_par = this->train(all_history);

    // use sweep agent to determine treatment
    SweepAgent<State> a(this->network_, this->features_, optim_par, 2,
            this->do_sweep_);
    a.rng(this->rng());
    return a.apply_trt(curr_state, history);
}


template <typename State>
std::vector<double> BrMinSimPerturbAgent<State>::train(
        const std::vector<Transition<State> > & history) {
    std::vector<double> starting_vals(this->features_->num_features(), 0.0);
    const std::vector<double> optim_par = this->train(history,
            starting_vals);

    return optim_par;
}




template <typename State>
std::vector<double> BrMinSimPerturbAgent<State>::train(
        const std::vector<Transition<State> > & history,
        const std::vector<double> & starting_vals) {

    // setup optimization function
    auto f = [&](const std::vector<double> & par,
            const std::vector<double> & par_orig) {
                 // q function for time t
                 auto q_fn = [&](const State & state_t,
                         const boost::dynamic_bitset<> & trt_bits_t) {
                                 return njm::linalg::dot_a_and_b(par,
                                         this->features_->get_features(state_t,
                                                 trt_bits_t));
                             };

                 // q function for time t + 1
                 auto q_fn_next = [&](const State & state_t,
                         const boost::dynamic_bitset<> & trt_bits_t) {
                                      return njm::linalg::dot_a_and_b(par_orig,
                                              this->features_->get_features(
                                                      state_t, trt_bits_t));
                                  };

                 if (this->gs_step_ && this->sq_total_br_) {
                     // gauss-seidel step
                     // (E[td-error])^2
                     SweepAgent<State> a(this->network_, this->features_,
                             par_orig, 2, this->do_sweep_);
                     a.rng(this->rng());
                     return sq_bellman_residual<State>(history, &a, 0.9,
                             q_fn, q_fn_next);
                 } else if (this->gs_step_) {
                     // gauss-seidel step
                     // E[(td-error)^2]
                     SweepAgent<State> a(this->network_, this->features_,
                             par_orig, 2, this->do_sweep_);
                     a.rng(this->rng());
                     return bellman_residual_sq<State>(history, &a, 0.9,
                             q_fn, q_fn_next);
                 } else if (this->sq_total_br_) {
                     // update all parameters
                     // (E[td-error])^2
                     SweepAgent<State> a(this->network_, this->features_,
                             par, 2, this->do_sweep_);
                     a.rng(this->rng());
                     return sq_bellman_residual<State>(history, &a, 0.9,
                             q_fn, q_fn);
                 } else {
                     // update all parameters
                     // E[(td-error)^2]
                     SweepAgent<State> a(this->network_, this->features_,
                             par, 2, this->do_sweep_);
                     a.rng(this->rng());
                     return bellman_residual_sq<State>(history, &a, 0.9,
                             q_fn, q_fn);
                 }
             };

    // optimize
    njm::optim::SimPerturb sp(f, starting_vals, this->c_, this->t_,
            this->a_, this->b_, this->ell_, this->min_step_size_);
    sp.rng(this->rng());

    if (this->record_) {
        this->train_history_.clear();
        this->train_history_.emplace_back(sp.obj_fn(), starting_vals);
    }

    njm::optim::ErrorCode ec;
    do {
        ec = sp.step();

        if (this->record_) {
            this->train_history_.emplace_back(sp.obj_fn(), sp.par());
        }
    } while (ec == njm::optim::ErrorCode::CONTINUE);

    // check convergence
    CHECK_EQ(ec, njm::optim::ErrorCode::SUCCESS)
        << std::endl
        << "seed: " << this->seed() << std::endl
        << "steps: " << sp.completed_steps() << std::endl
        << "c: " << this->c_ << std::endl
        << "t: " << this->t_ << std::endl
        << "a: " << this->a_ << std::endl
        << "b: " << this->b_ << std::endl
        << "ell: " << this->min_step_size_ << std::endl;

    return sp.par();
}


template <typename State>
void BrMinSimPerturbAgent<State>::record(const bool & record) {
    this->record_ = record;
}


template <typename State>
std::vector<std::pair<double, std::vector<double> > >
BrMinSimPerturbAgent<State>::train_history() const {
    return this->train_history_;
}



template<typename State>
void BrMinSimPerturbAgent<State>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
}



template class BrMinSimPerturbAgent<InfState>;
template class BrMinSimPerturbAgent<InfShieldState>;


} // namespace stdmMf
