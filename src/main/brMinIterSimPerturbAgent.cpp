#include "brMinIterSimPerturbAgent.hpp"

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
BrMinIterSimPerturbAgent<State>::BrMinIterSimPerturbAgent(
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
        const bool & sq_total_br,
        const uint32_t & obs_per_iter)
    : Agent<State>(network), features_(features),
      c_(c), t_(t), a_(a), b_(b), ell_(ell), min_step_size_(min_step_size),
      do_sweep_(do_sweep), gs_step_(gs_step), sq_total_br_(sq_total_br),
      obs_per_iter_(obs_per_iter) {
}


template <typename State>
BrMinIterSimPerturbAgent<State>::BrMinIterSimPerturbAgent(
        const BrMinIterSimPerturbAgent & other)
    : Agent<State>(other), features_(other.features_->clone()),
      c_(other.c_), t_(other.t_), a_(other.a_),
      b_(other.b_), ell_(other.ell_), min_step_size_(other.min_step_size_),
      do_sweep_(other.do_sweep_), gs_step_(other.gs_step_),
      sq_total_br_(other.sq_total_br_), obs_per_iter_(other.obs_per_iter_) {
}


template <typename State>
std::shared_ptr<Agent<State> > BrMinIterSimPerturbAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(
            new BrMinIterSimPerturbAgent<State>(*this));
}


template <typename State>
boost::dynamic_bitset<> BrMinIterSimPerturbAgent<State>::apply_trt(
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
std::vector<double> BrMinIterSimPerturbAgent<State>::train(
        const std::vector<Transition<State> > & history) {
    BrMinSimPerturbAgent<State> agent (this->network_, this->features_,
            this->c_, this->t_, this->a_, this->b_, this->ell_,
            this->min_step_size_, this->do_sweep_, this->gs_step_,
            this->sq_total_br_, 1);
    agent.rng(this->rng());

    std::vector<double> optim_par(this->features_->num_features(), 0.0);

    if (this->obs_per_iter_ > 0) {

        const uint32_t total_obs = history.size();

        uint32_t num_obs = 0;
        while (num_obs < total_obs) {
            // add more observations each iteration
            num_obs += this->obs_per_iter_;

            const uint32_t num_advance =
                std::min<unsigned long>(total_obs, num_obs);

            const std::vector<Transition<State> > partial_history(
                    history.begin(), history.begin() + num_advance);

            optim_par = agent.train(partial_history, optim_par);
        }
    } else {
        optim_par = agent.train(history, optim_par);
    }

    return optim_par;
}


template<typename State>
void BrMinIterSimPerturbAgent<State>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
}



template class BrMinIterSimPerturbAgent<InfState>;
template class BrMinIterSimPerturbAgent<InfShieldState>;


} // namespace stdmMf
