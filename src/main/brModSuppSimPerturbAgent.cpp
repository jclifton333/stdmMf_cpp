#include "brModSuppSimPerturbAgent.hpp"

#include "sweepAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "epsAgent.hpp"
#include "brMinSimPerturbAgent.hpp"

#include "system.hpp"

#include <glog/logging.h>

namespace stdmMf {


template <typename State>
BrModSuppSimPerturbAgent<State>::BrModSuppSimPerturbAgent(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features<State> > & features,
        const std::shared_ptr<Model<State> > & model,
        const double & c,
        const double & t,
        const double & a,
        const double & b,
        const double & ell,
        const double & min_step_size,
        const bool & do_sweep,
        const bool & gs_step,
        const bool & sq_total_br,
        const uint32_t & num_points)
    : Agent<State>(network), features_(features), model_(model),
      c_(c), t_(t), a_(a), b_(b), ell_(ell), min_step_size_(min_step_size),
      do_sweep_(do_sweep), gs_step_(gs_step), sq_total_br_(sq_total_br),
      num_points_(num_points) {
    // share rng
    this->model_->rng(this->rng());
}


template <typename State>
BrModSuppSimPerturbAgent<State>::BrModSuppSimPerturbAgent(
        const BrModSuppSimPerturbAgent & other)
    : Agent<State>(other), features_(other.features_->clone()),
      model_(other.model_->clone()), c_(other.c_), t_(other.t_), a_(other.a_),
      b_(other.b_), ell_(other.ell_), min_step_size_(other.min_step_size_),
      do_sweep_(other.do_sweep_), gs_step_(other.gs_step_),
      sq_total_br_(other.sq_total_br_), num_points_(other.num_points_) {
    // share rng
    this->model_->rng(this->rng());
}


template <typename State>
std::shared_ptr<Agent<State> > BrModSuppSimPerturbAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(
            new BrModSuppSimPerturbAgent<State>(*this));
}


template <typename State>
boost::dynamic_bitset<> BrModSuppSimPerturbAgent<State>::apply_trt(
        const State & curr_state,
        const std::vector<StateAndTrt<State> > & history) {
    if (history.size() < 1) {
        ProximalAgent<State> a(this->network_);
        a.rng(this->rng());
        return a.apply_trt(curr_state, history);
    }

    const std::vector<Transition<State> > all_history(
            Transition<State>::from_sequence(history, curr_state));

    const std::vector<double> optim_par = this->train(all_history,
            std::vector<double>(this->features_->num_features(), 0.0));

    SweepAgent<State> a(this->network_, this->features_, optim_par, 2,
            this->do_sweep_);
    a.rng(this->rng());
    return a.apply_trt(curr_state, history);
}


template <typename State>
std::vector<double> BrModSuppSimPerturbAgent<State>::train(
        const std::vector<Transition<State> > & history,
        const std::vector<double> & starting_vals) {

    // std::vector<Transition<State> > supp_history(history);
    std::vector<Transition<State> > supp_history;
    if (supp_history.size() > this->num_points_) {
        typename std::vector<Transition<State> >::iterator it(
                supp_history.begin());
        std::advance(it, supp_history.size() - this->num_points_);
        supp_history.erase(supp_history.begin(), it);
    } else if (supp_history.size() < this->num_points_) {
        // this->model_->est_par(history);
        System<State> s(this->network_, this->model_);
        s.rng(this->rng());

        // eps agent
        std::shared_ptr<ProximalAgent<State> > pa(
                new ProximalAgent<State>(this->network_));
        pa->rng(this->rng());
        std::shared_ptr<RandomAgent<State> > ra(
                new RandomAgent<State>(this->network_));
        ra->rng(this->rng());
        EpsAgent<State> ea(this->network_, pa, ra, 0.2);
        ea.rng(this->rng());

        s.start();
        const uint32_t num_to_supp = this->num_points_ - supp_history.size();
        for (uint32_t i = 0; i < num_to_supp; i++) {
            const boost::dynamic_bitset<> trt_bits = ea.apply_trt(s.state(),
                    s.history());

            s.trt_bits(trt_bits);

            s.turn_clock();
        }

        const std::vector<Transition<State> > trans_to_supp(
                Transition<State>::from_sequence(s.history(), s.state()));

        supp_history.insert(supp_history.end(), trans_to_supp.begin(),
                trans_to_supp.end());

        CHECK_EQ(supp_history.size(), this->num_points_);
    }

    BrMinSimPerturbAgent<State> brMinAgent(this->network_, this->features_,
            this->c_, this->t_, this->a_, this->b_, this->ell_,
            this->min_step_size_, this->do_sweep_, this->gs_step_,
            this->sq_total_br_);
    brMinAgent.rng(this->rng());

    return brMinAgent.train(supp_history,
            starting_vals);
}


template <typename State>
void BrModSuppSimPerturbAgent<State>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
    this->model_->rng(rng);
}



template class BrModSuppSimPerturbAgent<InfState>;
template class BrModSuppSimPerturbAgent<InfShieldState>;


} // namespace stdmMf
