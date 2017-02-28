#include "vfnBrAdaptSimPerturbAgent.hpp"

#include <glog/logging.h>

#include <armadillo>

#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

#include "system.hpp"
#include "objFns.hpp"

#include "proximalAgent.hpp"
#include "myopicAgent.hpp"

#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"

namespace stdmMf {


template <typename State>
VfnBrAdaptSimPerturbAgent<State>::VfnBrAdaptSimPerturbAgent(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features<State> > & features,
        const std::shared_ptr<Model<State> > & model,
        const uint32_t & vfn_num_reps,
        const uint32_t & vfn_final_t,
        const double & vfn_c,
        const double & vfn_t,
        const double & vfn_a,
        const double & vfn_b,
        const double & vfn_ell,
        const double & vfn_min_step_size,
        const double & br_c,
        const double & br_t,
        const double & br_a,
        const double & br_b,
        const double & br_ell,
        const double & br_min_step_size,
        const uint32_t & step_cap_mult)
: Agent<State>(network), features_(features), model_(model),

    vfn_num_reps_(vfn_num_reps), vfn_final_t_(vfn_final_t), vfn_c_(vfn_c),
    vfn_t_(vfn_t), vfn_a_(vfn_a), vfn_b_(vfn_b), vfn_ell_(vfn_ell),
    vfn_min_step_size_(vfn_min_step_size),

    br_c_(br_c), br_t_(br_t), br_a_(br_a), br_b_(br_b), br_ell_(br_ell),
    br_min_step_size_(br_min_step_size),

    step_cap_mult_(step_cap_mult) {
    // share rng
    this->model_->rng(this->rng());
}


template <typename State>
VfnBrAdaptSimPerturbAgent<State>::VfnBrAdaptSimPerturbAgent(
        const VfnBrAdaptSimPerturbAgent<State> & other)
    : Agent<State>(other), features_(other.features_->clone()),
    model_(other.model_->clone()),

    vfn_num_reps_(other.vfn_num_reps_), vfn_final_t_(other.vfn_final_t_),
    vfn_c_(other.vfn_c_), vfn_t_(other.vfn_t_), vfn_a_(other.vfn_a_),
    vfn_b_(other.vfn_b_), vfn_ell_(other.vfn_ell_),
    vfn_min_step_size_(other.vfn_min_step_size_),

    br_c_(other.br_c_), br_t_(other.br_t_), br_a_(other.br_a_),
    br_b_(other.br_b_), br_ell_(other.br_ell_),
    br_min_step_size_(other.br_min_step_size_),

    step_cap_mult_(other.step_cap_mult_) {
    // share rng
    this->model_->rng(this->rng());
}


template <typename State>
std::shared_ptr<Agent<State> > VfnBrAdaptSimPerturbAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(
            new VfnBrAdaptSimPerturbAgent<State>(*this));
}


template <typename State>
boost::dynamic_bitset<> VfnBrAdaptSimPerturbAgent<State>::apply_trt(
        const State & curr_state,
        const std::vector<StateAndTrt<State> > & history) {
    if (history.size() < 1) {
        ProximalAgent<State> a(this->network_);
        a.rng(this->rng());
        return a.apply_trt(curr_state, history);
        // } else if (history.size() < 2) {
        //     MyopicAgent ma(this->network_, this->model_->clone());
        //     return ma.apply_trt(inf_bits, history);
    }


    const std::vector<double> optim_par = this->train(curr_state, history,
            std::vector<double>(this->features_->num_features(), 0.0));


    SweepAgent<State> a(this->network_, this->features_, optim_par, 2, false);
    a.rng(this->rng());
    return a.apply_trt(curr_state, history);
}

template <typename State>
std::vector<double> VfnBrAdaptSimPerturbAgent<State>::train(
        const State & curr_state,
        const std::vector<StateAndTrt<State> > & history,
        const std::vector<double> & starting_vals) {


    VfnMaxSimPerturbAgent<State> vfnMaxAgent(this->network_, this->features_,
            this->model_, this->vfn_num_reps_, this->vfn_final_t_, this->vfn_c_,
            this->vfn_t_, this->vfn_a_, this->vfn_b_, this->vfn_ell_,
            this->vfn_min_step_size_);
    vfnMaxAgent.rng(this->rng());
    std::vector<double> vfn_par = vfnMaxAgent.train(curr_state, history,
            starting_vals);


    // find minimizing scalar for parameters
    {
        const std::vector<Transition<State> > all_history(
                Transition<State>::from_sequence(history, curr_state));

        SweepAgent<State> a(this->network_, this->features_, vfn_par,
                2, false);
        a.rng(this->rng());

        auto q_fn = [&](const State & state_t,
                const boost::dynamic_bitset<> & trt_bits_t) {
                        return njm::linalg::dot_a_and_b(vfn_par,
                                this->features_->get_features(state_t,
                                        trt_bits_t));
                    };

        const std::vector<std::pair<double, double> > parts =
            bellman_residual_parts<State>(all_history, &a, 0.9, q_fn, q_fn);

        const double numer = std::accumulate(parts.begin(), parts.end(),
                0.0, [](const double & x,
                        const std::pair<double,double> & a) {
                         return x - a.first * a.second;
                     });

        const double denom = std::accumulate(parts.begin(), parts.end(),
                0.0, [](const double & x,
                        const std::pair<double,double> & a) {
                         return x + a.second * a.second;
                     });

        // scale the par to minimize BR
        if (numer > 0) {
            // if scalor is positive
            njm::linalg::mult_b_to_a(vfn_par, numer / denom);
        } else {
            // other wise just make it norm one
            njm::linalg::mult_b_to_a(vfn_par,
                    njm::linalg::l2_norm(vfn_par));
        }
    }

    // cap number of steps for br min
    const uint32_t num_steps = history.size() * this->step_cap_mult_;
    const double min_step_size = this->br_a_ /
        std::pow(num_steps + this->br_b_, this->br_ell_);
    BrMinSimPerturbAgent<State> brMinAgent(this->network_, this->features_,
            this->br_c_, this->br_t_, this->br_a_, this->br_b_, this->br_ell_,
            std::max(this->br_min_step_size_, min_step_size));
    brMinAgent.rng(this->rng());
    const std::vector<double> br_par = brMinAgent.train(curr_state, history,
            vfn_par);

    return br_par;
}


template<typename State>
void VfnBrAdaptSimPerturbAgent<State>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
    this->model_->rng(rng);
}


template class VfnBrAdaptSimPerturbAgent<InfState>;
template class VfnBrAdaptSimPerturbAgent<InfShieldState>;


} // namespace stdmMf
