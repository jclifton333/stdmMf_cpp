#include "vfnMaxSimPerturbAgent.hpp"

#include <glog/logging.h>

#include <armadillo>

#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

#include "system.hpp"
#include "objFns.hpp"

#include "proximalAgent.hpp"
#include "myopicAgent.hpp"

namespace stdmMf {


template <typename State>
VfnMaxSimPerturbAgent<State>::VfnMaxSimPerturbAgent(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features<State> > & features,
        const std::shared_ptr<Model<State> > & model,
        const uint32_t & num_reps,
        const uint32_t & final_t,
        const double & c,
        const double & t,
        const double & a,
        const double & b,
        const double & ell,
        const double & min_step_size)
    : Agent<State>(network), features_(features), model_(model),
      num_reps_(num_reps), final_t_(final_t), c_(c), t_(t), a_(a), b_(b),
      ell_(ell), min_step_size_(min_step_size),
      last_optim_par_(this->features_->num_features(), 0.0) {
    // share rng
    this->model_->rng(this->rng());
}


template <typename State>
VfnMaxSimPerturbAgent<State>::VfnMaxSimPerturbAgent(
        const VfnMaxSimPerturbAgent<State> & other)
    : Agent<State>(other),
      features_(other.features_->clone()), model_(other.model_->clone()),
      num_reps_(other.num_reps_), final_t_(other.final_t_),
      c_(other.c_), t_(other.t_), a_(other.a_), b_(other.b_), ell_(other.ell_),
      min_step_size_(other.min_step_size_) ,
      last_optim_par_(other.last_optim_par_) {
    // share rng
    this->model_->rng(this->rng());
}


template <typename State>
std::shared_ptr<Agent<State> > VfnMaxSimPerturbAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(
            new VfnMaxSimPerturbAgent<State>(*this));
}


template <typename State>
boost::dynamic_bitset<> VfnMaxSimPerturbAgent<State>::apply_trt(
        const State & curr_state,
        const std::vector<StateAndTrt<State> > & history) {
    if (history.size() < 1) {
        ProximalAgent<State> a(this->network_);
        a.rng(this->rng());
        return a.apply_trt(curr_state, history);
    } else if (history.size() < 2) {
        MyopicAgent<State> ma(this->network_, this->model_);
        ma.rng(this->rng());
        return ma.apply_trt(curr_state, history);
        // } else if (history.size() < 3) {
        //     MyopicAgent<State> ma(this->network_, this->model_->clone());
        //     return ma.apply_trt(state, history);
    }

    const std::vector<Transition<State> > & all_history(
            Transition<State>::from_sequence(history, curr_state));

    const std::vector<double> optim_par = this->train(all_history,
            this->last_optim_par_);

    // store parameter values and scale to norm 1 (don't need to scale
    // optim_par as the policy is scale invariant)
    this->last_optim_par_ = optim_par;
    njm::linalg::mult_b_to_a(this->last_optim_par_,
            1.0 / njm::linalg::l2_norm(this->last_optim_par_));

    // sweep to get treatments
    SweepAgent<State> a(this->network_, this->features_, optim_par,
            njm::linalg::dot_a_and_b, 2, true);
    a.rng(this->rng());
    return a.apply_trt(curr_state, history);
}


template <typename State>
std::vector<double> VfnMaxSimPerturbAgent<State>::train(
        const std::vector<Transition<State> > & history,
        const std::vector<double> & starting_vals) {

    this->model_->est_par(history);

    // get information matrix and take inverse sqrt
    std::vector<double> hess = this->model_->ll_hess(history);
    njm::linalg::mult_b_to_a(hess, -1.0 * history.size());

    const arma::mat hess_mat(hess.data(), this->model_->par_size(),
            this->model_->par_size());
    arma::mat eigvec;
    arma::vec eigval;
    arma::eig_sym(eigval, eigvec, hess_mat);
    for (uint32_t i = 0; i < this->model_->par_size(); ++i) {
        if (eigval(i) > 0.0)
            eigval(i) = std::sqrt(1.0 / eigval(i));
        else
            eigval(i) = 0.0;
    }
    const arma::mat var_sqrt = eigvec * arma::diagmat(eigval) * eigvec.t();

    // sample new parameters
    arma::vec std_norm(this->model_->par_size());
    for (uint32_t i = 0; i < this->model_->par_size(); ++i) {
        std_norm(i) = this->rng_->rnorm_01();
        LOG_IF(FATAL, !std::isfinite(std_norm(i)));
    }
    const std::vector<double> par_samp(
            njm::linalg::add_a_and_b(this->model_->par(),
                    arma::conv_to<std::vector<double> >::from(
                            var_sqrt * std_norm)));

    // set new parameters
    this->model_->par(par_samp);



    CHECK_GT(this->final_t_, history.size());
    const uint32_t num_points = this->final_t_ - history.size();

    const State & curr_state = history.at(history.size() - 1).next_state;


    auto f = [&](const std::vector<double> & par,
            const std::vector<double> & par_orig) {
                 SweepAgent<State> a(this->network_, this->features_,
                         par, njm::linalg::dot_a_and_b, 2, true);
                 a.rng(this->rng());
                 System<State> s(this->network_, this->model_);
                 s.rng(this->rng());
                 double val = 0.0;
                 for (uint32_t i = 0; i < this->num_reps_; ++i) {
                     s.reset();
                     s.state(curr_state);

                     val += runner(&s, &a, num_points, 1.0);
                 }
                 val /= this->num_reps_;

                 // return negative since optim minimizes functions
                 return -val;
             };

    njm::optim::SimPerturb sp(f, starting_vals, this->c_, this->t_,
            this->a_, this->b_, this->ell_, this->min_step_size_);
    sp.rng(this->rng());

    njm::optim::ErrorCode ec;
    do {
        ec = sp.step();
    } while (ec == njm::optim::ErrorCode::CONTINUE);

    CHECK_EQ(ec, njm::optim::ErrorCode::SUCCESS);

    return sp.par();
}



template<typename State>
void VfnMaxSimPerturbAgent<State>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
    this->model_->rng(rng);
}



template class VfnMaxSimPerturbAgent<InfState>;
template class VfnMaxSimPerturbAgent<InfShieldState>;


} // namespace stdmMf
