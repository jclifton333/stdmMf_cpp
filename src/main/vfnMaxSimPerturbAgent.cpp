#include "vfnMaxSimPerturbAgent.hpp"

#include <glog/logging.h>

#include <armadillo>

#include "system.hpp"
#include "objFns.hpp"

#include "proximalAgent.hpp"
#include "myopicAgent.hpp"

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
    : Agent(network), features_(features), model_(model),
      coef_(this->features_->num_features(), 0), num_reps_(num_reps),
      final_t_(final_t), c_(c), t_(t), a_(a), b_(b), ell_(ell),
      min_step_size_(min_step_size) {
}

VfnMaxSimPerturbAgent::VfnMaxSimPerturbAgent(
        const VfnMaxSimPerturbAgent & other)
    : Agent(other), RngClass(other), features_(other.features_->clone()),
      model_(other.model_->clone()), coef_(other.coef_),
      num_reps_(other.num_reps_), final_t_(other.final_t_), c_(other.c_),
      t_(other.t_), a_(other.a_), b_(other.b_), ell_(other.ell_),
      min_step_size_(other.min_step_size_){
}

std::shared_ptr<Agent> VfnMaxSimPerturbAgent::clone() const {
    return std::shared_ptr<Agent>(new VfnMaxSimPerturbAgent(*this));
}

boost::dynamic_bitset<> VfnMaxSimPerturbAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits) {
    LOG(FATAL) << "Needs history to apply treatment.";
}


boost::dynamic_bitset<> VfnMaxSimPerturbAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<InfAndTrt> & history) {
    if (history.size() < 1) {
        ProximalAgent a(this->network_);
        return a.apply_trt(inf_bits, history);
    } else if (history.size() < 2) {
        MyopicAgent ma(this->network_, this->model_->clone());
        return ma.apply_trt(inf_bits, history);
    // } else if (history.size() < 3) {
    //     MyopicAgent ma(this->network_, this->model_->clone());
    //     return ma.apply_trt(inf_bits, history);
    }

    const std::vector<Transition> all_history(
            Transition::from_sequence(history, inf_bits));

    this->model_->est_par(all_history);

    // get information matrix and take inverse sqrt
    std::vector<double> hess = this->model_->ll_hess(all_history);
    mult_b_to_a(hess, -1.0 * (all_history.size() - 1));

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
            add_a_and_b(this->model_->par(),
                    arma::conv_to<std::vector<double> >::from(
                            var_sqrt * std_norm)));

    // set new parameters
    this->model_->par(par_samp);


    const uint32_t num_points = this->final_t_ - history.size();


    auto f = [&](const std::vector<double> & par, void * const data) {
        SweepAgent a(this->network_, this->features_, par, 2, false);
        a.rng(this->rng());
        System s(this->network_, this->model_);
        s.rng(this->rng());
        double val = 0.0;
        for (uint32_t i = 0; i < this->num_reps_; ++i) {
            s.cleanse();
            s.wipe_trt();
            s.erase_history();
            s.inf_bits(inf_bits);

            val += runner(&s, &a, num_points, 1.0);
        }
        val /= this->num_reps_;

        // return negative since optim minimizes functions
        return -val;
    };

    SimPerturb sp(f, std::vector<double>(this->features_->num_features(), 0.),
            NULL, this->c_, this->t_, this->a_, this->b_, this->ell_,
            this->min_step_size_);
    sp.rng(this->rng());

    Optim::ErrorCode ec;
    do {
        ec = sp.step();
    } while (ec == Optim::ErrorCode::CONTINUE);

    CHECK_EQ(ec, Optim::ErrorCode::SUCCESS);

    this->coef_ = sp.par();
    SweepAgent a(this->network_, this->features_, sp.par(), 2, false);
    a.rng(this->rng());
    return a.apply_trt(inf_bits, history);
}


std::vector<double> VfnMaxSimPerturbAgent::coef() const {
    return this->coef_;
}


} // namespace stdmMf
