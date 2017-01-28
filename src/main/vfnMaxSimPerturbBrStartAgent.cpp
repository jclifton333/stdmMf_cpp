#include "vfnMaxSimPerturbBrStartAgent.hpp"

#include <glog/logging.h>

#include <armadillo>

#include "system.hpp"
#include "objFns.hpp"

#include "proximalAgent.hpp"
#include "myopicAgent.hpp"

namespace stdmMf {


VfnMaxSimPerturbBrStartAgent::VfnMaxSimPerturbBrStartAgent(
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

VfnMaxSimPerturbBrStartAgent::VfnMaxSimPerturbBrStartAgent(
        const VfnMaxSimPerturbBrStartAgent & other)
    : Agent(other), features_(other.features_->clone()),
      model_(other.model_->clone()), coef_(other.coef_),
      num_reps_(other.num_reps_), final_t_(other.final_t_), c_(other.c_),
      t_(other.t_), a_(other.a_), b_(other.b_), ell_(other.ell_),
      min_step_size_(other.min_step_size_){
}

std::shared_ptr<Agent> VfnMaxSimPerturbBrStartAgent::clone() const {
    return std::shared_ptr<Agent>(new VfnMaxSimPerturbBrStartAgent(*this));
}

boost::dynamic_bitset<> VfnMaxSimPerturbBrStartAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits) {
    LOG(FATAL) << "Needs history to apply treatment.";
}


boost::dynamic_bitset<> VfnMaxSimPerturbBrStartAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
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
    // br minimization first to get starting value

    std::vector<double> starting_value;
    {
        std::vector<BitsetPair> all_history = history;
        all_history.push_back(BitsetPair(inf_bits, boost::dynamic_bitset<>()));

        auto f = [&](const std::vector<double> & par, void * const data) {
            SweepAgent a(this->network_, this->features_, par, 2, false);
            a.set_rng(this->get_rng());

            auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits_t,
                    const boost::dynamic_bitset<> & trt_bits_t) {
                return dot_a_and_b(par,
                        this->features_->get_features(inf_bits_t, trt_bits_t));
            };

            return bellman_residual_sq(all_history, &a, 0.9, q_fn);
        };

        SimPerturb sp(f,
                std::vector<double>(this->features_->num_features(), 0.),
                NULL, this->c_, this->t_, this->a_, this->b_, this->ell_,
                this->min_step_size_);
        sp.set_rng(this->get_rng());

        Optim::ErrorCode ec;
        do {
            ec = sp.step();
        } while (ec == Optim::ErrorCode::CONTINUE);

        CHECK_EQ(ec, Optim::ErrorCode::SUCCESS);

        starting_value = sp.par();
    }



    std::vector<BitsetPair> all_history(history);
    all_history.push_back(BitsetPair(inf_bits,
                    boost::dynamic_bitset<>(this->network_->size())));

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
        std_norm(i) = this->rng->rnorm_01();
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
        a.set_rng(this->get_rng());
        System s(this->network_, this->model_);
        s.set_rng(this->get_rng());
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

    SimPerturb sp(f, starting_value,
            NULL, this->c_, this->t_, this->a_, this->b_, this->ell_,
            this->min_step_size_);
    sp.set_rng(this->get_rng());

    Optim::ErrorCode ec;
    do {
        ec = sp.step();
    } while (ec == Optim::ErrorCode::CONTINUE);

    CHECK_EQ(ec, Optim::ErrorCode::SUCCESS);

    this->coef_ = sp.par();
    SweepAgent a(this->network_, this->features_, sp.par(), 2, false);
    a.set_rng(this->get_rng());
    return a.apply_trt(inf_bits, history);
}


std::vector<double> VfnMaxSimPerturbBrStartAgent::coef() const {
    return this->coef_;
}


} // namespace stdmMf
