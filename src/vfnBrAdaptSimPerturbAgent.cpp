#include "vfnBrAdaptSimPerturbAgent.hpp"

#include <glog/logging.h>

#include <armadillo>

#include "system.hpp"
#include "objFns.hpp"

#include "proximalAgent.hpp"
#include "myopicAgent.hpp"

namespace stdmMf {


VfnBrAdaptSimPerturbAgent::VfnBrAdaptSimPerturbAgent(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features> & features,
        const std::shared_ptr<Model> & model,
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
        const double & br_min_step_size)
    : Agent(network), features_(features), model_(model),

      vfn_num_reps_(vfn_num_reps), vfn_final_t_(vfn_final_t), vfn_c_(vfn_c),
      vfn_t_(vfn_t), vfn_a_(vfn_a), vfn_b_(vfn_b), vfn_ell_(vfn_ell),
      vfn_min_step_size_(vfn_min_step_size),

      br_c_(br_c), br_t_(br_t), br_a_(br_a), br_b_(br_b), br_ell_(br_ell),
      br_min_step_size_(br_min_step_size) {
}

VfnBrAdaptSimPerturbAgent::VfnBrAdaptSimPerturbAgent(
        const VfnBrAdaptSimPerturbAgent & other)
    : Agent(other), features_(other.features_->clone()),
      model_(other.model_->clone()),

      vfn_num_reps_(other.vfn_num_reps_), vfn_final_t_(other.vfn_final_t_),
      vfn_c_(other.vfn_c_), vfn_t_(other.vfn_t_), vfn_a_(other.vfn_a_),
      vfn_b_(other.vfn_b_), vfn_ell_(other.vfn_ell_),
      vfn_min_step_size_(other.vfn_min_step_size_),

      br_c_(other.br_c_), br_t_(other.br_t_), br_a_(other.br_a_),
      br_b_(other.br_b_), br_ell_(other.br_ell_),
      br_min_step_size_(other.br_min_step_size_){
}

std::shared_ptr<Agent> VfnBrAdaptSimPerturbAgent::clone() const {
    return std::shared_ptr<Agent>(new VfnBrAdaptSimPerturbAgent(*this));
}

boost::dynamic_bitset<> VfnBrAdaptSimPerturbAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits) {
    LOG(FATAL) << "Needs history to apply treatment.";
}


boost::dynamic_bitset<> VfnBrAdaptSimPerturbAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    if (history.size() < 1) {
        ProximalAgent a(this->network_);
        return a.apply_trt(inf_bits, history);
    } else if (history.size() < 1) {
        MyopicAgent ma(this->network_, this->model_->clone());
        return ma.apply_trt(inf_bits, history);
    }

    // estimate model
    this->model_->est_par(inf_bits, history);


    // get information matrix and take inverse sqrt
    std::vector<BitsetPair> all_history(history);
    all_history.push_back(BitsetPair(inf_bits,
                    boost::dynamic_bitset<>(this->network_->size())));
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
    }
    const std::vector<double> par_samp(
            add_a_and_b(this->model_->par(),
                    arma::conv_to<std::vector<double> >::from(
                            var_sqrt * std_norm)));

    // set new parameters
    this->model_->par(par_samp);


    std::vector<double> optim_par(this->features_->num_features(), 0.);

    // maximize value function
    {
        const uint32_t num_points = this->vfn_final_t_ - history.size();

        auto f = [&](const std::vector<double> & par, void * const data) {
            SweepAgent a(this->network_, this->features_, par, 2);
            a.set_rng(this->get_rng());
            System s(this->network_, this->model_);
            s.set_rng(this->get_rng());
            double val = 0.0;
            for (uint32_t i = 0; i < this->vfn_num_reps_; ++i) {
                s.cleanse();
                s.wipe_trt();
                s.erase_history();
                s.inf_bits(inf_bits);

                val += runner(&s, &a, num_points, 0.9);
            }
            val /= this->vfn_num_reps_;

            // return negative since optim minimizes functions
            return -val;
        };

        SimPerturb sp(f, optim_par, NULL, this->vfn_c_, this->vfn_t_,
                this->vfn_a_, this->vfn_b_, this->vfn_ell_,
                this->vfn_min_step_size_);
        sp.set_rng(this->get_rng());

        Optim::ErrorCode ec;
        do {
            ec = sp.step();
        } while (ec == Optim::ErrorCode::CONTINUE);

        CHECK_EQ(ec, Optim::ErrorCode::SUCCESS);

        optim_par = sp.par();
    }


    // minimize bellman residual
    {
        std::vector<BitsetPair> all_history = history;
        all_history.push_back(BitsetPair(inf_bits, boost::dynamic_bitset<>()));

        // find minimizing scalar for parameters
        {
            SweepAgent a(this->network_, this->features_, optim_par, 2);

            auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits_t,
                    const boost::dynamic_bitset<> & trt_bits_t) {
                return dot_a_and_b(optim_par,
                        this->features_->get_features(inf_bits_t, trt_bits_t));
            };

            const std::vector<std::pair<double, double> > parts =
                bellman_residual_parts(history, &a, 0.9, q_fn);

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
                mult_b_to_a(optim_par, numer / denom);
            } else {
                // other wise just make it norm one
                mult_b_to_a(optim_par, l2_norm(optim_par));
            }
        }


        auto f = [&](const std::vector<double> & par, void * const data) {
            SweepAgent a(this->network_, this->features_, par, 2);
            a.set_rng(this->get_rng());

            auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits_t,
                    const boost::dynamic_bitset<> & trt_bits_t) {
                return dot_a_and_b(par,
                        this->features_->get_features(inf_bits_t, trt_bits_t));
            };

            return bellman_residual_sq(all_history, &a, 0.9, q_fn);
        };

        SimPerturb sp(f, optim_par, NULL, this->br_c_, this->br_t_, this->br_a_,
                this->br_b_, this->br_ell_, this->br_min_step_size_);
        sp.set_rng(this->get_rng());

        Optim::ErrorCode ec;
        const uint32_t num_steps = history.size() * 1;
        do {
            ec = sp.step();
        } while (ec == Optim::ErrorCode::CONTINUE
                && sp.completed_steps() < num_steps);

        CHECK(ec == Optim::ErrorCode::SUCCESS
                || (ec == Optim::ErrorCode::CONTINUE
                        && sp.completed_steps() == num_steps));

        optim_par = sp.par();
    }

    SweepAgent a(this->network_, this->features_, optim_par, 2);
    a.set_rng(this->get_rng());
    return a.apply_trt(inf_bits, history);
}


} // namespace stdmMf
