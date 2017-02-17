#include "vfnBrAdaptSimPerturbAgent.hpp"

#include <glog/logging.h>

#include <armadillo>

#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

#include "system.hpp"
#include "objFns.hpp"

#include "proximalAgent.hpp"
#include "myopicAgent.hpp"

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
: Agent(network), RngClass(), features_(features), model_(model),

  vfn_num_reps_(vfn_num_reps), vfn_final_t_(vfn_final_t), vfn_c_(vfn_c),
  vfn_t_(vfn_t), vfn_a_(vfn_a), vfn_b_(vfn_b), vfn_ell_(vfn_ell),
  vfn_min_step_size_(vfn_min_step_size),

  br_c_(br_c), br_t_(br_t), br_a_(br_a), br_b_(br_b), br_ell_(br_ell),
  br_min_step_size_(br_min_step_size),

  step_cap_mult_(step_cap_mult) {
}


template <typename State>
VfnBrAdaptSimPerturbAgent<State>::VfnBrAdaptSimPerturbAgent(
        const VfnBrAdaptSimPerturbAgent<State> & other)
    : Agent(other), RngClass(other), features_(other.features_->clone()),
      model_(other.model_->clone()),

      vfn_num_reps_(other.vfn_num_reps_), vfn_final_t_(other.vfn_final_t_),
      vfn_c_(other.vfn_c_), vfn_t_(other.vfn_t_), vfn_a_(other.vfn_a_),
      vfn_b_(other.vfn_b_), vfn_ell_(other.vfn_ell_),
      vfn_min_step_size_(other.vfn_min_step_size_),

      br_c_(other.br_c_), br_t_(other.br_t_), br_a_(other.br_a_),
      br_b_(other.br_b_), br_ell_(other.br_ell_),
      br_min_step_size_(other.br_min_step_size_),

      step_cap_mult_(other.step_cap_mult_) {
}


template <typename State>
std::shared_ptr<Agent<State> > VfnBrAdaptSimPerturbAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(
            new VfnBrAdaptSimPerturbAgent<State>(*this));
}


template <typename State>
boost::dynamic_bitset<> VfnBrAdaptSimPerturbAgent<State>::apply_trt(
        const State & state,
        const std::vector<StateAndTrt<State> > & history) {
    if (history.size() < 1) {
        ProximalAgent<State> a(this->network_);
        return a.apply_trt(state, history);
        // } else if (history.size() < 2) {
        //     MyopicAgent ma(this->network_, this->model_->clone());
        //     return ma.apply_trt(inf_bits, history);
    }


    const std::vector<Transition<State> > all_history(
            Transition<State>::from_sequence(history, state));

    // estimate model
    this->model_->est_par(all_history);


    // get information matrix and take inverse sqrt
    std::vector<double> hess = this->model_->ll_hess(all_history);
    njm::linalg::mult_b_to_a(hess, -1.0 * all_history.size());

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


    std::vector<double> optim_par(this->features_->num_features(), 0.);

    // maximize value function
    {
        const uint32_t num_points = this->vfn_final_t_ - history.size();

        auto f = [&](const std::vector<double> & par, void * const data) {
            SweepAgent<State> a(this->network_, this->features_, par, 2, false);
            a.rng(this->rng());
            System<State> s(this->network_, this->model_);
            s.rng(this->rng());
            double val = 0.0;
            for (uint32_t i = 0; i < this->vfn_num_reps_; ++i) {
                s.cleanse();
                s.wipe_trt();
                s.erase_history();
                s.state(state);

                val += runner(&s, &a, num_points, 0.9);
            }
            val /= this->vfn_num_reps_;

            // return negative since optim minimizes functions
            return -val;
        };

        njm::optim::SimPerturb sp(f, optim_par, NULL, this->vfn_c_,
                this->vfn_t_, this->vfn_a_, this->vfn_b_, this->vfn_ell_,
                this->vfn_min_step_size_);
        sp.rng(this->rng());

        njm::optim::ErrorCode ec;
        do {
            ec = sp.step();
        } while (ec == njm::optim::ErrorCode::CONTINUE);

        CHECK_EQ(ec, njm::optim::ErrorCode::SUCCESS);

        optim_par = sp.par();
    }


    // minimize bellman residual
    {

        // find minimizing scalar for parameters
        {
            SweepAgent<State> a(this->network_, this->features_, optim_par,
                    2, false);

            auto q_fn = [&](const State & state_t,
                    const boost::dynamic_bitset<> & trt_bits_t) {
                return njm::linalg::dot_a_and_b(optim_par,
                        this->features_->get_features(state_t, trt_bits_t));
            };

            const std::vector<std::pair<double, double> > parts =
                bellman_residual_parts(all_history, &a, 0.9, q_fn);

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
                njm::linalg::mult_b_to_a(optim_par, numer / denom);
            } else {
                // other wise just make it norm one
                njm::linalg::mult_b_to_a(optim_par,
                        njm::linalg::l2_norm(optim_par));
            }
        }


        auto f = [&](const std::vector<double> & par, void * const data) {
            SweepAgent<State> a(this->network_, this->features_, par, 2, false);
            a.rng(this->rng());

            auto q_fn = [&](const State & state_t,
                    const boost::dynamic_bitset<> & trt_bits_t) {
                return njm::linalg::dot_a_and_b(par,
                        this->features_->get_features(state_t, trt_bits_t));
            };

            return bellman_residual_sq(all_history, &a, 0.9, q_fn);
        };

        njm::optim::SimPerturb sp(f, optim_par, NULL, this->br_c_, this->br_t_,
                this->br_a_, this->br_b_, this->br_ell_,
                this->br_min_step_size_);
        sp.rng(this->rng());

        njm::optim::ErrorCode ec;
        const uint32_t num_steps = history.size() * this->step_cap_mult_;
        do {
            ec = sp.step();
        } while (ec == njm::optim::ErrorCode::CONTINUE
                && sp.completed_steps() < num_steps);

        CHECK(ec == njm::optim::ErrorCode::SUCCESS
                || (ec == njm::optim::ErrorCode::CONTINUE
                        && sp.completed_steps() == num_steps))
            << "error code: " << ec;

        optim_par = sp.par();
    }

    SweepAgent<State> a(this->network_, this->features_, optim_par, 2, false);
    a.rng(this->rng());
    return a.apply_trt(state, history);
}



template class VfnBrAdaptSimPerturbAgent<InfState>;
template class VfnBrAdaptSimPerturbAgent<InfShieldState>;


} // namespace stdmMf
