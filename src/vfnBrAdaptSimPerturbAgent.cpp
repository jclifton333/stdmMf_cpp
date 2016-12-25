#include "vfnBrAdaptSimPerturbAgent.hpp"

#include <glog/logging.h>

#include "system.hpp"
#include "objFns.hpp"

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
      vfn_min_step_size_(vfn_min_step_size_),

      br_c_(br_c), br_t_(br_t), br_a_(br_a), br_b_(br_b), br_ell_(br_ell),
      br_min_step_size_(br_min_step_size_) {
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
    // estimate model
    this->model_->est_par(inf_bits, history);

    std::vector<double> optim_par(this->features_->num_features(), 0.);

    // maximize value function
    {
        const uint32_t num_points = this->vfn_final_t_ - history.size();

        auto f = [&](const std::vector<double> & par, void * const data) {
            SweepAgent a(this->network_, this->features_, par, 2);
            System s(this->network_, this->model_);
            double val = 0.0;
            for (uint32_t i = 0; i < this->vfn_num_reps_; ++i) {
                s.cleanse();
                s.wipe_trt();
                s.erase_history();
                s.inf_bits(inf_bits);

                val += runner(&s, &a, num_points, 1.0);
            }
            val /= this->vfn_num_reps_;

            // return negative since optim minimizes functions
            return -val;
        };

        SimPerturb sp(f, optim_par, NULL, this->vfn_c_, this->vfn_t_,
                this->vfn_a_, this->vfn_b_, this->vfn_ell_,
                this->vfn_min_step_size_);

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

        auto f = [&](const std::vector<double> & par, void * const data) {
            SweepAgent a(this->network_, this->features_, par, 2);

            auto q_fn = [&](const boost::dynamic_bitset<> & inf_bits,
                    const boost::dynamic_bitset<> & trt_bits) {
                return dot_a_and_b(par,
                        this->features_->get_features(inf_bits, trt_bits));
            };

            return bellman_residual_sq(all_history, &a, 1.0, q_fn);
        };

        SimPerturb sp(f, optim_par, NULL, this->br_c_, this->br_t_, this->br_a_,
                this->br_b_, this->br_ell_, this->br_min_step_size_);

        Optim::ErrorCode ec;
        const uint32_t num_steps = history.size();
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
    return a.apply_trt(inf_bits, history);
}


} // namespace stdmMf