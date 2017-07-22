#include "mixtureModel.hpp"
#include "infStateModel.hpp"
#include "infShieldStateModel.hpp"

#include "utilities.hpp"

#include <glog/logging.h>

#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

#include <memory>

namespace stdmMf {


template <typename State, typename Mod>
MixtureModel<State, Mod>::MixtureModel(
        const std::vector<std::shared_ptr<Mod> > & models,
        const std::vector<double> & weights,
        const std::shared_ptr<const Network> & network)
    : Model<State>(std::accumulate(models.begin(), models.end(), 0u,
                    [] (const uint32_t & a_,
                            const std::shared_ptr<Mod> & mod_) {
                        return a_ + mod_->par_size();
                    }),
            network),
      models_(models), num_models_(this->models_.size()), weights_(weights) {
    CHECK(std::all_of(weights.begin(), weights.end(),
                    [] (const double & x_) {
                        return x_ >= 0;
                    }));
    CHECK_NEAR(std::accumulate(weights.begin(), weights.end(), 0.0,
                    [] (const double & a_, const double & x_) {
                        return a_ + x_;
                    }), 1.0, 1e-10);
}


template <typename State, typename Mod>
MixtureModel<State, Mod>::MixtureModel(const MixtureModel<State, Mod> & other)
    : Model<State>(other.par_size(), other.network_),
      models_(clone_vec(other.models_)), num_models_(other.num_models_),
      weights_(other.weights_) {
}


template <typename State, typename Mod>
std::shared_ptr<Model<State> > MixtureModel<State, Mod>::clone() const {
    return std::shared_ptr<Model<State> > (new MixtureModel<State, Mod>(*this));
}


template <typename State, typename Mod>
std::vector<double> MixtureModel<State, Mod>::par() const {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return std::vector<double>();
}


template <typename State, typename Mod>
void MixtureModel<State, Mod>::par(const std::vector<double> & par) {
    LOG(FATAL) << "NOT IMPLEMENTED";
}


template <typename State, typename Mod>
std::vector<double> MixtureModel<State, Mod>::probs(
        const State & curr_state,
        const boost::dynamic_bitset<> & trt_bits) const {
    std::vector<double> weighted_probs(this->network_->size(), 0.0);
    for (uint32_t i = 0; i < this->num_models_; ++i) {
        std::vector<double> probs_i(
                this->models_.at(i)->probs(curr_state, trt_bits));
        njm::linalg::mult_b_to_a(probs_i, this->weights_.at(i));
        njm::linalg::add_b_to_a(weighted_probs, probs_i);
    }
    return weighted_probs;
}


template <typename State, typename Mod>
double MixtureModel<State, Mod>::ll(
        const std::vector<Transition<State> > & history) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return 0.0;
}


template <typename State, typename Mod>
std::vector<double> MixtureModel<State, Mod>::ll_grad(
        const std::vector<Transition<State> > & history) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return std::vector<double>();
}


template <typename State, typename Mod>
std::vector<double> MixtureModel<State, Mod>::ll_hess(
        const std::vector<Transition<State> > & history) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
    return std::vector<double>();
}


template <>
InfState MixtureModel<InfState, InfStateModel>::turn_clock(
        const InfState & curr_state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const std::vector<double> probs(this->probs(curr_state, trt_bits));

    InfState next_state(curr_state);
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        const double & prob_i = probs.at(i);

        const double r = this->rng_->runif_01();
        if (r < prob_i) {
            next_state.inf_bits.flip(i);
        }
    }

    return next_state;
}


template <>
InfShieldState MixtureModel<InfShieldState, InfShieldStateModel>::turn_clock(
        const InfShieldState & curr_state,
        const boost::dynamic_bitset<> & trt_bits) const {

    const uint32_t m(this->rng()->rint(0, this->num_models_));

    const std::vector<double> probs(
            this->models_.at(m)->probs(curr_state, trt_bits));

    InfShieldState next_state(curr_state);
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        const double & prob_i = probs.at(i);

        const double r = this->rng_->runif_01();
        if (r < prob_i) {
            next_state.inf_bits.flip(i);
        }

        next_state.shield.at(i) =
            this->models_.at(m)->shield_draw(i, curr_state);
    }

    return next_state;
}


template <typename State, typename Mod>
void MixtureModel<State, Mod>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
    std::for_each(this->models_.begin(), this->models_.end(),
            [&rng] (const std::shared_ptr<Mod > & mod_) {
                mod_->rng(rng);
            });
}


template class MixtureModel<InfState, InfStateModel>;
template class MixtureModel<InfShieldState, InfShieldStateModel>;



} // namespace stdmMf
