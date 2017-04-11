#ifndef FINITE_QFN_FEATURES_HPP
#define FINITE_QFN_FEATURES_HPP


#include "model.hpp"
#include "features.hpp"
#include "neuralNetwork.hpp"

namespace stdmMf {

// TODO: write test cases for finite qfn features
template <typename State>
class FiniteQfnFeatures : public Features<State> {
protected:
    const std::shared_ptr<const Network> network_;
    const std::shared_ptr<Model<State> > model_;
    const uint32_t num_nodes_;
    const uint32_t look_ahead_;

    std::vector<NeuralNetwork<State> > nn_;

public:
    FiniteQfnFeatures(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Model<State> > & model,
            const uint32_t & look_ahead);

    FiniteQfnFeatures(const FiniteQfnFeatures<State> & other);

    virtual ~FiniteQfnFeatures() = default;

    virtual std::shared_ptr<Features<State> > clone() const override;

    virtual void update(const State & curr_state,
            const std::vector<StateAndTrt<State> > & history) override;

    virtual std::vector<Transition<State> > generate_data(
            const uint32_t & num_episodes,
            const uint32_t & num_obs_per_episode);

    virtual void fit_q_functions(const std::vector<Transition<State> > & obs);

    using njm::tools::RngClass::rng;
    virtual void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};



} // namespace stdmMf


#endif // FINITE_QFN_FEATURES_HPP
