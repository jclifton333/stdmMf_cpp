#ifndef FINITE_QFN_FEATURES_HPP
#define FINITE_QFN_FEATURES_HPP


#include "model.hpp"
#include "features.hpp"

namespace stdmMf {

// TODO: write test cases for finite qfn features
template <typename State>
class FiniteQfnFeatures : public Features<State> {
protected:
    const std::shared_ptr<const Network> network_;
    const uint32_t num_nodes_;
    const std::vector<std::shared_ptr<Model<State> > > models_;
    const uint32_t num_models_;
    const std::shared_ptr<Features<State> > features_;
    const uint32_t look_ahead_;
    std::vector<std::vector<std::vector<double> > > coef_;

    std::vector<double> last_feat_;

public:
    FiniteQfnFeatures(const std::shared_ptr<const Network> & network,
            const std::vector<std::shared_ptr<Model<State> > > & model,
            const std::shared_ptr<Features<State> > & features,
            const uint32_t & look_ahead);

    FiniteQfnFeatures(const FiniteQfnFeatures<State> & other);

    virtual ~FiniteQfnFeatures() = default;

    virtual std::shared_ptr<Features<State> > clone() const override;

    virtual void update(const State & curr_state,
            const std::vector<StateAndTrt<State> > & history,
            const uint32_t & num_trt) override;

    virtual std::vector<double> get_features(
            const State & state,
            const boost::dynamic_bitset<> & trt_bits) override;

    virtual void update_features(
            const uint32_t & changed_node,
            const State & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const State & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) override;

    virtual void update_features_async(
            const uint32_t & changed_node,
            const State & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const State & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) const override;

    virtual uint32_t num_features() const override;

    virtual std::vector<std::vector<Transition<State> > > generate_data(
            const uint32_t & num_episodes,
            const uint32_t & num_obs_per_episode);

    virtual void fit_q_function(const uint32_t & qfn_index,
            const uint32_t & model_index,
            const std::vector<StateAndTrt<State> > & state_trt_train,
            const std::vector<double> & outcomes_train,
            const std::vector<StateAndTrt<State> > & state_trt_test,
            const std::vector<double> & outcomes_test);

    virtual void fit_q_functions(
            const std::vector<std::vector<Transition<State> > > & obs);

    using njm::tools::RngClass::rng;
    virtual void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};



} // namespace stdmMf


#endif // FINITE_QFN_FEATURES_HPP
