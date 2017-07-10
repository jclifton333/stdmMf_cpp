#ifndef EBOLA_TRANS_PROB_FEATURES_HPP
#define EBOLA_TRANS_PROB_FEATURES_HPP


#include "features.hpp"
#include "network.hpp"
#include "model.hpp"

namespace stdmMf {


class EbolaTransProbFeatures : public Features<EbolaState> {
protected:
    const std::shared_ptr<const Network> network_;

    const std::shared_ptr<Model<EbolaState> > model_;

    std::vector<std::vector<std::vector<double> > > all_probs_;

    std::vector<double> to_probs_;
    std::vector<double> from_probs_;

public:
    EbolaTransProbFeatures(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Model<EbolaState> > & model);

    EbolaTransProbFeatures(const EbolaTransProbFeatures & other);

    virtual ~EbolaTransProbFeatures() override = default;

    virtual std::shared_ptr<Features<EbolaState> > clone() const override;

    virtual void update(const EbolaState & curr_state,
            const std::vector<StateAndTrt<EbolaState> > & history,
            const uint32_t & num_trt) override;

    virtual std::vector<double> get_features(
            const EbolaState & state,
            const boost::dynamic_bitset<> & trt_bits) override;

    virtual void update_features(
            const uint32_t & changed_node,
            const EbolaState & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const EbolaState & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) override;

    virtual void update_features_async(
            const uint32_t & changed_node,
            const EbolaState & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const EbolaState & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) const override;

    virtual void update_all_probs(const EbolaState & curr_state);

    virtual uint32_t num_features() const override;
};


} // namespace stdmMf


#endif // EBOLA_TRANS_PROB_FEATURES_HPP
