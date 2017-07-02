#ifndef EBOLA_MODEL_FEATURES_HPP
#define EBOLA_MODEL_FEATURES_HPP


#include "features.hpp"
#include "network.hpp"
#include "model.hpp"

namespace stdmMf {


class EbolaModelFeatures : public Features<EbolaState> {
protected:
    struct Term {
        uint32_t index;
        double weight;
    };

    const std::shared_ptr<const Network> network_;

    const std::shared_ptr<Model<EbolaState> > model_;

    static const uint32_t num_inf_features_;
    static const uint32_t num_not_features_;

    std::vector<std::vector<Term> > terms_;



public:
    EbolaModelFeatures(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Model<EbolaState> > & model);

    EbolaModelFeatures(const EbolaModelFeatures & other);

    virtual ~EbolaModelFeatures() override = default;

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

    virtual uint32_t num_features() const override;
};


} // namespace stdmMf


#endif // EBOLA_MODEL_FEATURES_HPP
