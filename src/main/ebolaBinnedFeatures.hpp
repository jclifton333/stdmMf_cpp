#ifndef EBOLA_BINNED_FEATURES_HPP
#define EBOLA_BINNED_FEATURES_HPP


#include "features.hpp"
#include "network.hpp"


namespace stdmMf {


class EbolaBinnedFeatures : public Features<EbolaState> {
protected:
    struct Term {
        uint32_t index;
        double weight;
    };

    const std::shared_ptr<const Network> network_;

    const uint32_t num_bins_;
    const uint32_t num_per_bin_;
    const uint32_t num_extra_;
    static const uint32_t num_features_per_bin_;
    const uint32_t num_neigh_;

    const std::vector<uint32_t> bins_;
    // a vector of neighbors for each location
    const std::vector<std::vector<uint32_t> > neigh_;

    const std::vector<std::vector<Term> > terms_;



public:
    EbolaBinnedFeatures(const std::shared_ptr<const Network> & network,
            const uint32_t num_bins, const uint32_t num_neigh);

    EbolaBinnedFeatures(const EbolaBinnedFeatures & other);

    virtual ~EbolaBinnedFeatures() override = default;

    virtual std::shared_ptr<Features<EbolaState> > clone() const override;

    virtual void update(const EbolaState & curr_state,
            const std::vector<StateAndTrt<EbolaState> > & history,
            const uint32_t & num_trt) override { /* do nothing by default */};

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

    std::vector<uint32_t> get_bins() const;

    std::vector<std::vector<uint32_t> > get_neigh() const;

    std::vector<std::vector<Term> > get_terms() const;
};


} // namespace stdmMf


#endif // EBOLA_BINNED_FEATURES_HPP
