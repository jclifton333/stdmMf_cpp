#ifndef EBOLA_FEATURES_HPP
#define EBOLA_FEATURES_HPP

#include "features.hpp"
#include "network.hpp"

namespace stdmMf {


class EbolaFeatures : public Features<EbolaState> {
private:
    struct Term {
        uint32_t index;
        double weight;
    };

    const static uint32_t num_solo_;
    const static uint32_t num_joint_;

    const std::shared_ptr<const Network> network_;

    const uint32_t num_base_locs_;
    const uint32_t num_neigh_;

    std::vector<std::vector<Term> > neither_;
    std::vector<std::vector<Term> > only_inf_;
    std::vector<std::vector<Term> > only_trt_;
    std::vector<std::vector<Term> > both_;

public:
    EbolaFeatures(const std::shared_ptr<const Network> & network,
            const uint32_t & num_base_locs, const uint32_t & num_neigh);

    EbolaFeatures(const EbolaFeatures & other);

    ~EbolaFeatures() override = default;

    std::shared_ptr<Features<EbolaState> > clone() const override;

    std::vector<double> get_features(
            const EbolaState & state,
            const boost::dynamic_bitset<> & trt_bits) override;

    void update_features(
            const uint32_t & changed_node,
            const EbolaState & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const EbolaState & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) override;

    void update_features_async(
            const uint32_t & changed_node,
            const EbolaState & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const EbolaState & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) const override;

    uint32_t num_features() const override;
};


} // namespace stdmMf


#endif // EBOLA_FEATURES_HPP
