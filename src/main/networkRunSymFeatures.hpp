#ifndef NETWORK_RUN_SYM_FEATURES_HPP
#define NETWORK_RUN_SYM_FEATURES_HPP

#include <vector>
#include <boost/dynamic_bitset.hpp>
#include "network.hpp"
#include "features.hpp"

namespace stdmMf {

template <typename State>
class NetworkRunSymFeatures : public Features<State> {
private:
    const std::shared_ptr<const Network> network_;

    const static uint32_t bits_per_node_;

    const std::vector<NetworkRun> runs_;
    const std::vector<std::vector<NetworkRun> > runs_by_node_;
    const uint32_t num_nodes_;
    const uint32_t run_length_;
    const uint32_t num_runs_;
    std::vector<std::vector<uint32_t> > index_by_len_;
    uint32_t num_features_;

    std::vector<uint32_t> num_runs_by_len_;
    std::vector<double> increment_by_len_;
    std::vector<uint32_t *> masks_;
    std::vector<std::vector<uint32_t *> > masks_by_node_;

public:
    NetworkRunSymFeatures(const std::shared_ptr<const Network> & network,
            const uint32_t & run_length);

    NetworkRunSymFeatures(const NetworkRunSymFeatures & other);

    ~NetworkRunSymFeatures() override;

    std::shared_ptr<Features<State> > clone() const override;

    std::vector<double> get_features(
            const State & state,
            const boost::dynamic_bitset<> & trt_bits) override;

    void update_features(
            const uint32_t & changed_node,
            const State & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const State & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) override;

    void update_features_async(
            const uint32_t & changed_node,
            const State & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const State & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) const override;

    uint32_t num_features() const override;
};


} // namespace stdmMf


#endif // NETWORK_RUN_SYM_FEATURES_HPP
