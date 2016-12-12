#ifndef NETWORK_RUN_FEATURES_HPP
#define NETWORK_RUN_FEATURES_HPP

#include <vector>
#include <boost/dynamic_bitset.hpp>
#include "network.hpp"
#include "features.hpp"

namespace stdmMf {


class NetworkRunFeatures : public Features {
private:
    const std::shared_ptr<const Network> network_;

    const std::vector<NetworkRun> paths_;
    const std::vector<std::vector<NetworkRun> > paths_by_node_;
    const uint32_t num_nodes_;
    const uint32_t run_length_;

public:
    NetworkRunFeatures(const std::shared_ptr<const Network> & network,
            const uint32_t & run_length);

    virtual std::vector<double> get_features(
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits);

    virtual std::vector<double> update_features(
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits,
            const std::vector<double> & prev_feat);

    virtual uint32_t num_features();
};


} // namespace stdmMf


#endif // NETWORK_RUN_FEATURES_HPP
