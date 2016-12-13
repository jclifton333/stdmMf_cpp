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

    const std::vector<NetworkRun> runs_;
    const std::vector<std::vector<NetworkRun> > runs_by_node_;
    const uint32_t num_nodes_;
    const uint32_t run_length_;
    const uint32_t num_runs_;
    std::vector<uint32_t> offset_;
    uint32_t num_features_;

public:
    NetworkRunFeatures(const std::shared_ptr<const Network> & network,
            const uint32_t & run_length);

    virtual std::vector<double> get_features(
            const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits);

    virtual void update_features(
            const uint32_t & changed_node,
            const boost::dynamic_bitset<> & inf_bits_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const boost::dynamic_bitset<> & inf_bits_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat);

    virtual uint32_t num_features() const;
};


} // namespace stdmMf


#endif // NETWORK_RUN_FEATURES_HPP
