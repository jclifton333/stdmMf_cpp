#ifndef FEATURES_HPP
#define FEATURES_HPP

#include <boost/dynamic_bitset.hpp>
#include <vector>
#include <memory>

namespace stdmMf {


template <typename State>
class Features {
public:
    virtual ~Features() = default;

    virtual std::shared_ptr<Features<State> > clone() const = 0;

    virtual std::vector<double> get_features(
            const State & state,
            const boost::dynamic_bitset<> & trt_bits) = 0;

    virtual void update_features(
            const uint32_t & changed_node,
            const State & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const State & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) = 0;

    virtual void update_features_async(
            const uint32_t & changed_node,
            const State & state_new,
            const boost::dynamic_bitset<> & trt_bits_new,
            const State & state_old,
            const boost::dynamic_bitset<> & trt_bits_old,
            std::vector<double> & feat) const = 0;

    virtual uint32_t num_features() const = 0;
};


} // namespace stdmMf


#endif // FEATURES_HPP
