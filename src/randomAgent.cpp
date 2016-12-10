#include "randomAgent.hpp"


namespace stdmMf {


RandomAgent::RandomAgent(const std::shared_ptr<const Network> & network)
    : Agent(network) {
}

boost::dynamic_bitset<> RandomAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    const std::vector<int> ind_to_trt = this->rng->sample_range(0,
            this->num_nodes_, this->num_trt_);
    boost::dynamic_bitset<> trt_bits;
    for (uint32_t i = 0; i < this->num_trt_; ++i) {
        trt_bits.set(ind_to_trt.at(i));
    }

    return trt_bits;
}

} // namespace stdmMf
