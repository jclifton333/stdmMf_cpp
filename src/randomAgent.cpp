#include "randomAgent.hpp"


namespace stdmMf {


RandomAgent::RandomAgent(const std::shared_ptr<const Network> & network)
    : Agent(network) {
}

RandomAgent::RandomAgent(const RandomAgent & other)
    : Agent(other) {
}

std::shared_ptr<Agent> RandomAgent::clone() const {
    return std::shared_ptr<Agent>(new RandomAgent(*this));
}

boost::dynamic_bitset<> RandomAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    return this->apply_trt(inf_bits);
}

boost::dynamic_bitset<> RandomAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits) {
    const std::vector<int> ind_to_trt = this->rng->sample_range(0,
            this->num_nodes_, this->num_trt_);
    boost::dynamic_bitset<> trt_bits(this->num_nodes_);
    for (uint32_t i = 0; i < this->num_trt_; ++i) {
        trt_bits.set(ind_to_trt.at(i));
    }

    return trt_bits;
}

} // namespace stdmMf
