#include "noTrtAgent.hpp"

namespace stdmMf {


NoTrtAgent::NoTrtAgent(const std::shared_ptr<const Network> & network)
    : Agent(network) {
}

NoTrtAgent::NoTrtAgent(const NoTrtAgent & other)
    : Agent(other.network_->clone()) {
}

std::shared_ptr<Agent> NoTrtAgent::clone() const {
    return std::shared_ptr<Agent>(new NoTrtAgent(*this));
}

boost::dynamic_bitset<> NoTrtAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    return boost::dynamic_bitset<> (this->num_nodes_);
}

boost::dynamic_bitset<> NoTrtAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits) {
    return boost::dynamic_bitset<>(this->num_nodes_);
}

uint32_t NoTrtAgent::num_trt() const {
    return 0;
}



} // namespace stdmMf
