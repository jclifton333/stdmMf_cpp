#include "proximalAgent.hpp"

namespace stdmMf {


ProximalAgent::ProximalAgent(const std::shared_ptr<const Network> & network)
    : network_(network), num_nodes_(network->size()), num_trt_(this->num_trt()){
}

boost::dynamic_bitset<> apply_trt(const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    std::vector<std::pair<double, uint32_t> > sorted;

    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        bool next_to_opp = false;
        const double draw = this->rng.runif_01();
    }
}

} // namespace stdmMf
