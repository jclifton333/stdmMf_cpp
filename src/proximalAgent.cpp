#include "proximalAgent.hpp"

namespace stdmMf {


ProximalAgent::ProximalAgent(const std::shared_ptr<const Network> & network)
    : Agent(network) {
}

ProximalAgent::ProximalAgent(const ProximalAgent & other)
    : Agent(other) {
}

std::shared_ptr<Agent> ProximalAgent::clone() const {
    return std::shared_ptr<Agent>(new ProximalAgent(*this));
}

boost::dynamic_bitset<> ProximalAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    std::vector<std::pair<double, uint32_t> > sorted;

    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        const bool inf_i = inf_bits.test(i);
        bool next_to_opp = false;
        const double draw = this->rng->runif_01();

        const Node & node = this->network_->get_node(i);
        const uint32_t num_neigh = node.neigh_size();
        for (uint32_t j = 0; j < num_neigh; ++j) {
            if (inf_i != inf_bits.test(node.neigh(j))) {
                next_to_opp = true;
                break;
            }
        }

        // negative since sorting is ascending order
        sorted.push_back(std::pair<double, uint32_t>(- draw - next_to_opp, i));
    }

    boost::dynamic_bitset<> trt_bits(this->num_nodes_);
    // sort in ascending order
    std::sort(sorted.begin(), sorted.end());
    for (uint32_t i = 0; i < this->num_trt_; ++i) {
        trt_bits.set(sorted.at(i).second);
    }

    return trt_bits;
}

} // namespace stdmMf
