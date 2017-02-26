#include "proximalAgent.hpp"

namespace stdmMf {


template<typename State>
ProximalAgent<State>::ProximalAgent(
        const std::shared_ptr<const Network> & network)
    : Agent<State>(network) {
}


template<typename State>
ProximalAgent<State>::ProximalAgent(const ProximalAgent & other)
    : Agent<State>(other) {
}


template<typename State>
std::shared_ptr<Agent<State> > ProximalAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(new ProximalAgent<State>(*this));
}


template<typename State>
boost::dynamic_bitset<> ProximalAgent<State>::apply_trt(
        const State & state,
        const std::vector<StateAndTrt<State> > & history) {
    return this->apply_trt(state);
}


template<typename State>
boost::dynamic_bitset<> ProximalAgent<State>::apply_trt(
        const State & state) {
    std::vector<std::pair<double, uint32_t> > sorted;

    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        const bool inf_i = state.inf_bits.test(i);
        bool next_to_opp = false;
        const double draw = this->rng_->runif_01();

        const Node & node = this->network_->get_node(i);
        const uint32_t num_neigh = node.neigh_size();
        for (uint32_t j = 0; j < num_neigh; ++j) {
            if (inf_i != state.inf_bits.test(node.neigh(j))) {
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

template<typename State>
void ProximalAgent<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->RngClass::rng(rng);
}


template class ProximalAgent<InfState>;
template class ProximalAgent<InfShieldState>;



} // namespace stdmMf
