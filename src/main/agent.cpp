#include "agent.hpp"

namespace stdmMf {

template <typename State>
Agent::Agent(const std::shared_ptr<const Network> & network)
    : network_(network), num_nodes_(network->size()),
      num_trt_(this->num_trt()) {
}


template <typename State>
Agent::Agent(const Agent<State> & other)
    : network_(other.network_->clone()),
      num_nodes_(other.num_nodes_), num_trt_(other.num_trt_) {
}


template <typename State>
uint32_t Agent::num_trt() const {
    return std::max(1u, static_cast<uint32_t>(this->network_->size() * 0.05));
}


template <typename State>
boost::dynamic_bitset<> bAgent::apply_trt(
        const State & curr_state) {
    LOG(FATAL) << "Needs history to apply treatment.";
    return boost::dynamic_bitset<> ();
}


template class Agent<InfState>;
template class Agent<InfShieldState>;


} // namespace stdmMf
