#include "agent.hpp"

namespace stdmMf {

Agent::Agent(const std::shared_ptr<const Network> & network)
    : network_(network), num_nodes_(network->size()),
      num_trt_(this->num_trt()) {
}

Agent::Agent(const Agent & other)
    : network_(other.network_->clone()), num_nodes_(other.num_nodes_),
      num_trt_(other.num_trt_) {
}

uint32_t Agent::num_trt() const {
    return std::max(1u, static_cast<uint32_t>(this->network_->size() * 0.1));
}

} // namespace stdmMf
