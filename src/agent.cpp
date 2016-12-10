#include "agent.hpp"

namespace stdmMf {

Agent::Agent(const std::shared_ptr<const Network> & network)
    : network_(network), num_nodes_(network->size()),
      num_trt_(this->num_trt()) {
}

uint32_t Agent::num_trt() {
    return std::max(1u, static_cast<uint32_t>(this->network_->size() * 0.1));
}


} // namespace stdmMf
