#include "agent.hpp"

namespace stdmMf {


uint32_t Agent::num_trt() {
    return std::max(1u, static_cast<uint32_t>(this->network_->size() * 0.1));
}


} // namespace stdmMf
