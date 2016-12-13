#include "epsAgent.hpp"

namespace stdmMf {


EpsAgent::EpsAgent(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Agent> & agent,
        const std::shared_ptr<Agent> & eps_agent,
        const double & eps)
    : agent_(agent), eps_agent_(eps_agent), eps_(eps), Agent(network) {
}

boost::dynamic_bitset<> EpsAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    if (this->rng->runif_01() < this->eps_) {
        return this->eps_agent_->apply_trt(inf_bits, history);
    } else {
        return this->agent_->apply_trt(inf_bits, history);
    }
}

uint32_t EpsAgent::num_trt() const {
    this->agent_->num_trt();
}

uint32_t EpsAgent::num_trt_eps() const {
    this->eps_agent_->num_trt();
}


} // namespace stdmMf
