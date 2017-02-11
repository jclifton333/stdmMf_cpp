#include "epsAgent.hpp"

namespace stdmMf {


EpsAgent::EpsAgent(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Agent> & agent,
        const std::shared_ptr<Agent> & eps_agent,
        const double & eps)
    : Agent(network), agent_(agent), eps_agent_(eps_agent), eps_(eps) {
}

EpsAgent::EpsAgent(const EpsAgent & other)
    : Agent(other), RngClass(other), agent_(other.agent_->clone()),
      eps_agent_(other.eps_agent_->clone()), eps_(other.eps_) {
}

std::shared_ptr<Agent> EpsAgent::clone() const {
    return std::shared_ptr<Agent>(new EpsAgent(*this));
}

boost::dynamic_bitset<> EpsAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits) {
    if (this->rng_->runif_01() < this->eps_) {
        return this->eps_agent_->apply_trt(inf_bits);
    } else {
        return this->agent_->apply_trt(inf_bits);
    }
}

boost::dynamic_bitset<> EpsAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<InfAndTrt> & history) {
    if (this->rng_->runif_01() < this->eps_) {
        return this->eps_agent_->apply_trt(inf_bits, history);
    } else {
        return this->agent_->apply_trt(inf_bits, history);
    }
}

uint32_t EpsAgent::num_trt() const {
    return this->agent_->num_trt();
}

uint32_t EpsAgent::num_trt_eps() const {
    return this->eps_agent_->num_trt();
}


} // namespace stdmMf
