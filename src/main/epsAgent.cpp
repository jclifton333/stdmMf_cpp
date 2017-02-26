#include "epsAgent.hpp"
#include "states.hpp"

namespace stdmMf {


template <typename State>
EpsAgent<State>::EpsAgent(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Agent<State> > & agent,
        const std::shared_ptr<Agent<State> > & eps_agent,
        const double & eps)
    : Agent<State>(network), agent_(agent), eps_agent_(eps_agent),
    eps_(eps) {
    // share rng
    this->agent_->rng(this->rng());
    this->eps_agent_->rng(this->rng());
}


template <typename State>
EpsAgent<State>::EpsAgent(const EpsAgent & other)
    : Agent<State>(other), agent_(other.agent_->clone()),
      eps_agent_(other.eps_agent_->clone()), eps_(other.eps_) {
    // share rng
    this->agent_->rng(this->rng());
    this->eps_agent_->rng(this->rng());
}


template <typename State>
std::shared_ptr<Agent<State> > EpsAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(new EpsAgent<State>(*this));
}


template <typename State>
boost::dynamic_bitset<> EpsAgent<State>::apply_trt(
        const State & curr_state) {
    if (this->rng_->runif_01() < this->eps_) {
        return this->eps_agent_->apply_trt(curr_state);
    } else {
        return this->agent_->apply_trt(curr_state);
    }
}


template <typename State>
boost::dynamic_bitset<> EpsAgent<State>::apply_trt(
        const State & curr_state,
        const std::vector<StateAndTrt<State> > & history) {
    if (this->rng_->runif_01() < this->eps_) {
        return this->eps_agent_->apply_trt(curr_state, history);
    } else {
        return this->agent_->apply_trt(curr_state, history);
    }
}


template <typename State>
uint32_t EpsAgent<State>::num_trt() const {
    return this->agent_->num_trt();
}


template <typename State>
uint32_t EpsAgent<State>::num_trt_eps() const {
    return this->eps_agent_->num_trt();
}

template<typename State>
void EpsAgent<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
    this->eps_agent_->rng(rng);
    this->agent_->rng(rng);
}



template class EpsAgent<InfState>;
template class EpsAgent<InfShieldState>;


} // namespace stdmMf
