#include "noTrtAgent.hpp"

namespace stdmMf {


template<typename State>
NoTrtAgent<State>::NoTrtAgent(const std::shared_ptr<const Network> & network)
    : Agent<State>(network) {
}


template<typename State>
NoTrtAgent<State>::NoTrtAgent(const NoTrtAgent<State> & other)
    : Agent<State>(other.network_->clone()) {
}


template<typename State>
std::shared_ptr<Agent<State> > NoTrtAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(new NoTrtAgent<State>(*this));
}


template<typename State>
boost::dynamic_bitset<> NoTrtAgent<State>::apply_trt(
        const State & state,
        const std::vector<StateAndTrt<State> > & history) {
    return boost::dynamic_bitset<> (this->num_nodes_);
}


template<typename State>
boost::dynamic_bitset<> NoTrtAgent<State>::apply_trt(
        const State & state) {
    return boost::dynamic_bitset<>(this->num_nodes_);
}


template<typename State>
uint32_t NoTrtAgent<State>::num_trt() const {
    return 0;
}


template<typename State>
void NoTrtAgent<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->RngClass::rng(rng);
}



template class NoTrtAgent<InfState>;
template class NoTrtAgent<InfShieldState>;


} // namespace stdmMf
