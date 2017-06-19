#include "allTrtAgent.hpp"

namespace stdmMf {


template<typename State>
AllTrtAgent<State>::AllTrtAgent(const std::shared_ptr<const Network> & network)
    : Agent<State>(network) {
}


template<typename State>
AllTrtAgent<State>::AllTrtAgent(const AllTrtAgent<State> & other)
    : Agent<State>(other.network_) {
}


template<typename State>
std::shared_ptr<Agent<State> > AllTrtAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(new AllTrtAgent<State>(*this));
}


template<typename State>
boost::dynamic_bitset<> AllTrtAgent<State>::apply_trt(
        const State & state,
        const std::vector<StateAndTrt<State> > & history) {
    // treat all
    return boost::dynamic_bitset<> (this->num_nodes_).set();
}


template<typename State>
boost::dynamic_bitset<> AllTrtAgent<State>::apply_trt(
        const State & state) {
    return boost::dynamic_bitset<>(this->num_nodes_).set();
}


template<typename State>
uint32_t AllTrtAgent<State>::num_trt() const {
    return this->network_->size();
}


template<typename State>
void AllTrtAgent<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
}



template class AllTrtAgent<InfState>;
template class AllTrtAgent<InfShieldState>;
template class AllTrtAgent<EbolaState>;

} // namespace stdmMf
