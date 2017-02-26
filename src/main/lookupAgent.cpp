#include "lookupAgent.hpp"

namespace stdmMf {


LookupAgent::LookupAgent(
        const StateLookup<InfState, boost::dynamic_bitset<> > & lookup,
        const std::shared_ptr<const Network> & network)
    : Agent(network), lookup_(lookup) {
}


LookupAgent::LookupAgent(const LookupAgent & other)
    : Agent(other), lookup_(other.lookup_) {
}


std::shared_ptr<Agent<InfState> > LookupAgent::clone() const {
    return std::shared_ptr<Agent<InfState> >(new LookupAgent(*this));
}


boost::dynamic_bitset<> LookupAgent::apply_trt(
        const InfState & state,
        const std::vector<StateAndTrt<InfState> > & history) {
    return this->lookup_.get(state.inf_bits);
}


boost::dynamic_bitset<> LookupAgent::apply_trt(
        const InfState & state) {
    return this->lookup_.get(state.inf_bits);
}


void LookupAgent::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
}



} // namespace stdmMf
