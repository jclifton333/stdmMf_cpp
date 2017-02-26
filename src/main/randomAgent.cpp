#include "randomAgent.hpp"


namespace stdmMf {


template<typename State>
RandomAgent<State>::RandomAgent(const std::shared_ptr<const Network> & network)
    : Agent<State>(network) {
}


template<typename State>
RandomAgent<State>::RandomAgent(const RandomAgent & other)
    : Agent<State>(other) {
}


template<typename State>
std::shared_ptr<Agent<State> > RandomAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(new RandomAgent<State>(*this));
}


template<typename State>
boost::dynamic_bitset<> RandomAgent<State>::apply_trt(
        const State & state,
        const std::vector<StateAndTrt<State> > & history) {
    return this->apply_trt(state);
}


template<typename State>
boost::dynamic_bitset<> RandomAgent<State>::apply_trt(
        const State & state) {
    const std::vector<int> ind_to_trt = this->rng_->sample_range(0,
            this->num_nodes_, this->num_trt_);
    boost::dynamic_bitset<> trt_bits(this->num_nodes_);
    for (uint32_t i = 0; i < this->num_trt_; ++i) {
        trt_bits.set(ind_to_trt.at(i));
    }

    return trt_bits;
}

template<typename State>
void RandomAgent<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
}


template class RandomAgent<InfState>;
template class RandomAgent<InfShieldState>;

} // namespace stdmMf
