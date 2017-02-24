#include "polValIteration.hpp"

#include <glog/logging.h>


namespace stdmMf {


template <typename ret_val_t>
ret_val_t StateLookup<InfState, ret_val_t>::get(
        const boost::dynamic_bitset<> & inf_bits) {
    return this->lookup_.at(inf_bits);
}

template <typename ret_val_t>
void StateLookup<InfState, ret_val_t>::put(
        const boost::dynamic_bitset<> & inf_bits,
        const ret_val_t & ret_val) {
    this->lookup_[inf_bits] = ret_val;
}


template class StateLookup<InfState, boost::dynamic_bitset<> >;
template class StateLookup<InfState, double >;


StateLookup<InfState, boost::dynamic_bitset<> > policyIteration(
        const std::shared_ptr<const Network> network,
        const std::shared_ptr<Model<InfState> > model) {
    StateLookup<InfState, double> value_lookup;
    StateLookup<InfState, boost::dynamic_bitset<> > policy_lookup;

    // initialize lookups
    CHECK_LT(network->size(), 10);
    const uint32_t max_inf_bits = (1 << network->size());
    for (uint32_t i = 0; i < max_inf_bits; ++i) {
        const boost::dynamic_bitset<> inf_bits(network->size(), i);
        lookup.put(inf_bits, 0.0);


    }

    // calculate expected reward for each location
    std::vector<double> exp_r;
    for (uint32_t i = 0; i < max_inf_bits; ++i) {
        model.probs(
    }



    return policy_lookup;
}



} // namespace stdmMf
