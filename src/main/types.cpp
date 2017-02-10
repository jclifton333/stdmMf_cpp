#include "types.hpp"

#include <glog/logging.h>

namespace stdmMf {


InfAndTrt::InfAndTrt(const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits)
    : inf_bits(inf_bits), trt_bits(trt_bits) {
}

Transition::Transition(const boost::dynamic_bitset<> & curr_inf_bits,
        const boost::dynamic_bitset<> & curr_trt_bits,
        const boost::dynamic_bitset<> & next_inf_bits)
    : curr_inf_bits(curr_inf_bits), curr_trt_bits(curr_trt_bits),
      next_inf_bits(next_inf_bits) {
}

std::vector<Transition> Transition::from_sequence(
        const std::vector<InfAndTrt> & sequence) {
    CHECK_GT(sequence.size(), 1);
    const uint32_t num_transitions = sequence.size() - 1;
    std::vector<Transition> transitions;
    for (uint32_t i = 0; i < num_transitions; ++i) {
        transitions.emplace_back(sequence.at(i).inf_bits,
                sequence.at(i).trt_bits, sequence.at(i+1).inf_bits);
    }
    return transitions;
}


std::vector<Transition> Transition::from_sequence(
        const std::vector<InfAndTrt> & sequence,
        const boost::dynamic_bitset<> & final_inf) {
    CHECK_GT(sequence.size(), 1);
    std::vector<Transition> transitions(Transition::from_sequence(sequence));
    const uint32_t sequence_size = sequence.size();
    transitions.emplace_back(sequence.at(sequence_size - 1).inf_bits,
            sequence.at(sequence_size - 1).trt_bits, final_inf);
    return transitions;
}




} // namespace stdmMf
