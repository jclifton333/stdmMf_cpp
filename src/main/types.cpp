#include "types.hpp"

#include <glog/logging.h>

namespace stdmMf {

State::State(const uint32_t & num_nodes)
    : inf_bits(num_nodes), shield(num_nodes, 0.) {
}

State::State(const boost::dynamic_bitset<> & inf_bits,
        const std::vector<double> & shield)
    : inf_bits(inf_bits), shield(shield) {
}


StateAndTrt::StateAndTrt(const State & state,
        const boost::dynamic_bitset<> & trt_bits)
    : state(state), trt_bits(trt_bits) {
}

StateAndTrt::StateAndTrt(const boost::dynamic_bitset<> & inf_bits,
        const std::vector<double> & shield,
        const boost::dynamic_bitset<> & trt_bits)
    : state(inf_bits, shield), trt_bits(trt_bits) {
}

Transition::Transition(const State & curr_state,
        const boost::dynamic_bitset<> & curr_trt_bits,
        const State & next_state)
    : curr_state(curr_state), curr_trt_bits(curr_trt_bits),
      next_state(next_state) {
}

std::vector<Transition> Transition::from_sequence(
        const std::vector<StateAndTrt> & sequence) {
    CHECK_GT(sequence.size(), 1);
    const uint32_t num_transitions = sequence.size() - 1;
    std::vector<Transition> transitions;
    for (uint32_t i = 0; i < num_transitions; ++i) {
        transitions.emplace_back(sequence.at(i).state,
                sequence.at(i).trt_bits, sequence.at(i+1).state);
    }
    return transitions;
}


std::vector<Transition> Transition::from_sequence(
        const std::vector<StateAndTrt> & sequence,
        const State & final_state) {
    CHECK_GE(sequence.size(), 1);
    std::vector<Transition> transitions;
    if (sequence.size() > 1) {
        transitions = Transition::from_sequence(sequence);
    }
    const uint32_t sequence_size = sequence.size();
    transitions.emplace_back(sequence.at(sequence_size - 1).state,
            sequence.at(sequence_size - 1).trt_bits, final_state);
    return transitions;
}




} // namespace stdmMf
