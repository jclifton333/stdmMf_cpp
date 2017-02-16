#ifndef TYPES_HPP
#define TYPES_HPP

#include <boost/dynamic_bitset.hpp>
#include <boost/serialization/strong_typedef.hpp>

namespace stdmMf {

struct State {
    // infection status
    boost::dynamic_bitset<> inf_bits;
    // disease can have a shield against treatment
    // if shield is large, then treatment has a decreased effect
    std::vector<double> shield;

    State(const uint32_t & num_nodes);

    State(const boost::dynamic_bitset<> & inf_bits,
            const std::vector<double> & shield);
};


struct StateAndTrt {
    State state;
    boost::dynamic_bitset<> trt_bits;

    StateAndTrt(const State & state,
            const boost::dynamic_bitset<> & trt_bits);

    StateAndTrt(const boost::dynamic_bitset<> & inf_bits,
            const std::vector<double> & shield,
            const boost::dynamic_bitset<> & trt_bits);
};

struct Transition {
    State curr_state;
    boost::dynamic_bitset<> curr_trt_bits;
    State next_state;

    Transition(const State & curr_state,
            const boost::dynamic_bitset<> & curr_trt_bits,
            const State & next_state);

    static std::vector<Transition> from_sequence(
            const std::vector<StateAndTrt> & sequence);
    static std::vector<Transition> from_sequence(
            const std::vector<StateAndTrt> & sequence,
            const State & final_state);
};


} // namespace stdmMf


#endif // TYPES_HPP
