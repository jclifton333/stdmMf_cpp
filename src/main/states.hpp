#ifndef STATES_HPP
#define STATES_HPP

#include <boost/dynamic_bitset.hpp>

namespace stdmMf {

struct InfState {
    // infection status
    boost::dynamic_bitset<> inf_bits;
    // disease can have a shield against treatment
    // if shield is large, then treatment has a decreased effect

    InfState(const uint32_t & num_nodes);

    InfState(const InfState & other) = default;

    InfState(const boost::dynamic_bitset<> & inf_bits);
};


struct InfShieldState {
    // infection status
    boost::dynamic_bitset<> inf_bits;
    // disease can have a shield against treatment
    // if shield is large, then treatment has a decreased effect
    std::vector<double> shield;

    InfShieldState(const uint32_t & num_nodes);

    InfShieldState(const InfState & other) = default;

    InfShieldState(const boost::dynamic_bitset<> & inf_bits,
            const std::vector<double> & shield);
};


template <typename State>
struct StateAndTrt {
    State state;
    boost::dynamic_bitset<> trt_bits;

    StateAndTrt(const State & state,
            const boost::dynamic_bitset<> & trt_bits);
};

template <typename State>
struct Transition {
    State curr_state;
    boost::dynamic_bitset<> curr_trt_bits;
    State next_state;

    Transition(const State & curr_state,
            const boost::dynamic_bitset<> & curr_trt_bits,
            const State & next_state);

    // turn a sequence of states into a vector of transitions
    static std::vector<Transition<State> > from_sequence(
            const std::vector<StateAndTrt<State> > & sequence);

    // turn a sequence of states into a vector of transitions
    static std::vector<Transition<State> > from_sequence(
            const std::vector<StateAndTrt<State> > & sequence,
            const State & final_state);
};


} // namespace stdmMf


#endif // STATES_HPP
