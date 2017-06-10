#ifndef STATES_HPP
#define STATES_HPP

#include <boost/dynamic_bitset.hpp>

#include <njm_cpp/tools/random.hpp>

namespace stdmMf {

struct InfState {
    // infection status
    boost::dynamic_bitset<> inf_bits;
    // disease can have a shield against treatment
    // if shield is large, then treatment has a decreased effect

    InfState(const uint32_t & num_nodes);

    InfState(const InfState & other) = default;

    InfState(const boost::dynamic_bitset<> & inf_bits);

    void reset();

    static InfState random(const uint32_t & num_nodes,
            njm::tools::Rng & rng);
};


struct InfShieldState {
    // infection status
    boost::dynamic_bitset<> inf_bits;
    // disease can have a shield against treatment
    // if shield is large, then treatment has a decreased effect
    std::vector<double> shield;

    InfShieldState(const uint32_t & num_nodes);

    InfShieldState(const InfShieldState & other) = default;

    InfShieldState(const boost::dynamic_bitset<> & inf_bits,
            const std::vector<double> & shield);

    void reset();

    static InfShieldState random(const uint32_t & num_nodes,
            njm::tools::Rng & rng);
};


struct EbolaState {
    // infection status
    boost::dynamic_bitset<> inf_bits;

    const std::vector<double> & pop;

    EbolaState();

    EbolaState(const EbolaState & other) = default;

    EbolaState(const boost::dynamic_bitset<> & inf_bits);

    void reset();

    static EbolaState random(njm::tools::Rng & rng);
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


template <typename State>
bool inf_has_changed(const State & curr_state,
        const std::vector<StateAndTrt<State> > & sequence)  {
    return std::any_of(sequence.begin(), sequence.end(),
            [&curr_state] (const StateAndTrt<State> & state_trt) {
                return state_trt.state.inf_bits != curr_state.inf_bits;
            });
}


} // namespace stdmMf


#endif // STATES_HPP
