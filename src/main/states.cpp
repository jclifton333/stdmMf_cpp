#include "states.hpp"

#include <glog/logging.h>

#include <njm_cpp/info/project.hpp>
#include <fstream>
#include <iterator>

namespace stdmMf {

// infection only state
InfState::InfState(const uint32_t & num_nodes)
    : inf_bits(num_nodes) {
}


InfState::InfState(const boost::dynamic_bitset<> & inf_bits)
    : inf_bits(inf_bits) {
}


void InfState::reset() {
    this->inf_bits.reset();
}


InfState InfState::random(const uint32_t & num_nodes,
        njm::tools::Rng & rng) {
    InfState state(num_nodes);
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (rng.runif_01() < 0.5) {
            state.inf_bits.flip();
        }
    }

    return state;
}


// infection and shield state
InfShieldState::InfShieldState(const uint32_t & num_nodes)
    : inf_bits(num_nodes), shield(num_nodes, 0.) {
}


InfShieldState::InfShieldState(const boost::dynamic_bitset<> & inf_bits,
        const std::vector<double> & shield)
    : inf_bits(inf_bits), shield(shield) {
}


void InfShieldState::reset() {
    this->inf_bits.reset();
    std::fill(this->shield.begin(), this->shield.end(), 0.);
}


InfShieldState InfShieldState::random(const uint32_t & num_nodes,
        njm::tools::Rng & rng) {
    InfShieldState state(num_nodes);
    for (uint32_t i = 0; i < num_nodes; ++i) {
        if (rng.runif_01() < 0.5) {
            state.inf_bits.flip();
        }

        state.shield.at(i) = rng.rnorm_01();
    }

    return state;
}


// Ebola state
const std::string ebola_root_dir(
        njm::info::project::PROJECT_ROOT_DIR + "/src/data/");
std::ifstream in(ebola_root_dir + "ebola_population.txt");
const std::vector<double> ebola_pop(std::istream_iterator<double>(in), {});


EbolaState::EbolaState()
    : inf_bits(290), pop(ebola_pop) {
}


EbolaState::EbolaState(const boost::dynamic_bitset<> & inf_bits)
    : inf_bits(inf_bits), pop(ebola_pop) {
}


void EbolaState::reset() {
    this->inf_bits.reset();
}


EbolaState EbolaState::random(njm::tools::Rng & rng) {
    EbolaState state;
    for (uint32_t i = 0; i < 290; ++i) {
        if (rng.runif_01() < 0.5) {
            state.inf_bits.flip(i);
        }
    }

    return state;
}



// state and treatment together
template <typename State>
StateAndTrt<State>::StateAndTrt(const State & state,
        const boost::dynamic_bitset<> & trt_bits)
    : state(state), trt_bits(trt_bits) {
}

// transitions
template <typename State>
Transition<State>::Transition(const State & curr_state,
        const boost::dynamic_bitset<> & curr_trt_bits,
        const State & next_state)
    : curr_state(curr_state), curr_trt_bits(curr_trt_bits),
      next_state(next_state) {
}

template struct StateAndTrt<InfState>;
template struct StateAndTrt<InfShieldState>;

template <typename State>
std::vector<Transition<State> > Transition<State>::from_sequence(
        const std::vector<StateAndTrt<State> > & sequence) {
    CHECK_GT(sequence.size(), 1);
    const uint32_t num_transitions = sequence.size() - 1;
    std::vector<Transition<State> > transitions;
    for (uint32_t i = 0; i < num_transitions; ++i) {
        transitions.emplace_back(sequence.at(i).state,
                sequence.at(i).trt_bits, sequence.at(i+1).state);
    }
    return transitions;
}

template <typename State>
std::vector<Transition<State> > Transition<State>::from_sequence(
        const std::vector<StateAndTrt<State> > & sequence,
        const State & final_state) {
    CHECK_GE(sequence.size(), 1);
    std::vector<Transition<State> > transitions;
    if (sequence.size() > 1) {
        transitions = Transition::from_sequence(sequence);
    }
    const uint32_t sequence_size = sequence.size();
    transitions.emplace_back(sequence.at(sequence_size - 1).state,
            sequence.at(sequence_size - 1).trt_bits, final_state);
    return transitions;
}

template struct Transition<InfState>;
template struct Transition<InfShieldState>;




} // namespace stdmMf
