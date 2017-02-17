#ifndef RANDOM_AGENT_HPP
#define RANDOM_AGENT_HPP

#include <njm_cpp/tools/random.hpp>
#include "states.hpp"
#include "network.hpp"
#include "agent.hpp"

namespace stdmMf {


template <typename State>
class RandomAgent : public Agent<State>, public njm::tools::RngClass {
public:
    RandomAgent(const std::shared_ptr<const Network> & network);

    RandomAgent(const RandomAgent<State> & agent);

    ~RandomAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & state,
            const std::vector<StateAndTrt<State> > & history) override;

    boost::dynamic_bitset<> apply_trt(
            const State & state) override;
};


} // namespace stdmMf


#endif // RANDOM_AGENT_HPP
