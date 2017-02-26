#ifndef RANDOM_AGENT_HPP
#define RANDOM_AGENT_HPP

#include "states.hpp"
#include "network.hpp"
#include "agent.hpp"

namespace stdmMf {


template <typename State>
class RandomAgent : public Agent<State> {
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

    void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};


} // namespace stdmMf


#endif // RANDOM_AGENT_HPP
