#ifndef PROXIMAL_AGENT_HPP
#define PROXIMAL_AGENT_HPP

#include "agent.hpp"
#include "network.hpp"
#include <njm_cpp/tools/random.hpp>

namespace stdmMf {


template <typename State>
class ProximalAgent : public Agent<State> {

public:
    ProximalAgent(const std::shared_ptr<const Network> & network);

    ProximalAgent(const ProximalAgent<State> & other);

    ~ProximalAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & state,
            const std::vector<StateAndTrt<State> > & history) override;

    boost::dynamic_bitset<> apply_trt(
            const State & state) override;

    void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};


} // namespace stdmMf


#endif // PROXIMAL_AGENT_HPP
