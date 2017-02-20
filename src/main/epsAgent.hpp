#ifndef EPS_AGENT_HPP
#define EPS_AGENT_HPP

#include "agent.hpp"
#include "network.hpp"

namespace stdmMf {


template <typename State>
class EpsAgent : public Agent<State> {
    const std::shared_ptr<Agent<State> > agent_;
    const std::shared_ptr<Agent<State> > eps_agent_;
    const double eps_;

public:
    EpsAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Agent<State> > & agent,
            const std::shared_ptr<Agent<State> > & eps_agent,
            const double & eps);

    EpsAgent(const EpsAgent<State> & eps_agent);

    ~EpsAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & curr_state,
            const std::vector<StateAndTrt<State> > & history) override;

    boost::dynamic_bitset<> apply_trt(
            const State & curr_state) override;

    uint32_t num_trt() const override;

    uint32_t num_trt_eps() const;
};


} // namespace stdmMf


#endif // EPS_AGENT_HPP
