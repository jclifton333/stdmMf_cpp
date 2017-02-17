#ifndef NO_TRT_AGENT_HPP
#define NO_TRT_AGENT_HPP

#include "agent.hpp"

namespace stdmMf {


template <typename State>
class NoTrtAgent : public Agent<State> {
public:
    NoTrtAgent(const std::shared_ptr<const Network> & network);

    NoTrtAgent(const NoTrtAgent<State> & other);

    ~NoTrtAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & state,
            const std::vector<StateAndTrt<State> > & history) override;

    boost::dynamic_bitset<> apply_trt(
            const State & state) override;

    uint32_t num_trt() const override;
};


} // namespace stdmMf


#endif // NO_TRT_AGENT_HPP
