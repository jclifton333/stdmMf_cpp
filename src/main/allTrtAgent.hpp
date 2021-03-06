#ifndef ALL_TRT_AGENT_HPP
#define ALL_TRT_AGENT_HPP

#include "agent.hpp"

namespace stdmMf {


template <typename State>
class AllTrtAgent : public Agent<State> {
public:
    AllTrtAgent(const std::shared_ptr<const Network> & network);

    AllTrtAgent(const AllTrtAgent<State> & other);

    ~AllTrtAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & state,
            const std::vector<StateAndTrt<State> > & history) override;

    boost::dynamic_bitset<> apply_trt(
            const State & state) override;

    uint32_t num_trt() const override;

    void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};


} // namespace stdmMf


#endif // NO_TRT_AGENT_HPP
