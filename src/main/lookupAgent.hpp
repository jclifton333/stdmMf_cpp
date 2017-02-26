#ifndef LOOKUP_AGENT_HPP
#define LOOKUP_AGENT_HPP


#include "agent.hpp"
#include "polValIteration.hpp"

namespace stdmMf {

class LookupAgent : public Agent<InfState> {
protected:
    const StateLookup<InfState, boost::dynamic_bitset<> > lookup_;

public:
    LookupAgent(const StateLookup<InfState, boost::dynamic_bitset<> > & lookup,
            const std::shared_ptr<const Network> & network);

    LookupAgent(const LookupAgent & other);

    ~LookupAgent() override = default;

    std::shared_ptr<Agent<InfState> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const InfState & state,
            const std::vector<StateAndTrt<InfState> > & history) override;

    boost::dynamic_bitset<> apply_trt(
            const InfState & state) override;

    using njm::tools::RngClass::rng;
    void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};



} // namespace stdmMf


#endif // LOOKUP_AGENT_HPP
