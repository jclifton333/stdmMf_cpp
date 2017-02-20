#ifndef SWEEP_AGENT_SLOW_HPP
#define SWEEP_AGENT_SLOW_HPP

#include <njm_cpp/tools/random.hpp>
#include "agent.hpp"
#include "features.hpp"


namespace stdmMf {


template <typename State>
class SweepAgentSlow : public Agent<State> {
protected:
    const std::shared_ptr<Features<State> > features_;
    const std::vector<double> coef_;

    const uint32_t max_sweeps_;

public:
    SweepAgentSlow(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features<State> > & features,
            const std::vector<double> & coef,
            const uint32_t & max_sweeps);

    SweepAgentSlow(const SweepAgentSlow<State> & other);

    ~SweepAgentSlow() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & state,
            const std::vector<StateAndTrt<State> > & history) override;

    boost::dynamic_bitset<> apply_trt(
            const State & state) override;

    void set_new_treatment(boost::dynamic_bitset<> & trt_bits,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const State & state) const;

    bool sweep_treatments(boost::dynamic_bitset<> & trt_bits,
            double & best_val,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const State & state) const;
};


} // namespace stdmMf


#endif // SWEEP_AGENT_SLOW_HPP
