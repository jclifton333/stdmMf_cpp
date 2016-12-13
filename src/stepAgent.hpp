#ifndef STEP_AGENT_HPP
#define STEP_AGENT_HPP

#include "random.hpp"
#include "agent.hpp"
#include "features.hpp"


namespace stdmMf {


class StepAgent : public Agent, public RngClass {
protected:
    const std::shared_ptr<Features> features_;

    const uint32_t max_sweeps_;

public:
    StepAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features> & features,
            const uint32_t & max_sweeps,
            const std::vector<double> & coef);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history);
};


} // namespace stdmMf


#endif // STEP_AGENT_HPP
