#ifndef STEP_AGENT_HPP
#define STEP_AGENT_HPP

#include "random.hpp"
#include "agent.hpp"
#include "features.hpp"


namespace stdmMf {


class StepAgent : public Agent, public RngClass {
protected:
    const std::shared_ptr<Features> features_;
    const std::vector<double> coef_;

    const uint32_t max_sweeps_;

public:
    StepAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features> & features,
            const std::vector<double> & coef,
            const uint32_t & max_sweeps);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history);

    void set_new_treatment(boost::dynamic_bitset<> & trt_bits,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const boost::dynamic_bitset<> & inf_bits) const;

    bool sweep_treatments(boost::dynamic_bitset<> & trt_bits,
            double & best_val,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const boost::dynamic_bitset<> & inf_bits) const;
};


} // namespace stdmMf


#endif // STEP_AGENT_HPP
