#ifndef SWEEP_AGENT_HPP
#define SWEEP_AGENT_HPP

#include "random.hpp"
#include "agent.hpp"
#include "features.hpp"


namespace stdmMf {


class SweepAgent : public Agent, public RngClass {
protected:
    const std::shared_ptr<Features> features_;
    const std::vector<double> coef_;

    const uint32_t max_sweeps_;

    const bool do_sweep_;

public:
    SweepAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features> & features,
            const std::vector<double> & coef,
            const uint32_t & max_sweeps,
            const bool & do_sweep);

    SweepAgent(const SweepAgent & other);

    virtual std::shared_ptr<Agent> clone() const;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits);

    void set_new_treatment(boost::dynamic_bitset<> & trt_bits,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const boost::dynamic_bitset<> & inf_bits,
            std::vector<double> & feat) const;

    bool sweep_treatments(boost::dynamic_bitset<> & trt_bits,
            double & best_val,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const boost::dynamic_bitset<> & inf_bits,
            std::vector<double> & feat) const;
};


} // namespace stdmMf


#endif // SWEEP_AGENT_SLOW_HPP
