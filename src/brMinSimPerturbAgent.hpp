#ifndef BR_MIN_SIM_PERTURB_AGENT_HPP
#define BR_MIN_SIM_PERTURB_AGENT_HPP

#include "random.hpp"
#include "agent.hpp"
#include "features.hpp"
#include "model.hpp"

namespace stdmMf {


class BrMinSimPerturbAgent : public Agent, public RngClass {
    const std::shared_ptr<Features> features_;
    const std::shared_ptr<Model> model_;

    const uint32_t num_reps_;
    const uint32_t final_t_;
    const double c_;
    const double t_;
    const double a_;
    const double b_;
    const double ell_;
    const double min_step_size_;

public:
    BrMinSimPerturbAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features> & features,
            const std::shared_ptr<Model> & model,
            const uint32_t & num_reps,
            const uint32_t & final_t,
            const double & c,
            const double & t,
            const double & a,
            const double & b,
            const double & ell,
            const double & min_step_size);

    BrMinSimPerturbAgent(const BrMinSimPerturbAgent & other);

    virtual std::shared_ptr<Agent> clone() const;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits);

    virtual uint32_t num_trt() const;

};


} // namespace stdmMf


#endif // BR_MIN_SIM_PERTURB_AGENT_HPP
