#ifndef VFN_BR_ADAPT_SIM_PERTURB_AGENT_HPP
#define VFN_BR_ADAPT_SIM_PERTURB_AGENT_HPP

#include <cstdint>
#include "model.hpp"
#include "agent.hpp"
#include "features.hpp"
#include "sweepAgent.hpp"
#include "simPerturb.hpp"

namespace stdmMf {


class VfnBrAdaptSimPerturbAgent : public Agent, public RngClass {
protected:
    const std::shared_ptr<Features> features_;
    const std::shared_ptr<Model> model_;

    const uint32_t vfn_num_reps_;
    const uint32_t vfn_final_t_;
    const double vfn_c_;
    const double vfn_t_;
    const double vfn_a_;
    const double vfn_b_;
    const double vfn_ell_;
    const double vfn_min_step_size_;

    const double br_c_;
    const double br_t_;
    const double br_a_;
    const double br_b_;
    const double br_ell_;
    const double br_min_step_size_;

    const uint32_t step_cap_mult_;

public:
    VfnBrAdaptSimPerturbAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features> & features,
            const std::shared_ptr<Model> & model,
            const uint32_t & vfn_num_reps,
            const uint32_t & vfn_final_t,
            const double & vfn_c,
            const double & vfn_t,
            const double & vfn_a,
            const double & vfn_b,
            const double & vfn_ell,
            const double & vfn_min_step_size,
            const double & br_c,
            const double & br_t,
            const double & br_a,
            const double & br_b,
            const double & br_ell,
            const double & br_min_step_size,
            const uint32_t & step_cap_mult);

    VfnBrAdaptSimPerturbAgent(const VfnBrAdaptSimPerturbAgent & other);

    virtual std::shared_ptr<Agent> clone() const;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits);
};


} // namespace stdmMf


#endif // VFN_BR_ADAPT_SIM_PERTURB_AGENT_HPP
