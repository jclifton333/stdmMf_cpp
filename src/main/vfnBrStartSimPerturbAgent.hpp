#ifndef VFN_BR_START_SIM_PERTURB_AGENT_HPP
#define VFN_BR_START_SIM_PERTURB_AGENT_HPP

#include <cstdint>
#include <njm_cpp/tools/random.hpp>
#include "model.hpp"
#include "agent.hpp"
#include "features.hpp"
#include "sweepAgent.hpp"

namespace stdmMf {


template <typename State>
class VfnBrStartSimPerturbAgent : public Agent<State> {
protected:
    const std::shared_ptr<Features<State> > features_;
    const std::shared_ptr<Model<State> > model_;

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

public:
    VfnBrStartSimPerturbAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features<State> > & features,
            const std::shared_ptr<Model<State> > & model,
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
            const double & br_min_step_size);

    VfnBrStartSimPerturbAgent(const VfnBrStartSimPerturbAgent<State> & other);

    ~VfnBrStartSimPerturbAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & state,
            const std::vector<StateAndTrt<State> > & history) override;
};


} // namespace stdmMf


#endif // VFN_BR_START_SIM_PERTURB_AGENT_HPP
