#ifndef VFN_MAX_SIM_PERTURB_AGENT_HPP
#define VFN_MAX_SIM_PERTURB_AGENT_HPP

#include <cstdint>
#include <njm_cpp/optim/simPerturb.hpp>
#include "model.hpp"
#include "agent.hpp"
#include "features.hpp"
#include "sweepAgent.hpp"

namespace stdmMf {


template <typename State>
class VfnMaxSimPerturbAgent : public Agent<State> {
protected:
    const std::shared_ptr<Features<State> > features_;
    const std::shared_ptr<Model<State> > model_;

    const uint32_t num_reps_;
    const uint32_t final_t_;
    const uint32_t proj_t_;
    const double c_;
    const double t_;
    const double a_;
    const double b_;
    const double ell_;
    const double min_step_size_;

    std::vector<double> last_optim_par_;

    std::vector<std::vector<double> > optim_par_history_;

public:
    VfnMaxSimPerturbAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features<State> > & features,
            const std::shared_ptr<Model<State> > & model,
            const uint32_t & num_reps,
            const uint32_t & final_t,
            const uint32_t & proj_t,
            const double & c,
            const double & t,
            const double & a,
            const double & b,
            const double & ell,
            const double & min_step_size);

    VfnMaxSimPerturbAgent(const VfnMaxSimPerturbAgent & other);

    ~VfnMaxSimPerturbAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & state,
            const std::vector<StateAndTrt<State> > & history) override;

    std::vector<double> train(
            const std::vector<Transition<State> > & history,
            const std::vector<double> & starting_vals);

    const std::vector<std::vector<double> > & history() const;

    using njm::tools::RngClass::rng;
    void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};


} // namespace stdmMf


#endif // VFN_MAX_SIM_PERTURB_AGENT_HPP
