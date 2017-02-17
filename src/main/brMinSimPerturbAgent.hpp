#ifndef BR_MIN_SIM_PERTURB_AGENT_HPP
#define BR_MIN_SIM_PERTURB_AGENT_HPP

#include <njm_cpp/tools/random.hpp>
#include "agent.hpp"
#include "features.hpp"
#include "model.hpp"

namespace stdmMf {

template <typename State>
class BrMinSimPerturbAgent : public Agent<State>, public njm::tools::RngClass {
    const std::shared_ptr<Features<State> > features_;

    const double c_;
    const double t_;
    const double a_;
    const double b_;
    const double ell_;
    const double min_step_size_;

public:
    BrMinSimPerturbAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features<State> > & features,
            const double & c,
            const double & t,
            const double & a,
            const double & b,
            const double & ell,
            const double & min_step_size);

    BrMinSimPerturbAgent(const BrMinSimPerturbAgent<State> & other);

    ~BrMinSimPerturbAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & curr_state,
            const std::vector<InfAndTrt> & history) override;
};


} // namespace stdmMf


#endif // BR_MIN_SIM_PERTURB_AGENT_HPP
