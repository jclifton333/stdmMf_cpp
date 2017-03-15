#ifndef BR_MIN_ITER_SIM_PERTURB_AGENT_HPP
#define BR_MIN_ITER_SIM_PERTURB_AGENT_HPP

#include "agent.hpp"
#include "features.hpp"
#include "model.hpp"

namespace stdmMf {

template <typename State>
class BrMinIterSimPerturbAgent : public Agent<State> {
    const std::shared_ptr<Features<State> > features_;

    const double c_;
    const double t_;
    const double a_;
    const double b_;
    const double ell_;
    const double min_step_size_;
    const bool do_sweep_;
    const bool gs_step_;
    const bool sq_total_br_;
    const uint32_t obs_per_iter_;

    std::vector<double> last_optim_par_;

public:
    BrMinIterSimPerturbAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features<State> > & features,
            const double & c,
            const double & t,
            const double & a,
            const double & b,
            const double & ell,
            const double & min_step_size,
            const bool & do_sweep,
            const bool & gs_step,
            const bool & sq_total_br,
            const uint32_t & obs_per_iter);

    BrMinIterSimPerturbAgent(const BrMinIterSimPerturbAgent<State> & other);

    ~BrMinIterSimPerturbAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & curr_state,
            const std::vector<StateAndTrt<State> > & history) override;

    std::vector<double> train(
            const std::vector<Transition<State> > & history);

    std::vector<double> train(
            const std::vector<Transition<State> > & history,
            const std::vector<double> & starting_vals);

    using njm::tools::RngClass::rng;
    void rng(const std::shared_ptr<njm::tools::Rng> & rng) override;
};


} // namespace stdmMf


#endif // BR_MIN_ITER_SIM_PERTURB_AGENT_HPP
