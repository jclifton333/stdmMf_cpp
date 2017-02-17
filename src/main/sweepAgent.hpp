#ifndef SWEEP_AGENT_HPP
#define SWEEP_AGENT_HPP

#include <njm_cpp/tools/random.hpp>
#include <njm_cpp/thread/pool.hpp>
#include "agent.hpp"
#include "features.hpp"



namespace stdmMf {


template <typename State>
class SweepAgent : public Agent<State>, public njm::tools::RngClass {
protected:
    const std::shared_ptr<Features<State> > features_;
    const std::vector<double> coef_;

    const uint32_t max_sweeps_;

    const bool do_sweep_;

    bool do_parallel_;

    std::shared_ptr<njm::thread::Pool> pool_;

public:
    SweepAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Features<State> > & features,
            const std::vector<double> & coef,
            const uint32_t & max_sweeps,
            const bool & do_sweep);

    SweepAgent(const SweepAgent & other);

    ~SweepAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    void set_parallel(const bool & do_parallel,
            const uint32_t & num_threads);

    boost::dynamic_bitset<> apply_trt(
            const State & state,
            const std::vector<StateAndTrt<State> > & history) override;

    boost::dynamic_bitset<> apply_trt(
            const State & state) override;

    void set_new_treatment(boost::dynamic_bitset<> & trt_bits,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const State & state,
            std::vector<double> & feat) const;

    void set_new_treatment_serial(boost::dynamic_bitset<> & trt_bits,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const State & state,
            std::vector<double> & feat) const;

    void set_new_treatment_parallel(boost::dynamic_bitset<> & trt_bits,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const State & state,
            std::vector<double> & feat) const;

    bool sweep_treatments(boost::dynamic_bitset<> & trt_bits,
            double & best_val,
            std::set<uint32_t> & not_trt,
            std::set<uint32_t> & has_trt,
            const State & state,
            std::vector<double> & feat) const;
};


} // namespace stdmMf


#endif // SWEEP_AGENT_SLOW_HPP
