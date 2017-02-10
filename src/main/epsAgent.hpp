#ifndef EPS_AGENT_HPP
#define EPS_AGENT_HPP

#include "agent.hpp"
#include "random.hpp"
#include "network.hpp"

namespace stdmMf {


class EpsAgent : public Agent, public RngClass {
    const std::shared_ptr<Agent> agent_;
    const std::shared_ptr<Agent> eps_agent_;
    const double eps_;

public:
    EpsAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Agent> & agent,
            const std::shared_ptr<Agent> & eps_agent,
            const double & eps);

    EpsAgent(const EpsAgent & eps_agent) ;

    virtual std::shared_ptr<Agent> clone() const;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<InfAndTrt> & history);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits);

    virtual uint32_t num_trt() const;

    uint32_t num_trt_eps() const;
};


} // namespace stdmMf


#endif // EPS_AGENT_HPP
