#ifndef PROXIMAL_AGENT_HPP
#define PROXIMAL_AGENT_HPP

#include "agent.hpp"
#include "network.hpp"
#include "random.hpp"

namespace stdmMf {


class ProximalAgent : public Agent, public RngClass {

public:
    ProximalAgent(const std::shared_ptr<const Network> & network);

    ProximalAgent(const ProximalAgent & other);

    virtual std::shared_ptr<Agent> clone() const;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<InfAndTrt> & history);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits);
};


} // namespace stdmMf


#endif // PROXIMAL_AGENT_HPP
