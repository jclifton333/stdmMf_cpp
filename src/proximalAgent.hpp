#ifndef PROXIMAL_AGENT_HPP
#define PROXIMAL_AGENT_HPP

#include "agent.hpp"
#include "network.hpp"
#include "random.hpp"

namespace stdmMf {


class ProximalAgent : public Agent, public RngClass {

public:
    ProximalAgent(const std::shared_ptr<const Network> & network);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history);
};


} // namespace stdmMf


#endif // PROXIMAL_AGENT_HPP
