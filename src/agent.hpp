#ifndef AGENT_HPP
#define AGENT_HPP

#include "network.hpp"
#include "types.hpp"

namespace stdmMf {


class Agent {
protected:
    const std::shared_ptr<const Network> network_;

    const uint32_t num_nodes_;

    const uint32_t num_trt_;

public:
    Agent(const std::shared_ptr<const Network> & network);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history) = 0;

    virtual uint32_t num_trt();
};


} // namespace stdmMf


#endif // AGENT_HPP
