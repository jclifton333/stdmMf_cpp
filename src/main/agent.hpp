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

    Agent(const Agent & other);

    virtual std::shared_ptr<Agent> clone() const = 0;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<InfAndTrt> & history) = 0;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits) = 0;

    virtual uint32_t num_trt() const;
};


} // namespace stdmMf


#endif // AGENT_HPP
