#ifndef RANDOM_AGENT_HPP
#define RANDOM_AGENT_HPP

#include "types.hpp"
#include "random.hpp"
#include "network.hpp"
#include "agent.hpp"

namespace stdmMf {


class RandomAgent : public Agent, public RngClass {
public:
    RandomAgent(const std::shared_ptr<const Network> & network);

    RandomAgent(const RandomAgent & agent);

    virtual std::shared_ptr<Agent> clone() const;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits);
};


} // namespace stdmMf


#endif // RANDOM_AGENT_HPP
