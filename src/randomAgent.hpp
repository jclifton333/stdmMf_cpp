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

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history);
};


} // namespace stdmMf


#endif // RANDOM_AGENT_HPP
