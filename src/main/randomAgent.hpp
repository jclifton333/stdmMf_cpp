#ifndef RANDOM_AGENT_HPP
#define RANDOM_AGENT_HPP

#include <njm_cpp/tools/random.hpp>
#include "types.hpp"
#include "network.hpp"
#include "agent.hpp"

namespace stdmMf {


class RandomAgent : public Agent, public njm::tools::RngClass {
public:
    RandomAgent(const std::shared_ptr<const Network> & network);

    RandomAgent(const RandomAgent & agent);

    virtual std::shared_ptr<Agent> clone() const;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<InfAndTrt> & history);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits);
};


} // namespace stdmMf


#endif // RANDOM_AGENT_HPP
