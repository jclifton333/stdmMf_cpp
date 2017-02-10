#ifndef NO_TRT_AGENT_HPP
#define NO_TRT_AGENT_HPP

#include "agent.hpp"

namespace stdmMf {


class NoTrtAgent : public Agent {
public:
    NoTrtAgent(const std::shared_ptr<const Network> & network);

    NoTrtAgent(const NoTrtAgent & other);

    virtual std::shared_ptr<Agent> clone() const;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<InfAndTrt> & history);

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits);

    virtual uint32_t num_trt() const;
};


} // namespace stdmMf


#endif // NO_TRT_AGENT_HPP
