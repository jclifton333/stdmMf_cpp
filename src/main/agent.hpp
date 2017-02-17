#ifndef AGENT_HPP
#define AGENT_HPP

#include "network.hpp"
#include "states.hpp"

namespace stdmMf {

template<typename State>
class Agent {
protected:
    const std::shared_ptr<const Network> network_;

    const uint32_t num_nodes_;

    const uint32_t num_trt_;

public:
    Agent(const std::shared_ptr<const Network> & network);

    Agent(const Agent<State> & other);

    virtual ~Agent() = default;

    virtual std::shared_ptr<Agent<State> > clone() const = 0;

    virtual boost::dynamic_bitset<> apply_trt(
            const State & inf_bits,
            const std::vector<StateAndTrt<State> > & history) = 0;

    virtual boost::dynamic_bitset<> apply_trt(
            const State & inf_bits);

    virtual uint32_t num_trt() const;
};


} // namespace stdmMf


#endif // AGENT_HPP
