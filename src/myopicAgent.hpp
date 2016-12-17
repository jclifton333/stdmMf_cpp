#ifndef MYOPIC_AGENT_HPP
#define MYOPIC_AGENT_HPP

#include "agent.hpp"
#include "random.hpp"
#include "model.hpp"

namespace stdmMf {


class MyopicAgent : public Agent, public RngClass {
protected:
    const std::shared_ptr<Model> model_;

public:
    MyopicAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Model> & model);

    MyopicAgent(const MyopicAgent & other);

    virtual std::shared_ptr<Agent> clone() const;

    virtual boost::dynamic_bitset<> apply_trt(
            const boost::dynamic_bitset<> & inf_bits,
            const std::vector<BitsetPair> & history);

};


} // namespace stdmMf


#endif // MYOPIC_AGENT_HPP
