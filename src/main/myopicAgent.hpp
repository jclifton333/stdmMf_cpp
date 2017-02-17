#ifndef MYOPIC_AGENT_HPP
#define MYOPIC_AGENT_HPP

#include "agent.hpp"
#include <njm_cpp/tools/random.hpp>
#include "model.hpp"

namespace stdmMf {


template <typename State>
class MyopicAgent : public Agent<State>, public njm::tools::RngClass {
protected:
    const std::shared_ptr<Model<State> > model_;

public:
    MyopicAgent(const std::shared_ptr<const Network> & network,
            const std::shared_ptr<Model<State> > & model);

    MyopicAgent(const MyopicAgent<State> & other);

    ~MyopicAgent() override = default;

    std::shared_ptr<Agent<State> > clone() const override;

    boost::dynamic_bitset<> apply_trt(
            const State & state,
            const std::vector<StateAndTrt<State> > & history) override;

    boost::dynamic_bitset<> apply_trt(
            const State & state) override;

};


} // namespace stdmMf


#endif // MYOPIC_AGENT_HPP
