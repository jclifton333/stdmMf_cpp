#ifndef OBJ_FNS_HPP
#define OBJ_FNS_HPP

#include "agent.hpp"
#include "network.hpp"
#include "system.hpp"

namespace stdmMf {


template<typename State>
double runner(System<State> * system, Agent<State> * agent,
        const uint32_t & final_time, const double gamma);


template<typename State>
double bellman_residual_sq(const std::vector<Transition<State> > & history,
        Agent<State> * const agent,
        const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn);


template <typename State>
std::vector<std::pair<double, double> > bellman_residual_parts(
        const std::vector<Transition<State>> & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn);



} // namespace stdmMf


#endif // OBJ_FNS_HPP
