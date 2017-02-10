#ifndef OBJ_FNS_HPP
#define OBJ_FNS_HPP

#include "agent.hpp"
#include "network.hpp"
#include "system.hpp"

namespace stdmMf {


double runner(System * system, Agent * agent, const uint32_t & final_time,
        const double gamma);

double bellman_residual_sq(const std::vector<Transition> & history,
        Agent * const agent,
        const double gamma,
        const std::function<double(const boost::dynamic_bitset<> & inf_bits,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn);

std::vector<std::pair<double, double> > bellman_residual_parts(
        const std::vector<Transition> & history, Agent * const agent,
        const double gamma, const std::function<double(
                const boost::dynamic_bitset<> & inf_bits,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn);



} // namespace stdmMf


#endif // OBJ_FNS_HPP
