#ifndef RUNNER_HPP
#define RUNNER_HPP

#include "agent.hpp"
#include "network.hpp"
#include "system.hpp"

namespace stdmMf {


double runner(System & system, Agent * agent, const uint32_t & final_time);


} // namespace stdmMf


#endif // RUNNER_HPP
