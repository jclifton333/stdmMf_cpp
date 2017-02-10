#include "objFns.hpp"
#include <glog/logging.h>

namespace stdmMf {


double runner(System * system, Agent * agent, const uint32_t & final_time,
        const double gamma) {
    double value = 0.0;
    for (uint32_t i = 0; i < final_time; ++i) {
        const boost::dynamic_bitset<> trt_bits = agent->apply_trt(
                system->inf_bits(), system->history());

        CHECK_EQ(trt_bits.count(), agent->num_trt());

        system->trt_bits(trt_bits);

        system->turn_clock();

        // negative of the number of infected nodes
        value += - gamma * static_cast<double>(system->inf_bits().count())
            / static_cast<double>(system->inf_bits().size());
    }
    return value;
}

double bellman_residual_sq(const std::vector<Transition> & history,
        Agent * const agent,
        const double gamma,
        const std::function<double(const boost::dynamic_bitset<> & inf_bits,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn) {
    const std::vector<std::pair<double, double> > parts =
        bellman_residual_parts(history, agent, gamma, q_fn);

    const double br_sq = std::accumulate(parts.begin(), parts.end(), 0.0,
            [](const double & x, const std::pair<double, double> & a) {
                return x + (a.first + a.second) * (a.first + a.second);
            });
    return br_sq / static_cast<double>(history.size() - 1);
}

std::vector<std::pair<double, double> > bellman_residual_parts(
        const std::vector<Transition> & history, Agent * const agent,
        const double gamma, const std::function<double(
                const boost::dynamic_bitset<> & inf_bits,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn) {
    const uint32_t size = history.size();

    CHECK_GE(size, 1) << "need at least 1 transition";

    std::vector<std::pair<double, double> > parts;
    for (uint32_t i = 0; i < size; ++i) {
        const Transition & transition = history.at(i);

        // R
        const double r = static_cast<double>(transition.next_inf_bits.count())
            / static_cast<double>(transition.next_inf_bits.size());

        // Q(S, A)
        const double q_curr = q_fn(transition.curr_inf_bits,
                transition.curr_trt_bits);

        // Q(S', pi(S'))
        const boost::dynamic_bitset<> agent_trt =
            agent->apply_trt(transition.next_inf_bits);
        const double q_next = q_fn(transition.next_inf_bits, agent_trt);

        // R + gamma * Q(S', pi(S') - Q(S, A)
        const double br = r + gamma * q_next - q_curr;

        parts.push_back(std::pair<double,double>(r, gamma * q_next - q_curr));
    }
    return parts;
}


} // namespace stdmMf
