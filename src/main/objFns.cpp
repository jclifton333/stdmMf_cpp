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

double bellman_residual_sq(const std::vector<InfAndTrt> & history,
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
        const std::vector<InfAndTrt> & history, Agent * const agent,
        const double gamma, const std::function<double(
                const boost::dynamic_bitset<> & inf_bits,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn) {
    const uint32_t size = history.size();

    CHECK_GT(size, 1) << "need at least 2 points";

    std::vector<std::pair<double, double> > parts;
    for (uint32_t i = 0; i < (size - 1); ++i) {
        const InfAndTrt & bp_curr = history.at(i);
        const InfAndTrt & bp_next = history.at(i+1);

        const double r = static_cast<double>(bp_next.inf_bits.count())
            / static_cast<double>(bp_next.inf_bits.size());

        const boost::dynamic_bitset<> agent_trt =
            agent->apply_trt(bp_curr.inf_bits);

        const double q_curr = q_fn(bp_curr.inf_bits, bp_curr.trt_bits);
        const double q_next = q_fn(bp_next.inf_bits, agent_trt);


        const double br = r + gamma * q_next - q_curr;
        parts.push_back(std::pair<double,double>(r, gamma * q_next - q_curr));
    }
    return parts;
}


} // namespace stdmMf
