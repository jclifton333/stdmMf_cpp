#include "objFns.hpp"

namespace stdmMf {


double runner(System * system, Agent * agent, const uint32_t & final_time,
        const double gamma) {
    double value = 0.0;
    for (uint32_t i = 0; i < final_time; ++i) {
        const boost::dynamic_bitset<> trt_bits = agent->apply_trt(
                system->inf_bits(), system->history());

        system->trt_bits(trt_bits);

        system->turn_clock();

        value += gamma * static_cast<double>(system->inf_bits().count())
            / static_cast<double>(system->inf_bits().size());
    }
    return value;
}

double bellman_residual_sq(const std::vector<BitsetPair> & history,
        Agent * const agent,
        const double gamma,
        const std::function<double(const boost::dynamic_bitset<> & inf_bits,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn) {
    const uint32_t size = history.size();

    double avg_br_sq = 0.0;
    for (uint32_t i = 0; i < (size - 1); ++i) {
        const BitsetPair & bp_curr = history.at(i);
        const BitsetPair & bp_next = history.at(i+1);

        const double r = static_cast<double>(bp_next.first.count())
            / static_cast<double>(bp_next.first.size());

        const boost::dynamic_bitset<> agent_trt =
            agent->apply_trt(bp_curr.first);

        const double q_curr = q_fn(bp_curr.first, bp_curr.second);
        const double q_next = q_fn(bp_next.first, agent_trt);


        const double br = r + gamma * q_next - q_curr;

        avg_br_sq += br * br / static_cast<double>(size - 1);
    }
    return avg_br_sq;
}


} // namespace stdmMf
