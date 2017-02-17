#include "objFns.hpp"
#include <glog/logging.h>

namespace stdmMf {


template <typename State>
double runner(System<State> * system, Agent<State> * agent,
        const uint32_t & final_time, const double gamma) {
    double value = 0.0;
    for (uint32_t i = 0; i < final_time; ++i) {
        const boost::dynamic_bitset<> trt_bits = agent->apply_trt(
                system->state(), system->history());

        CHECK_EQ(trt_bits.count(), agent->num_trt());

        system->trt_bits(trt_bits);

        system->turn_clock();

        // negative of the number of infected nodes
        value += - gamma * static_cast<double>(system->n_inf())
            / static_cast<double>(system->num_nodes());
    }
    return value;
}


template double runner<InfState>(System<InfState> * system,
        Agent<InfState> * agent, const uint32_t & final_time,
        const double gamma);
template double runner<InfShieldState>(System<InfShieldState> * system,
        Agent<InfShieldState> * agent, const uint32_t & final_time,
        const double gamma);


template <typename State>
double bellman_residual_sq(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn) {
    const std::vector<std::pair<double, double> > parts =
        bellman_residual_parts(history, agent, gamma, q_fn);

    const double br_sq = std::accumulate(parts.begin(), parts.end(), 0.0,
            [](const double & x, const std::pair<double, double> & a) {
                return x + (a.first + a.second) * (a.first + a.second);
            });
    return br_sq / history.size();
}


template double bellman_residual_sq<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn);

template double bellman_residual_sq<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn);


template <typename State>
std::vector<std::pair<double, double> > bellman_residual_parts(
        const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn) {
    const uint32_t size = history.size();

    CHECK_GE(size, 1) << "need at least 1 transition";

    std::vector<std::pair<double, double> > parts;
    for (uint32_t i = 0; i < size; ++i) {
        const Transition<State> & transition = history.at(i);


        // R
        const uint32_t num_inf = transition.next_state.inf_bits.count();
        const uint32_t num_nodes = transition.next_state.inf_bits.size();
        const double r = static_cast<double>(num_inf)
            / static_cast<double>(num_nodes);

        // Q(S, A)
        const double q_curr = q_fn(transition.curr_state,
                transition.curr_trt_bits);

        // Q(S', pi(S'))
        const boost::dynamic_bitset<> agent_trt =
            agent->apply_trt(transition.next_state);
        const double q_next = q_fn(transition.next_state, agent_trt);

        // R + gamma * Q(S', pi(S') - Q(S, A)
        const double br = r + gamma * q_next - q_curr;

        parts.push_back(std::pair<double,double>(r, gamma * q_next - q_curr));
    }
    return parts;
}


template
std::vector<std::pair<double, double> > bellman_residual_parts<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn);

template
std::vector<std::pair<double, double> > bellman_residual_parts<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn);


} // namespace stdmMf
