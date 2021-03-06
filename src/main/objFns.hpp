#ifndef OBJ_FNS_HPP
#define OBJ_FNS_HPP

#include "agent.hpp"
#include "network.hpp"
#include "system.hpp"
#include <armadillo>

namespace stdmMf {


template<typename State>
double runner(System<State> * system, Agent<State> * agent,
        const uint32_t & final_time, const double gamma);



template<typename State>
double bellman_residual_sq(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);


template<typename State>
double bellman_residual_sq(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights);


template<typename State>
double sq_bellman_residual(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);


template<typename State>
double sq_bellman_residual(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights);


template <typename State>
std::vector<std::pair<double, double> > bellman_residual_parts(
        const std::vector<Transition<State>> & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);


// with gradients
template<typename State>
double bellman_residual_sq(
        const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);


template<typename State>
double bellman_residual_sq(
        const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad,
        const std::vector<double> & weights);


template<typename State>
double sq_bellman_residual(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);


template<typename State>
double sq_bellman_residual(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad,
        const std::vector<double> & weights);


template <typename State>
std::vector<std::pair<std::vector<double>, std::vector<double> > >
bellman_residual_parts(
        const std::vector<Transition<State>> & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);


template <typename State>
std::vector<std::pair<std::vector<double>, std::vector<double> > >
bellman_residual_parts(
        const std::vector<Transition<State>> & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad,
        const std::vector<double> & weights);


template <typename State>
arma::mat coef_variance_sqrt(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);




} // namespace stdmMf


#endif // OBJ_FNS_HPP
