#include "objFns.hpp"
#include <glog/logging.h>
#include <numeric>

#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

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
        value += - std::pow(gamma, i) * static_cast<double>(system->n_inf())
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
template double runner<EbolaState>(System<EbolaState> * system,
        Agent<EbolaState> * agent, const uint32_t & final_time,
        const double gamma);


template <typename State>
double bellman_residual_sq(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights) {

    const std::vector<std::pair<double, double> > parts =
        bellman_residual_parts(history, agent, gamma, q_fn, q_fn_next);

    const uint32_t n_points(history.size());
    CHECK_EQ(parts.size(), n_points);
    CHECK_EQ(weights.size(), n_points);

    double br_sq(0.0);
    for (uint32_t i = 0; i < n_points; ++i) {
        const double sum(parts.at(i).first + parts.at(i).second);
        br_sq += weights.at(i) * sum * sum;
    }

    return br_sq / n_points;
}


template <typename State>
double bellman_residual_sq(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next) {

    const std::vector<std::pair<double, double> > parts =
        bellman_residual_parts(history, agent, gamma, q_fn, q_fn_next);

    const uint32_t n_points(history.size());
    CHECK_EQ(parts.size(), n_points);

    double br_sq(0.0);
    for (uint32_t i = 0; i < n_points; ++i) {
        const double sum(parts.at(i).first + parts.at(i).second);
        br_sq += sum * sum;
    }

    return br_sq / n_points;
}




template double bellman_residual_sq<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);

template double bellman_residual_sq<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);

template double bellman_residual_sq<EbolaState>(
        const std::vector<Transition<EbolaState> > & history,
        Agent<EbolaState> * const agent, const double gamma,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);


template double bellman_residual_sq<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights);

template double bellman_residual_sq<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights);

template double bellman_residual_sq<EbolaState>(
        const std::vector<Transition<EbolaState> > & history,
        Agent<EbolaState> * const agent, const double gamma,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights);


template <typename State>
double sq_bellman_residual(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next) {

    const std::vector<std::pair<double, double> > parts =
        bellman_residual_parts(history, agent, gamma, q_fn, q_fn_next);

    const uint32_t n_points(history.size());
    CHECK_EQ(parts.size(), n_points);

    double tot_br(0.0);
    for (uint32_t i = 0; i < n_points; ++i) {
        const double sum(parts.at(i).first + parts.at(i).second);
        tot_br += sum;
    }

    const double mean_br = tot_br / n_points;
    return mean_br * mean_br;
}


template <typename State>
double sq_bellman_residual(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights) {

    const std::vector<std::pair<double, double> > parts =
        bellman_residual_parts(history, agent, gamma, q_fn, q_fn_next);

    const uint32_t n_points(history.size());
    CHECK_EQ(parts.size(), n_points);
    CHECK_EQ(weights.size(), n_points);

    double tot_br(0.0);
    for (uint32_t i = 0; i < n_points; ++i) {
        const double sum(parts.at(i).first + parts.at(i).second);
        tot_br += weights.at(i) * sum;
    }

    const double mean_br = tot_br / n_points;
    return mean_br * mean_br;
}


template double sq_bellman_residual<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);

template double sq_bellman_residual<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);

template double sq_bellman_residual<EbolaState>(
        const std::vector<Transition<EbolaState> > & history,
        Agent<EbolaState> * const agent, const double gamma,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);


template double sq_bellman_residual<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights);

template double sq_bellman_residual<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights);

template double sq_bellman_residual<EbolaState>(
        const std::vector<Transition<EbolaState> > & history,
        Agent<EbolaState> * const agent, const double gamma,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::vector<double> & weights);


template <typename State>
std::vector<std::pair<double, double> > bellman_residual_parts(
        const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next) {
    const uint32_t size = history.size();

    CHECK_GE(size, 1) << "need at least 1 transition";

    std::vector<std::pair<double, double> > parts;
    for (uint32_t i = 0; i < size; ++i) {
        const Transition<State> & transition = history.at(i);


        // R
        const uint32_t num_inf = transition.next_state.inf_bits.count();
        const uint32_t num_nodes = transition.next_state.inf_bits.size();
        const double r = - static_cast<double>(num_inf)
            / static_cast<double>(num_nodes);

        // Q(S, A)
        const double q_curr = q_fn(transition.curr_state,
                transition.curr_trt_bits);

        // Q(S', pi(S'))
        const boost::dynamic_bitset<> agent_trt =
            agent->apply_trt(transition.next_state);
        const double q_next = q_fn_next(transition.next_state, agent_trt);

        parts.push_back(std::pair<double,double>(r, gamma * q_next - q_curr));
    }
    return parts;
}


template
std::vector<std::pair<double, double> > bellman_residual_parts<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);

template
std::vector<std::pair<double, double> > bellman_residual_parts<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);

template
std::vector<std::pair<double, double> > bellman_residual_parts<EbolaState>(
        const std::vector<Transition<EbolaState> > & history,
        Agent<EbolaState> * const agent, const double gamma,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next);



// with gradient
template <typename State>
double sq_bellman_residual(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad) {

    const std::vector<std::pair<std::vector<double>, std::vector<double> > >
        parts =
        bellman_residual_parts(history, agent, gamma, q_fn, q_fn_next, grad);

    const uint32_t n_points(history.size());
    CHECK_EQ(parts.size(), n_points);

    CHECK_GT(parts.size(), 0);
    std::vector<double> tot_br(parts.at(0).first.size(), 0.0);
    for (uint32_t i = 0; i < n_points; ++i) {
        njm::linalg::add_b_to_a(tot_br,
                njm::linalg::add_a_and_b(parts.at(i).first,
                        parts.at(i).second));
    }

    const double mean_br = njm::linalg::l2_norm(tot_br) / n_points;
    return mean_br * mean_br;
}


template double sq_bellman_residual<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);

template double sq_bellman_residual<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);

template double sq_bellman_residual<EbolaState>(
        const std::vector<Transition<EbolaState> > & history,
        Agent<EbolaState> * const agent, const double gamma,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);


template <typename State>
double bellman_residual_sq(const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad) {

    const std::vector<std::pair<std::vector<double>, std::vector<double> > >
        parts =
        bellman_residual_parts(history, agent, gamma, q_fn, q_fn_next, grad);

    const uint32_t n_points(history.size());
    CHECK_EQ(parts.size(), n_points);

    double br_sq(0.0);
    for (uint32_t i = 0; i < n_points; ++i) {
        const double norm(njm::linalg::l2_norm(
                        njm::linalg::add_a_and_b(parts.at(i).first,
                                parts.at(i).second)));
        br_sq += norm;
    }

    return br_sq / n_points;
}

template double bellman_residual_sq<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);

template double bellman_residual_sq<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);

template double bellman_residual_sq<EbolaState>(
        const std::vector<Transition<EbolaState> > & history,
        Agent<EbolaState> * const agent, const double gamma,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);


template <typename State>
std::vector<std::pair<std::vector<double>, std::vector<double> > >
bellman_residual_parts(
        const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad) {
    const uint32_t size = history.size();

    CHECK_GE(size, 1) << "need at least 1 transition";

    std::vector<std::pair<std::vector<double>, std::vector<double> > > parts;
    for (uint32_t i = 0; i < size; ++i) {
        const Transition<State> & transition = history.at(i);

        // Gradient
        const std::vector<double> grad_vec(grad(transition.curr_state,
                        transition.curr_trt_bits));

        // R
        const uint32_t num_inf = transition.next_state.inf_bits.count();
        const uint32_t num_nodes = transition.next_state.inf_bits.size();
        const double r = - static_cast<double>(num_inf)
            / static_cast<double>(num_nodes);

        // Q(S, A)
        const double q_curr = q_fn(transition.curr_state,
                transition.curr_trt_bits);

        // Q(S', pi(S'))
        const boost::dynamic_bitset<> agent_trt =
            agent->apply_trt(transition.next_state);
        const double q_next = q_fn_next(transition.next_state, agent_trt);

        parts.push_back(std::pair<std::vector<double>,
                std::vector<double> >(njm::linalg::mult_a_and_b(grad_vec, r),
                        njm::linalg::mult_a_and_b(grad_vec,
                                gamma * q_next - q_curr)));
    }
    return parts;
}



template
std::vector<std::pair<std::vector<double>, std::vector<double> > >
bellman_residual_parts<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);

template
std::vector<std::pair<std::vector<double>, std::vector<double> > >
bellman_residual_parts<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);

template
std::vector<std::pair<std::vector<double>, std::vector<double> > >
bellman_residual_parts<EbolaState>(
        const std::vector<Transition<EbolaState> > & history,
        Agent<EbolaState> * const agent, const double gamma,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);



// sampling variance of coefficients
template <typename State>
arma::mat coef_variance_sqrt(
        const std::vector<Transition<State> > & history,
        Agent<State> * const agent, const double gamma,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const State & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad) {
    const uint32_t size = history.size();

    CHECK_GE(size, 1) << "need at least 1 transition";

    // containers
    std::vector<double> delta;
    delta.reserve(size);

    std::vector<arma::colvec> psi, psi_next;
    psi.reserve(size);
    psi_next.reserve(size);

    for (uint32_t i = 0; i < size; ++i) {
        const Transition<State> & transition = history.at(i);

        // R
        const uint32_t num_inf = transition.next_state.inf_bits.count();
        const uint32_t num_nodes = transition.next_state.inf_bits.size();
        const double r = - static_cast<double>(num_inf)
            / static_cast<double>(num_nodes);

        // Q(S, A)
        const double q_curr = q_fn(transition.curr_state,
                transition.curr_trt_bits);

        // current gradient
        const std::vector<double> grad_vec(grad(transition.curr_state,
                        transition.curr_trt_bits));
        psi.push_back(arma::colvec(grad_vec));

        // Q(S', pi(S'))
        const boost::dynamic_bitset<> agent_trt =
            agent->apply_trt(transition.next_state);
        const double q_next = q_fn_next(transition.next_state, agent_trt);

        // next gradient
        const std::vector<double> grad_vec_next(grad(transition.next_state,
                        agent_trt));
        psi_next.push_back(arma::colvec(grad_vec));

        delta.push_back(r + gamma * q_next - q_curr);
    }

    CHECK_EQ(delta.size(), size);
    CHECK_EQ(psi.size(), size);
    CHECK_EQ(psi_next.size(), size);

    const uint32_t dim(psi.at(0).size());

    arma::colvec delta_psi(dim, arma::fill::zeros);
    arma::mat psi_next_psi(dim, dim, arma::fill::zeros);
    arma::mat w(dim, dim, arma::fill::zeros);

    for (uint32_t i = 0; i < size; ++i) {
        delta_psi += delta.at(i) * psi.at(i);

        psi_next_psi += psi_next.at(i) * psi.at(i).t();

        w += psi.at(i) * psi.at(i).t();
    }

    const arma::mat sigma_mat(delta_psi * delta_psi.t());

    const arma::mat w_inv(arma::pinv(w));

    const arma::mat gamma_a(arma::mat(dim, dim, arma::fill::eye) -
            gamma * w_inv * psi_next_psi);

    const arma::mat gamma_b(
            w + gamma * gamma * psi_next_psi * w_inv * psi_next_psi.t()
            - 2 * gamma * psi_next_psi.t());
    const arma::mat gamma_b_inv(arma::pinv(gamma_b));

    // take sqrt of sigma
    arma::mat sigma_eigvec;
    arma::colvec sigma_eigval;
    arma::eig_sym(sigma_eigval, sigma_eigvec, sigma_mat);
    for (uint32_t i = 0; i < dim; ++i) {
        if (sigma_eigvec(i) > 0.01) {
            sigma_eigvec(i) = std::sqrt(sigma_eigvec(i));
        }
    }

    arma::mat gamma_mat(gamma_a.t() * gamma_b_inv);

    const uint32_t num_nodes(history.at(0).curr_state.inf_bits.size());

    return gamma_mat.t() * sigma_eigvec * arma::diagmat(sigma_eigval)
        / std::sqrt(static_cast<double>(num_nodes));
}

template
arma::mat coef_variance_sqrt<InfState>(
        const std::vector<Transition<InfState> > & history,
        Agent<InfState> * const agent, const double gamma,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const InfState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);

template
arma::mat coef_variance_sqrt<InfShieldState>(
        const std::vector<Transition<InfShieldState> > & history,
        Agent<InfShieldState> * const agent, const double gamma,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const InfShieldState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);

template
arma::mat coef_variance_sqrt<EbolaState>(
        const std::vector<Transition<EbolaState> > & history,
        Agent<EbolaState> * const agent, const double gamma,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn,
        const std::function<double(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & q_fn_next,
        const std::function<std::vector<double>(const EbolaState & state,
                const boost::dynamic_bitset<> & trt_bits)> & grad);

} // namespace stdmMf
