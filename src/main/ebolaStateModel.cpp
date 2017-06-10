#include "ebolaStateModel.hpp"

#include <njm_cpp/tools/bitManip.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

namespace stdmMf {


EbolaStateModel::EbolaStateModel(const uint32_t & par_size,
        const std::shared_ptr<const Network> & network)
    : Model<EbolaState>(par_size, network) {
}


EbolaStateModel::EbolaStateModel(const EbolaStateModel & other)
    : Model<EbolaState>(other) {
}


std::vector<double> EbolaStateModel::probs(
        const EbolaState & state,
        const boost::dynamic_bitset<> & trt_status) const {
    const boost::dynamic_bitset<> & inf_status(state.inf_bits);

    std::vector<double> probs;
    const std::vector<uint32_t> status = njm::tools::combine_sets(
            state.inf_bits, trt_status);
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        const uint32_t status_i = status.at(i);
        const bool trt_i = status_i % 2 == 1;
        if (status_i < 2) {
            // not infected -> infection probability
            const Node & node = this->network_->get_node(i);
            const uint32_t num_neigh = node.neigh_size();

            double prob = 1.0;

            // factor in neighbors
            for (uint32_t j = 0; j < num_neigh; ++j) {
                const uint32_t neigh = node.neigh(j);
                const uint32_t status_neigh = status.at(neigh);
                if (status_neigh >= 2) {
                    // if neighbor is infected
                    const bool trt_neigh = status_neigh % 2 == 1;
                    prob *= 1.0 - this->a_inf_b(neigh, i, trt_neigh, trt_i,
                            inf_status, trt_status);
                }
            }
            probs.push_back(1.0 - prob);
        } else {
            // infected -> remains infected!
            probs.push_back(0.0);
        }
    }
    return probs;
}


double EbolaStateModel::ll(
        const std::vector<Transition<EbolaState> > & history) const {
    const uint32_t history_size = history.size();
    CHECK_GE(history_size, 1);
    double ll_value = 0.0;
    for (uint32_t i = 0; i < history_size; ++i) {
        // split up transition
        const Transition<EbolaState> & transition = history.at(i);
        const EbolaState & curr_state = transition.curr_state;
        const boost::dynamic_bitset<> & curr_trt =
            transition.curr_trt_bits;
        const EbolaState & next_state = transition.next_state;

        // pull out infection bits
        const boost::dynamic_bitset<> & curr_inf = curr_state.inf_bits;
        const boost::dynamic_bitset<> & next_inf = next_state.inf_bits;

        // infection probabilities
        const std::vector<double> probs = this->probs(curr_state, curr_trt);

        // get bits for changes in infection
        const boost::dynamic_bitset<> change_inf = curr_inf ^ next_inf;

        // convert bits to sets of indices
        const auto change_both_sets = njm::tools::both_sets(change_inf);
        const std::vector<uint32_t> changed = change_both_sets.first;
        const uint32_t num_changed = changed.size();
        const std::vector<uint32_t> unchanged = change_both_sets.second;
        const uint32_t num_unchanged = unchanged.size();

        for (uint32_t j = 0; j < num_changed; ++j) {
            const double p = probs.at(changed.at(j));
            ll_value += std::log(std::max(1e-14, p)); // for stability
        }
        for (uint32_t j = 0; j < num_unchanged; ++j) {
            if (!curr_inf.test(unchanged.at(j))) { // only uninfected can change
                const double p = 1.0 - probs.at(unchanged.at(j));
                ll_value += std::log(std::max(1e-14, p));
            }
        }
    }
    return ll_value / history_size;
}



std::vector<double> EbolaStateModel::ll_grad(
        const std::vector<Transition<EbolaState> > & history) const {
    const uint32_t history_size = history.size();
    CHECK_GT(history.size(), 0);

    std::vector<double> grad_value (this->par_size_, 0.0);
    for (uint32_t i = 0; i < history_size; ++i) {
        // split up transition
        const Transition<EbolaState> & transition = history.at(i);
        const EbolaState & curr_state = transition.curr_state;
        const boost::dynamic_bitset<> & curr_trt = transition.curr_trt_bits;
        const EbolaState & next_state = transition.next_state;


        // pull out inf bits
        const boost::dynamic_bitset<> & curr_inf = curr_state.inf_bits;
        const boost::dynamic_bitset<> & next_inf = next_state.inf_bits;

        // infection probabilities
        const std::vector<double> probs = this->probs(curr_state, curr_trt);

        // get bits for changes in infection
        const boost::dynamic_bitset<> change_inf = curr_inf ^ next_inf;

        // combine sets
        const std::vector<uint32_t> inf_and_trt =
            njm::tools::combine_sets(curr_inf, curr_trt);

        // convert bits to sets of indices
        const std::vector<uint32_t> inf_and_change =
            njm::tools::combine_sets(curr_inf, change_inf);

        for (uint32_t j = 0; j < this->num_nodes_; ++j) {
            const uint32_t status_j = inf_and_trt.at(j);
            const uint32_t change_j = inf_and_change.at(j);

            const bool trt_j = status_j % 2 == 1;
            const double prob_j = probs.at(j);

            if (status_j < 2) {
                // was uninfected
                std::vector<double> val_to_add(this->par_size_, 0.0);

                // neighbor effect
                const Node & node_j = this->network_->get_node(j);
                const uint32_t neigh_size = node_j.neigh_size();
                for (uint32_t k = 0; k < neigh_size; ++k) {
                    const uint32_t neigh = node_j.neigh(k);
                    if (inf_and_trt.at(neigh) >= 2) {
                        // if neighbor is infected
                        const bool trt_neigh = inf_and_trt.at(neigh) % 2 == 1;
                        const double prob_jneigh = this->a_inf_b(
                                neigh, j, trt_neigh, trt_j, curr_inf, curr_trt);
                        std::vector<double> grad_jneigh = this->a_inf_b_grad(
                                neigh, j , trt_neigh, trt_j, curr_inf,
                                curr_trt);

                        if (prob_jneigh < 1.0) {
                            njm::linalg::mult_b_to_a(grad_jneigh,
                                    - 1.0 / (1.0 - prob_jneigh));
                            njm::linalg::add_b_to_a(val_to_add, grad_jneigh);
                        }

                    }
                }

                if (change_j % 2 == 1) {
                    // becomes infected
                    if (prob_j > 0.0) {
                        njm::linalg::mult_b_to_a(val_to_add,
                                - (1.0 - prob_j) / prob_j);
                        njm::linalg::add_b_to_a(grad_value, val_to_add);
                    }
                } else {
                    // remains uninfected
                    njm::linalg::add_b_to_a(grad_value, val_to_add);
                }
            }
        }
    }
    njm::linalg::mult_b_to_a(grad_value, 1.0 / history_size);
    return grad_value;
}


std::vector<double> EbolaStateModel::ll_hess(
        const std::vector<Transition<EbolaState> > & history) const {
    const uint32_t history_size = history.size();
    CHECK_GT(history.size(), 0);

    std::vector<double> hess_val(this->par_size_ * this->par_size_, 0.);

    for (uint32_t i = 0; i < history_size; ++i) {
        // split up transition
        const Transition<EbolaState> & transition = history.at(i);
        const EbolaState & curr_state = transition.curr_state;
        const boost::dynamic_bitset<> & curr_trt = transition.curr_trt_bits;
        const EbolaState & next_state = transition.next_state;

        // get inf bits
        const boost::dynamic_bitset<> & curr_inf = curr_state.inf_bits;
        const boost::dynamic_bitset<> & next_inf = next_state.inf_bits;

        // infection probabilities
        const std::vector<double> probs = this->probs(curr_state, curr_trt);

        // combine infection and treatment status
        const std::vector<uint32_t> inf_and_trt =
            njm::tools::combine_sets(curr_inf, curr_trt);

        // get bits for changes in infection
        const boost::dynamic_bitset<> change_inf = curr_inf ^ next_inf;

        // convert bits to sets of indices
        const std::vector<uint32_t> inf_and_change =
            njm::tools::combine_sets(curr_inf, change_inf);

        for (uint32_t j = 0; j < this->num_nodes_; ++j) {
            const uint32_t status_j = inf_and_trt.at(j);
            const uint32_t change_j = inf_and_change.at(j);

            const bool trt_j = status_j % 2 == 1;
            const double prob_j = probs.at(j);

            if (status_j < 2) {
                // was uninfected

                if (prob_j > 0.0 && prob_j < 1.0) {
                    std::vector<double> grad_j(this->par_size());
                    std::vector<double> hess_j(this->par_size());

                    // neighbor probs
                    const Node & node_j = this->network_->get_node(j);
                    const uint32_t num_neigh = node_j.neigh_size();
                    for (uint32_t k = 0; k < num_neigh; ++k) {
                        const uint32_t neigh = node_j.neigh(k);
                        if (inf_and_trt.at(neigh) >= 2) {
                            const bool trt_neigh =
                                inf_and_trt.at(neigh) % 2 == 1;
                            const double prob_ineigh = this->a_inf_b(neigh, j,
                                    trt_neigh, trt_j, curr_inf, curr_trt);
                            // add
                            std::vector<double> add_to_grad(
                                    this->a_inf_b_grad(neigh, j, trt_neigh,
                                            trt_j, curr_inf, curr_trt));
                            njm::linalg::mult_b_to_a(add_to_grad,
                                    - 1.0 / (1.0 - prob_ineigh));
                            njm::linalg::add_b_to_a(grad_j, add_to_grad);

                            std::vector<double> add_to_hess(
                                    this->a_inf_b_hess(neigh, j, trt_neigh,
                                            trt_j, curr_inf, curr_trt));
                            njm::linalg::mult_b_to_a(add_to_hess,
                                    - 1.0 / (1.0 - prob_ineigh));
                            njm::linalg::add_b_to_a(add_to_hess,
                                    njm::linalg::mult_a_and_b(
                                            njm::linalg::outer_a_and_b(
                                                    add_to_grad, add_to_grad),
                                            -1));
                            njm::linalg::add_b_to_a(hess_j, add_to_hess);
                        }
                    }

                    njm::linalg::mult_b_to_a(hess_j,
                            (1.0 - ((change_j % 2) / prob_j)));

                    std::vector<double> outer_grad_j(
                            njm::linalg::outer_a_and_b(grad_j, grad_j));
                    const double scale = std::exp(std::log(1.0 - prob_j)
                            - 2.0 * std::log(prob_j));
                    njm::linalg::mult_b_to_a(outer_grad_j,
                            - ((change_j % 2) * scale));

                    njm::linalg::add_b_to_a(hess_val, hess_j);
                    njm::linalg::add_b_to_a(hess_val, outer_grad_j);
                }
            }
        }
    }

    njm::linalg::mult_b_to_a(hess_val, 1.0 / history_size);
    return hess_val;
}


EbolaState EbolaStateModel::turn_clock(const EbolaState & curr_state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const std::vector<double> probs = this->probs(curr_state, trt_bits);

    EbolaState next_state(curr_state);
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        if (!curr_state.inf_bits.test(i)) { // only uninfected can change state
            const double & prob_i = probs.at(i);

            const double r = this->rng_->runif_01();
            if (r < prob_i) {
                next_state.inf_bits.flip(i);
            }
        }
    }

    return next_state;
}



} // namespace stdmMf
