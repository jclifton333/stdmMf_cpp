#include "noCovEdgeModel.hpp"
#include "utilities.hpp"
#include <glog/logging.h>


namespace stdmMf {


NoCovEdgeModel::NoCovEdgeModel(const std::shared_ptr<const Network> & network)
    : par_size_(6), intcp_inf_latent_(0.0), intcp_inf_(0.0), intcp_rec_(0.0),
      trt_act_inf_(0.0), trt_act_rec_(0.0), trt_pre_inf_(0.0),
      network_(network), num_nodes_(this->network_->size()) {
}


std::vector<double> NoCovEdgeModel::par() const {
    std::vector<double> par;
    par.push_back(this->intcp_inf_latent_);
    par.push_back(this->intcp_inf_);
    par.push_back(this->intcp_rec_);
    par.push_back(this->trt_act_inf_);
    par.push_back(this->trt_act_rec_);
    par.push_back(this->trt_pre_inf_);
    return par;
}


void NoCovEdgeModel::par(const std::vector<double> & par) {
    CHECK_EQ(par.size(), this->par_size_);
    std::vector<double>::const_iterator it = par.begin();
    this->intcp_inf_latent_ = *it++;
    this->intcp_inf_ = *it++;
    this->intcp_rec_ = *it++;
    this->trt_act_inf_ = *it++;
    this->trt_act_rec_ = *it++;
    this->trt_pre_inf_ = *it++;
}


uint32_t NoCovEdgeModel::par_size() const {
    return this->par_size_;
}


void NoCovEdgeModel::est_par(const std::vector<BitsetPair> & history) {
    // TODO: Need ll() and ll_grad() first
}


std::vector<double> NoCovEdgeModel::probs(
        const boost::dynamic_bitset<> & inf_status,
        const boost::dynamic_bitset<> & trt_status) const {
    std::vector<double> probs;
    std::vector<uint32_t> status = combine_sets(inf_status, trt_status);
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        const uint32_t status_i = status.at(i);
        const bool trt_i = (status_i % 2) == 1;
        if (status_i < 2) {
            // not infected -> infection probability
            const Node & node = this->network_->get_node(i);
            const uint32_t num_neigh = node.neigh_size();

            double prob = 1.0 - this->inf_b(i, trt_i); // latent prob

            // factor in neighbors
            for (uint32_t j = 0; j < num_neigh; ++i) {
                const uint32_t neigh = node.neigh(j);
                const uint32_t status_neigh = status.at(neigh);
                if (status_neigh >= 2) {
                    // if neighbor is infected
                    const bool trt_neigh = (status_neigh % 2) == 1;
                    prob *= 1.0 - this->a_inf_b(neigh, i, trt_neigh, trt_i);
                }
            }
            probs.push_back(1.0 - prob);
        } else {
            // infected -> recovery probability
            const double prob = this->rec_b(i, trt_i);
            probs.push_back(prob);
        }
    }
    return probs;
}


double NoCovEdgeModel::ll(const std::vector<BitsetPair> & history) const {
    double ll_value = 0.0;
    const uint32_t history_size = history.size();
    for (uint32_t i = 0; i < (history_size - 1); ++i) {
        const BitsetPair & curr_history = history.at(i);
        const boost::dynamic_bitset<> & curr_inf = curr_history.first;
        const boost::dynamic_bitset<> & curr_trt = curr_history.second;
        // infection probabilities
        const std::vector<double> probs = this->probs(curr_inf, curr_trt);

        const boost::dynamic_bitset<> & next_inf = history.at(i + 1).first;

        // get bits for changes in infection
        const boost::dynamic_bitset<> & change_inf = curr_inf ^ next_inf;

        // convert bits to sets of indices
        const auto change_both_sets = both_sets(change_inf);
        const std::vector<uint32_t> changed = change_both_sets.first;
        const uint32_t num_changed = changed.size();
        const std::vector<uint32_t> unchanged = change_both_sets.second;
        const uint32_t num_unchanged = unchanged.size();

        for (uint32_t i = 0; i < num_changed; ++i) {
            const double p = probs.at(changed.at(i));
            ll_value += std::log(std::max(1e-14, p)); // for stability
        }
        for (uint32_t i = 0; i < num_unchanged; ++i) {
            const double p = 1.0 - probs.at(unchanged.at(i));
            ll_value += std::log(std::max(1e-14, p));
        }
    }
    return ll_value;
}


std::vector<double> NoCovEdgeModel::ll_grad(
        const std::vector<BitsetPair> & history) const {
    std::vector<double> grad_value (this->par_size_, 0.0);
    const uint32_t history_size = history.size();
    for (uint32_t i = 0; i < (history_size - 1); ++i) {
        const BitsetPair & curr_history = history.at(i);
        const boost::dynamic_bitset<> curr_inf = curr_history.first;
        const boost::dynamic_bitset<> curr_trt = curr_history.second;
        const std::vector<uint32_t> inf_and_trt =
            combine_sets(curr_inf, curr_trt);

        // infection probabilities
        const std::vector<double> probs = this->probs(curr_inf, curr_trt);

        const boost::dynamic_bitset<> next_inf = history.at(i + 1).first;

        // get bits for changes in infection
        const boost::dynamic_bitset<> change_inf = curr_inf ^ next_inf;

        // convert bits to sets of indices
        const std::vector<uint32_t> inf_and_change =
            combine_sets(curr_inf, change_inf);
        for (uint32_t j = 0; j < this->num_nodes_; ++i) {
            const uint32_t status_j = inf_and_trt.at(j);
            const uint32_t change_j = inf_and_change.at(j);

            const uint32_t trt_j = (status_j % 2) == 1;
            const double prob_j = probs.at(j);
            if (change_j < 2) {
                // was uninfected
                std::vector<double> grad = this->rec_b_grad(j, trt_j);
                if ((change_j % 2) == 0) {
                    // remains uninfected
                    if (prob_j > 0.0) {
                        mult_b_to_a(grad, 1.0 / prob_j);
                        add_b_to_a(grad_value, grad);
                    }
                } else {
                    // became infected
                    if (prob_j < 1.0) {
                        mult_b_to_a(grad, 1.0 / (1.0 - prob_j));
                        add_b_to_a(grad_value, grad);
                    }
                }
            } else {
                // was infected
                if ((change_j % 2) == 0) {
                    // remains infected
                } else {
                    // became uninfected
                }
            }
        }
    }
}


double NoCovEdgeModel::inf_b(const uint32_t & b_node,
        const bool & b_trt) const {
    const double base = this->intcp_inf_latent_ + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

double NoCovEdgeModel::a_inf_b(const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt) const {
    const double base = this->intcp_inf_ + this->trt_act_inf_ * a_trt
        + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

double NoCovEdgeModel::rec_b(const uint32_t & b_node,
        const bool & b_trt) const {
    const double base = this->intcp_rec_ + this->trt_act_rec_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}



} // namespace stdmMf
