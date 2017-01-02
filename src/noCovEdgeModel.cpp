#include "noCovEdgeModel.hpp"
#include "utilities.hpp"
#include <glog/logging.h>


namespace stdmMf {


NoCovEdgeModel::NoCovEdgeModel(const std::shared_ptr<const Network> & network)
    : par_size_(6), intcp_inf_latent_(0.0), intcp_inf_(0.0), intcp_rec_(0.0),
      trt_act_inf_(0.0), trt_act_rec_(0.0), trt_pre_inf_(0.0),
      network_(network), num_nodes_(this->network_->size()) {
}

NoCovEdgeModel::NoCovEdgeModel(const NoCovEdgeModel & other)
    : par_size_(other.par_size_), intcp_inf_latent_(other.intcp_inf_latent_),
      intcp_inf_(other.intcp_inf_), intcp_rec_(other.intcp_rec_),
      trt_act_inf_(other.trt_act_inf_), trt_act_rec_(other.trt_act_rec_),
      trt_pre_inf_(other.trt_pre_inf_), network_(other.network_->clone()),
      num_nodes_(other.num_nodes_) {
}

std::shared_ptr<Model> NoCovEdgeModel::clone() const {
    return std::shared_ptr<Model>(new NoCovEdgeModel(*this));
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


std::vector<double> NoCovEdgeModel::probs(
        const boost::dynamic_bitset<> & inf_status,
        const boost::dynamic_bitset<> & trt_status) const {
    std::vector<double> probs;
    std::vector<uint32_t> status = combine_sets(inf_status, trt_status);
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        const uint32_t status_i = status.at(i);
        const bool trt_i = status_i % 2 == 1;
        if (status_i < 2) {
            // not infected -> infection probability
            const Node & node = this->network_->get_node(i);
            const uint32_t num_neigh = node.neigh_size();

            double prob = 1.0 - this->inf_b(i, trt_i); // latent prob

            // factor in neighbors
            for (uint32_t j = 0; j < num_neigh; ++j) {
                const uint32_t neigh = node.neigh(j);
                const uint32_t status_neigh = status.at(neigh);
                if (status_neigh >= 2) {
                    // if neighbor is infected
                    const bool trt_neigh = status_neigh % 2 == 1;
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


std::vector<double> NoCovEdgeModel::ll_grad(
        const std::vector<BitsetPair> & history) const {
    std::vector<double> grad_value (this->par_size_, 0.0);
    const uint32_t history_size = history.size();
    for (uint32_t i = 0; i < (history_size - 1); ++i) {
        // current infection and treatments
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
                                neigh, j, trt_neigh, trt_j);
                        std::vector<double> grad_jneigh = this->a_inf_b_grad(
                                neigh, j , trt_neigh, trt_j);

                        if (prob_jneigh < 1.0) {
                            mult_b_to_a(grad_jneigh,
                                    - 1.0 / (1.0 - prob_jneigh));
                            add_b_to_a(val_to_add, grad_jneigh);
                        }

                    }
                }

                // latent effect
                {
                    const double prob_j_latent = this->inf_b(j, trt_j);
                    std::vector<double> grad_j_latent = this->inf_b_grad(
                            j, trt_j);
                    if (prob_j_latent < 1.0) {
                        mult_b_to_a(grad_j_latent,
                                - 1.0 / (1.0 - prob_j_latent));
                        add_b_to_a(val_to_add, grad_j_latent);
                    }
                }


                if (change_j % 2 == 1) {
                    // becomes infected
                    if (prob_j > 0.0) {
                        mult_b_to_a(val_to_add, - (1.0 - prob_j) / prob_j);
                        add_b_to_a(grad_value, val_to_add);
                    }
                } else {
                    // remains uninfected
                    add_b_to_a(grad_value, val_to_add);
                }
            } else {
                // was infected
                std::vector<double> grad = this->rec_b_grad(j, trt_j);
                if (change_j % 2 == 1) {
                    // becomes uninfected
                    if (prob_j > 0.0) {
                        mult_b_to_a(grad, 1.0 / prob_j);
                        add_b_to_a(grad_value, grad);
                    }
                } else {
                    // remains infected
                    if (prob_j < 1.0) {
                        mult_b_to_a(grad, - 1.0 / (1.0 - prob_j));
                        add_b_to_a(grad_value, grad);
                    }
                }
            }
        }
    }
    mult_b_to_a(grad_value, 1.0 / (history_size - 1));
    return grad_value;
}


std::vector<double> NoCovEdgeModel::ll_hess(
        const std::vector<BitsetPair> & history) const {
    LOG(FATAL) << "not implemented yet";
    return std::vector<double>();
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

std::vector<double> NoCovEdgeModel::inf_b_grad(const uint32_t & b_node,
        const bool & b_trt) const {
    const double base = this->intcp_inf_latent_ + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));
    const double val = std::exp(base - 2.0 * std::log(1.0 + expBase));

    std::vector<double> grad_val(this->par_size_, 0.0);
    grad_val.at(0) = val;
    grad_val.at(5) = b_trt * val;
    return grad_val;
}

std::vector<double> NoCovEdgeModel::a_inf_b_grad(
        const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt) const {
    const double base = this->intcp_inf_ + this->trt_act_inf_ * a_trt
        + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));
    const double val = std::exp(base - 2.0 * std::log(1.0 + expBase));

    std::vector<double> grad_val(this->par_size_, 0.0);
    grad_val.at(1) = val;
    grad_val.at(3) = a_trt * val;
    grad_val.at(5) = b_trt * val;
    return grad_val;
}

std::vector<double> NoCovEdgeModel::rec_b_grad(
        const uint32_t & b_node, const bool & b_trt) const {
    const double base = this->intcp_rec_ + this->trt_act_rec_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));
    const double val = std::exp(base - 2.0 * std::log(1.0 + expBase));

    std::vector<double> grad_val(this->par_size_, 0.0);
    grad_val.at(2) = val;
    grad_val.at(4) = b_trt * val;
    return grad_val;
}



} // namespace stdmMf
