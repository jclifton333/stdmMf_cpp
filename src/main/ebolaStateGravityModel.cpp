#include "ebolaStateGravityModel.hpp"

#include <glog/logging.h>

#include <njm_cpp/linalg/stdVectorAlgebra.hpp>


namespace stdmMf {


EbolaStateGravityModel::EbolaStateGravityModel(
        const std::shared_ptr<const Network> & network)
    : EbolaStateModel(5, network),
      beta_0_(-5.246), beta_1_(-155.8), beta_2_(0.186),
      trt_pre_(-2.0), trt_act_(-1.0) {
}


EbolaStateGravityModel::EbolaStateGravityModel(
        const EbolaStateGravityModel & other)
    : EbolaStateModel(other),
      beta_0_(other.beta_0_), beta_1_(other.beta_1_), beta_2_(other.beta_2_),
      trt_pre_(other.trt_pre_), trt_act_(other.trt_act_) {
}


std::shared_ptr<Model<EbolaState> > EbolaStateGravityModel::clone() const {
    return std::shared_ptr<Model<EbolaState> > (
            new EbolaStateGravityModel(*this));
}


std::vector<double> EbolaStateGravityModel::par() const {
    return {this->beta_0_, this->beta_1_, this->beta_2_,
            this->trt_pre_, this->trt_act_};
}


void EbolaStateGravityModel::par(const std::vector<double> & par) {
    CHECK_EQ(par.size(), this->par_size_);
    this->beta_0_ = par.at(0);
    this->beta_1_ = par.at(1);
    this->beta_2_ = par.at(2);
    this->trt_pre_ = par.at(3);
    this->trt_act_ = par.at(4);
}


double EbolaStateGravityModel::a_inf_b(
        const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt,
        const EbolaState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const double log_grav_term(
            std::log(this->network_->dist().at(a_node).at(b_node))
                    - this->beta_2_ * (std::log(state.pop.at(a_node))
                            + std::log(state.pop.at(b_node))));
    double logit_prob(this->beta_0_ +
            this->beta_1_ * std::exp(log_grav_term));
    if (a_trt) {
        logit_prob += this->trt_act_;
    }

    if (b_trt) {
        logit_prob += this->trt_pre_;
    }

    return 1.0 - 1.0 / (1.0 + std::exp(logit_prob));
}


std::vector<double> EbolaStateGravityModel::a_inf_b_grad(
        const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt,
        const EbolaState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const double pop(state.pop.at(a_node) * state.pop.at(b_node));
    const double log_grav_term(
            std::log(this->network_->dist().at(a_node).at(b_node))
            - this->beta_2_ * (std::log(state.pop.at(a_node))
                    + std::log(state.pop.at(b_node))));

    double logit_prob(this->beta_0_ +
            this->beta_1_ * std::exp(log_grav_term));
    if (a_trt) {
        logit_prob += this->trt_act_;
    }

    if (b_trt) {
        logit_prob += this->trt_pre_;
    }

    const double prob(1.0 - 1.0 / (1.0 + std::exp(logit_prob)));

    std::vector<double> grad(this->par_size_, prob * (1.0 - prob));

    grad.at(1) *= std::exp(log_grav_term);
    grad.at(2) *= this->beta_1_ * this->network_->dist().at(a_node).at(b_node)
        * std::pow(pop, - this->beta_2_) * (- std::log(pop));
    grad.at(3) *= b_trt;
    grad.at(4) *= a_trt;

    return grad;
}


std::vector<double> EbolaStateGravityModel::a_inf_b_hess(
        const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt,
        const EbolaState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const double pop(state.pop.at(a_node) * state.pop.at(b_node));
    const double log_grav_term(
            std::log(this->network_->dist().at(a_node).at(b_node))
            - this->beta_2_ * (std::log(state.pop.at(a_node))
                    + std::log(state.pop.at(b_node))));

    double logit_prob(this->beta_0_ +
            this->beta_1_ * std::exp(log_grav_term));
    if (a_trt) {
        logit_prob += this->trt_act_;
    }

    if (b_trt) {
        logit_prob += this->trt_pre_;
    }

    const double prob(1.0 - 1.0 / (1.0 + std::exp(logit_prob)));

    std::vector<double> grad_logit(this->par_size_, 1.0);
    grad_logit.at(1) = std::exp(log_grav_term);
    grad_logit.at(2) = this->beta_1_
        * this->network_->dist().at(a_node).at(b_node)
        * std::pow(pop, - this->beta_2_) * (- std::log(pop));
    grad_logit.at(3) *= b_trt;
    grad_logit.at(4) *= a_trt;

    std::vector<double> hess_one(this->par_size_ * this->par_size_, 0.0);
    const std::vector<double> hess_two(njm::linalg::outer_a_and_b(
                    njm::linalg::mult_a_and_b(grad_logit,
                            prob * (1.0 - prob) * (1.0 - 2.0 * prob)),
                    grad_logit));

    hess_one.at(1 * this->par_size_ + 2) =
        hess_one.at(2 * this->par_size_ + 1) =
        this->network_->dist().at(a_node).at(b_node)
        * std::pow(pop, - this->beta_2_) * (- std::log(pop));
    hess_one.at(2 * this->par_size_ + 2) =
        this->beta_1_ * this->network_->dist().at(a_node).at(b_node)
        * std::pow(pop, - this->beta_2_) * std::log(pop) * std::log(pop);

    return njm::linalg::add_a_and_b(hess_one, hess_two);
}



} // namespace stdmMf
