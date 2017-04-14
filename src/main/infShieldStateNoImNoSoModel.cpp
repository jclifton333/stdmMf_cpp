#include "infShieldStateNoImNoSoModel.hpp"

#include <glog/logging.h>

#include <math.h>

namespace stdmMf {


InfShieldStateNoImNoSoModel::InfShieldStateNoImNoSoModel(
        const std::shared_ptr<const Network> & network)
    : InfShieldStateModel(7, network),
      intcp_inf_latent_(0.0), intcp_inf_(0.0), intcp_rec_(0.0),
      trt_act_inf_(0.0), trt_act_rec_(0.0), trt_pre_inf_(0.0),
      shield_coef_(0.0) {
}


InfShieldStateNoImNoSoModel::InfShieldStateNoImNoSoModel(
        const InfShieldStateNoImNoSoModel & other)
    : InfShieldStateModel(other),
      intcp_inf_latent_(other.intcp_inf_latent_), intcp_inf_(other.intcp_inf_),
      intcp_rec_(other.intcp_rec_), trt_act_inf_(other.trt_act_inf_),
      trt_act_rec_(other.trt_act_rec_), trt_pre_inf_(other.trt_pre_inf_),
      shield_coef_(other.shield_coef_) {
}


std::shared_ptr<Model<InfShieldState> >
InfShieldStateNoImNoSoModel::clone() const {
    return std::shared_ptr<Model<InfShieldState> >(
            new InfShieldStateNoImNoSoModel(*this));
}


std::vector<double> InfShieldStateNoImNoSoModel::par() const {
    std::vector<double> par;
    par.push_back(this->intcp_inf_latent_);
    par.push_back(this->intcp_inf_);
    par.push_back(this->intcp_rec_);
    par.push_back(this->trt_act_inf_);
    par.push_back(this->trt_act_rec_);
    par.push_back(this->trt_pre_inf_);
    par.push_back(this->shield_coef_);
    CHECK_EQ(par.size(), this->par_size());
    return par;
}


void InfShieldStateNoImNoSoModel::par(const std::vector<double> & par) {
    CHECK_EQ(par.size(), this->par_size_);
    std::vector<double>::const_iterator it = par.begin();
    this->intcp_inf_latent_ = *it++;
    this->intcp_inf_ = *it++;
    this->intcp_rec_ = *it++;
    this->trt_act_inf_ = *it++;
    this->trt_act_rec_ = *it++;
    this->trt_pre_inf_ = *it++;
    this->shield_coef_ = *it++;
    CHECK(par.end() == it);
}



double InfShieldStateNoImNoSoModel::inf_b(
        const uint32_t & b_node, const bool & b_trt,
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const double base = this->intcp_inf_latent_ + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";
    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

double InfShieldStateNoImNoSoModel::a_inf_b(
        const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt,
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const double base = this->intcp_inf_ + this->trt_act_inf_ * a_trt
        + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

double InfShieldStateNoImNoSoModel::rec_b(
        const uint32_t & b_node, const bool & b_trt,
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const double base = this->intcp_rec_ + this->trt_act_rec_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

std::vector<double> InfShieldStateNoImNoSoModel::inf_b_grad(
        const uint32_t & b_node, const bool & b_trt,
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const double base = this->intcp_inf_latent_ + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));
    const double val = std::exp(base - 2.0 * std::log(1.0 + expBase));

    std::vector<double> grad_val(this->par_size_, 0.0);
    grad_val.at(0) = val;
    grad_val.at(5) = b_trt * val;
    return grad_val;
}

std::vector<double> InfShieldStateNoImNoSoModel::a_inf_b_grad(
        const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt,
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
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

std::vector<double> InfShieldStateNoImNoSoModel::rec_b_grad(
        const uint32_t & b_node, const bool & b_trt,
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const double base = this->intcp_rec_ + this->trt_act_rec_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));
    const double val = std::exp(base - 2.0 * std::log(1.0 + expBase));

    std::vector<double> grad_val(this->par_size_, 0.0);
    grad_val.at(2) = val;
    grad_val.at(4) = b_trt * val;
    return grad_val;
}


std::vector<double> InfShieldStateNoImNoSoModel::inf_b_hess(
        const uint32_t & b_node, const bool & b_trt,
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const std::vector<double> inner_grad({1, 0, 0, 0, 0,
                    static_cast<double>(b_trt),
                    0});

    const double base = this->intcp_inf_latent_ + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));

    const double expitBase = 1.0 - 1.0 / (1.0 + expBase);
    const double val = expitBase * (1.0 - expitBase) * (1.0 - 2.0 * expitBase);

    std::vector<double> hess_val;
    hess_val.reserve(this->par_size_ * this->par_size_);
    for (uint32_t i = 0; i < this->par_size_; ++i) {
        for (uint32_t j = 0; j < this->par_size_; ++j) {
            hess_val.push_back(val * inner_grad.at(i) *
                    inner_grad.at(j));
        }
    }
    CHECK_EQ(hess_val.size(), this->par_size_ * this->par_size_);
    return hess_val;
}

std::vector<double> InfShieldStateNoImNoSoModel::a_inf_b_hess(
        const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt,
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const std::vector<double> inner_grad({0, 1, 0, static_cast<double>(a_trt),
                    0, static_cast<double>(b_trt),
                    0});

    const double base = this->intcp_inf_ + this->trt_act_inf_ * a_trt
        + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));

    const double expitBase = 1.0 - 1.0 / (1.0 + expBase);
    const double val = expitBase * (1.0 - expitBase) * (1.0 - 2.0 * expitBase);

    std::vector<double> hess_val;
    hess_val.reserve(this->par_size_ * this->par_size_);
    for (uint32_t i = 0; i < this->par_size_; ++i) {
        for (uint32_t j = 0; j < this->par_size_; ++j) {
            hess_val.push_back(val * inner_grad.at(i) *
                    inner_grad.at(j));
        }
    }
    CHECK_EQ(hess_val.size(), this->par_size_ * this->par_size_);

    return hess_val;
}

std::vector<double> InfShieldStateNoImNoSoModel::rec_b_hess(
        const uint32_t & b_node, const bool & b_trt,
        const InfShieldState & state,
        const boost::dynamic_bitset<> & trt_bits) const {
    const std::vector<double> inner_grad({0, 0, 1, 0,
                    static_cast<double>(b_trt), 0,
                    0});

    const double base = this->intcp_rec_ + this->trt_act_rec_ * b_trt;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));

    const double expitBase = 1.0 - 1.0 / (1.0 + expBase);
    const double val = expitBase * (1.0 - expitBase) * (1.0 - 2.0 * expitBase);

    std::vector<double> hess_val;
    hess_val.reserve(this->par_size_ * this->par_size_);
    for (uint32_t i = 0; i < this->par_size_; ++i) {
        for (uint32_t j = 0; j < this->par_size_; ++j) {
            hess_val.push_back(val * inner_grad.at(i) *
                    inner_grad.at(j));
        }
    }
    CHECK_EQ(hess_val.size(), this->par_size_ * this->par_size_);

    return hess_val;
}


double InfShieldStateNoImNoSoModel::shield_draw(
        const uint32_t & loc, const InfShieldState & curr_state) const {
    const double r(this->rng_->rnorm_01());
    std::cout << "shield for " << loc << std::endl
              << "curr: " << curr_state.shield.at(loc) << std::endl
              << "coef: " << this->shield_coef_ << std::endl
              << "rand: " << r << std::endl;
    return curr_state.shield.at(loc) * this->shield_coef_
        + r;
}


double InfShieldStateNoImNoSoModel::shield_prob(
        const uint32_t & loc, const InfShieldState & curr_state,
        const InfShieldState & next_state,
        const bool & log_scale) const {
    const double diff_shield = next_state.shield.at(loc)
        - this->shield_coef_ * curr_state.shield.at(loc);

    // this is on the log scale
    const double p = - 0.5 * std::log(2 * M_PI)
        - 0.5 * diff_shield * diff_shield;
    if (log_scale) {
        return p;
    } else {
        return std::exp(p);
    }
}


std::vector<double> InfShieldStateNoImNoSoModel::shield_grad(
        const uint32_t & loc, const InfShieldState & curr_state,
        const InfShieldState & next_state) const {
    const double & curr_shield = curr_state.shield.at(loc);
    const double & next_shield = next_state.shield.at(loc);
    const double diff_shield = next_shield - this->shield_coef_ * curr_shield;

    std::vector<double> grad_val(this->par_size(), 0.0);
    grad_val.at(this->par_size() - 1) = curr_shield * diff_shield;

    return grad_val;
}


std::vector<double> InfShieldStateNoImNoSoModel::shield_hess(
        const uint32_t & loc, const InfShieldState & curr_state,
        const InfShieldState & next_state) const {
    const double & curr_shield = curr_state.shield.at(loc);

    std::vector<double> hess_val(this->par_size() * this->par_size(), 0.0);
    hess_val.at(hess_val.size() - 1) = - curr_shield * curr_shield;

    return hess_val;
}





} // namespace stdmMf
