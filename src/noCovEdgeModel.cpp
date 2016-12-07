#include "noCovEdgeModel.hpp"
#include <glog/logging.h>


NoCovEdgeModel::NoCovEdgeModel()
    : par_size_(6), intcp_inf_latent_(0.0), intcp_inf_(0.0), intcp_rec_(0.0),
      trt_act_inf_(0.0), trt_act_rec_(0.0), trt_pre_inf_(0.0) {
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
    std::vector<dobule>::const_iterator it = par.begin();
    this->intcp_inf_latent_ = it++;
    this->intcp_inf_ = it++;
    this->intcp_rec_ = it++;
    this->trt_act_inf_ = it++;
    this->trt_act_rec_ = it++;
    this->trt_pre_inf_ = it++;
}


uint32_t NoCovEdgeModel::par_size() const {
    return this->par_size_;
}


void NoCovEdgeModel::est_par() const {
    // TODO: Need ll() and ll_grad() first
}


std::vector<double> NoCovEdgeModel::probs(
        const boost::dynamic_bitset<> & inf_status,
        const boost::dynamic_bitset<> & trt_status) const {
    // TODO: Need intermediate probability functions first
}


double NoCovEdgeModel::ll(const std::vector<int_trt_par> & history) const {
    // TODO: Need intermediate probability functions first
}


std::vector<double> NoCovEdgeModel::ll_grad(
        const std::vector<inf_trt_pair> & history) const {
    // TODO: Need intermediate gradient functions first
}



double NoCovEdgeModel::inf_b(const uint32_t & b_node,
        const bool & b_trt) const {
    const double base = this->intcp_inf_latent_ + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, ! isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

double NoCovEdgeModel::a_inf_b(const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt) const {
    const double base = this->intcp_inf_ + this->trt_act_inf_ * a_trt
        + this->trt_pre_inf_ * b_trt;
    LOG_IF(FATAL, ! isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

double NoCovEdgeModel::rec_b(const uint32_t & b_node,
        const bool & b_trt) const {
    const double base = this->intcp_rec_ + this->trt_act_rec_ * b_trt;
    LOG_IF(FATAL, ! isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}
