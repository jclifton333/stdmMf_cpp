#include "infStateOrSoModel.hpp"
#include <njm_cpp/tools/bitManip.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <glog/logging.h>


namespace stdmMf {


InfStateOrSoModel::InfStateOrSoModel(
        const std::shared_ptr<const Network> & network)
    : InfStateModel(6, network),
      intcp_inf_latent_(0.0), intcp_inf_(0.0), intcp_rec_(0.0),
      trt_act_inf_(0.0), trt_act_rec_(0.0), trt_pre_inf_(0.0) {
}


InfStateOrSoModel::InfStateOrSoModel(const InfStateOrSoModel & other)
    : InfStateModel(other),
      intcp_inf_latent_(other.intcp_inf_latent_),
      intcp_inf_(other.intcp_inf_), intcp_rec_(other.intcp_rec_),
      trt_act_inf_(other.trt_act_inf_), trt_act_rec_(other.trt_act_rec_),
      trt_pre_inf_(other.trt_pre_inf_) {
}


std::shared_ptr<Model<InfState> > InfStateOrSoModel::clone() const {
    return std::shared_ptr<Model<InfState> >(new InfStateOrSoModel(*this));
}


std::vector<double> InfStateOrSoModel::par() const {
    std::vector<double> par;
    par.push_back(this->intcp_inf_latent_);
    par.push_back(this->intcp_inf_);
    par.push_back(this->intcp_rec_);
    par.push_back(this->trt_act_inf_);
    par.push_back(this->trt_act_rec_);
    par.push_back(this->trt_pre_inf_);
    return par;
}


void InfStateOrSoModel::par(const std::vector<double> & par) {
    CHECK_EQ(par.size(), this->par_size_);
    std::vector<double>::const_iterator it = par.begin();
    this->intcp_inf_latent_ = *it++;
    this->intcp_inf_ = *it++;
    this->intcp_rec_ = *it++;
    this->trt_act_inf_ = *it++;
    this->trt_act_rec_ = *it++;
    this->trt_pre_inf_ = *it++;
}


double InfStateOrSoModel::inf_b(const uint32_t & b_node,
        const bool & b_trt,
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) const {
    bool b_trt_so = false;
    if (b_trt) {
        b_trt_so = true;
    } else {
        const Node & b = this->network_->get_node(b_node);
        const uint32_t num_neigh = b.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = b.neigh(i);
            if (!inf_bits.test(neigh) && trt_bits.test(neigh)) {
                b_trt_so = true;
                break;
            }
        }
    }

    const double base = this->intcp_inf_latent_ + this->trt_pre_inf_ * b_trt_so;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";
    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

double InfStateOrSoModel::a_inf_b(const uint32_t & a_node,
        const uint32_t & b_node, const bool & a_trt, const bool & b_trt,
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) const {
    bool a_trt_so = false;
    if (a_trt) {
        a_trt_so = true;
    } else {
        const Node & a = this->network_->get_node(a_node);
        const uint32_t num_neigh = a.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = a.neigh(i);
            if (inf_bits.test(neigh) && trt_bits.test(neigh)) {
                a_trt_so = true;
                break;
            }
        }
    }

    bool b_trt_so = false;
    if (b_trt) {
        b_trt_so = true;
    } else {
        const Node & b = this->network_->get_node(b_node);
        const uint32_t num_neigh = b.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = b.neigh(i);
            if (!inf_bits.test(neigh) && trt_bits.test(neigh)) {
                b_trt_so = true;
                break;
            }
        }
    }

    const double base = this->intcp_inf_ + this->trt_act_inf_ * a_trt_so
        + this->trt_pre_inf_ * b_trt_so;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

double InfStateOrSoModel::rec_b(const uint32_t & b_node,
        const bool & b_trt,
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) const {
    bool b_trt_so = false;
    if (b_trt) {
        b_trt_so = true;
    } else {
        const Node & b = this->network_->get_node(b_node);
        const uint32_t num_neigh = b.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = b.neigh(i);
            if (inf_bits.test(neigh) && trt_bits.test(neigh)) {
                b_trt_so = true;
                break;
            }
        }
    }

    const double base = this->intcp_rec_ + this->trt_act_rec_ * b_trt_so;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    return 1.0 - 1.0 / (1.0 + std::exp(std::min(100.0, base)));
}

std::vector<double> InfStateOrSoModel::inf_b_grad(const uint32_t & b_node,
        const bool & b_trt,
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) const {
    bool b_trt_so = false;
    if (b_trt) {
        b_trt_so = true;
    } else {
        const Node & b = this->network_->get_node(b_node);
        const uint32_t num_neigh = b.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = b.neigh(i);
            if (!inf_bits.test(neigh) && trt_bits.test(neigh)) {
                b_trt_so = true;
                break;
            }
        }
    }

    const double base = this->intcp_inf_latent_ + this->trt_pre_inf_ * b_trt_so;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));
    const double val = std::exp(base - 2.0 * std::log(1.0 + expBase));

    std::vector<double> grad_val(this->par_size_, 0.0);
    grad_val.at(0) = val;
    grad_val.at(5) = b_trt_so * val;
    return grad_val;
}

std::vector<double> InfStateOrSoModel::a_inf_b_grad(
        const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt,
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) const {
    bool a_trt_so = false;
    if (a_trt) {
        a_trt_so = true;
    } else {
        const Node & a = this->network_->get_node(a_node);
        const uint32_t num_neigh = a.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = a.neigh(i);
            if (inf_bits.test(neigh) && trt_bits.test(neigh)) {
                a_trt_so = true;
                break;
            }
        }
    }

    bool b_trt_so = false;
    if (b_trt) {
        b_trt_so = true;
    } else {
        const Node & b = this->network_->get_node(b_node);
        const uint32_t num_neigh = b.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = b.neigh(i);
            if (!inf_bits.test(neigh) && trt_bits.test(neigh)) {
                b_trt_so = true;
                break;
            }
        }
    }

    const double base = this->intcp_inf_ + this->trt_act_inf_ * a_trt_so
        + this->trt_pre_inf_ * b_trt_so;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));
    const double val = std::exp(base - 2.0 * std::log(1.0 + expBase));

    std::vector<double> grad_val(this->par_size_, 0.0);
    grad_val.at(1) = val;
    grad_val.at(3) = a_trt_so * val;
    grad_val.at(5) = b_trt_so * val;
    return grad_val;
}

std::vector<double> InfStateOrSoModel::rec_b_grad(
        const uint32_t & b_node, const bool & b_trt,
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) const {
    bool b_trt_so = false;
    if (b_trt) {
        b_trt_so = true;
    } else {
        const Node & b = this->network_->get_node(b_node);
        const uint32_t num_neigh = b.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = b.neigh(i);
            if (inf_bits.test(neigh) && trt_bits.test(neigh)) {
                b_trt_so = true;
                break;
            }
        }
    }

    const double base = this->intcp_rec_ + this->trt_act_rec_ * b_trt_so;
    LOG_IF(FATAL, !std::isfinite(base)) << "base is not finite.";

    const double expBase = std::exp(std::min(100.0, base));
    const double val = std::exp(base - 2.0 * std::log(1.0 + expBase));

    std::vector<double> grad_val(this->par_size_, 0.0);
    grad_val.at(2) = val;
    grad_val.at(4) = b_trt_so * val;
    return grad_val;
}


std::vector<double> InfStateOrSoModel::inf_b_hess(const uint32_t & b_node,
        const bool & b_trt,
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) const {
    bool b_trt_so = false;
    if (b_trt) {
        b_trt_so = true;
    } else {
        const Node & b = this->network_->get_node(b_node);
        const uint32_t num_neigh = b.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = b.neigh(i);
            if (!inf_bits.test(neigh) && trt_bits.test(neigh)) {
                b_trt_so = true;
                break;
            }
        }
    }

    const std::vector<double> inner_grad({1, 0, 0, 0, 0,
                    static_cast<double>(b_trt_so)});

    const double base = this->intcp_inf_latent_ + this->trt_pre_inf_ * b_trt_so;
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

std::vector<double> InfStateOrSoModel::a_inf_b_hess(
        const uint32_t & a_node, const uint32_t & b_node,
        const bool & a_trt, const bool & b_trt,
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) const {
    bool a_trt_so = false;
    if (a_trt) {
        a_trt_so = true;
    } else {
        const Node & a = this->network_->get_node(a_node);
        const uint32_t num_neigh = a.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = a.neigh(i);
            if (inf_bits.test(neigh) && trt_bits.test(neigh)) {
                a_trt_so = true;
                break;
            }
        }
    }

    bool b_trt_so = false;
    if (b_trt) {
        b_trt_so = true;
    } else {
        const Node & b = this->network_->get_node(b_node);
        const uint32_t num_neigh = b.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = b.neigh(i);
            if (!inf_bits.test(neigh) && trt_bits.test(neigh)) {
                b_trt_so = true;
                break;
            }
        }
    }

    const std::vector<double> inner_grad({0, 1, 0,
                    static_cast<double>(a_trt_so), 0,
                    static_cast<double>(b_trt_so)});

    const double base = this->intcp_inf_ + this->trt_act_inf_ * a_trt_so
        + this->trt_pre_inf_ * b_trt_so;
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

std::vector<double> InfStateOrSoModel::rec_b_hess(
        const uint32_t & b_node, const bool & b_trt,
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) const {
    bool b_trt_so = false;
    if (b_trt) {
        b_trt_so = true;
    } else {
        const Node & b = this->network_->get_node(b_node);
        const uint32_t num_neigh = b.neigh_size();
        for (uint32_t i = 0; i < num_neigh; i++) {
            const uint32_t neigh = b.neigh(i);
            if (inf_bits.test(neigh) && trt_bits.test(neigh)) {
                b_trt_so = true;
                break;
            }
        }
    }

    const std::vector<double> inner_grad({0, 0, 1, 0,
                    static_cast<double>(b_trt_so), 0});

    const double base = this->intcp_rec_ + this->trt_act_rec_ * b_trt_so;
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



} // namespace stdmMf
