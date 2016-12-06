#include "noCovEdgeModel.hpp"


NoCovEdgeModel::NoCovEdgeModel()
    : par_size_(6), par_(par_size_) {
}


std::vector<double> NoCovEdgeModel::par() const {
    return this->par_;
}


void NoCovEdgeModel::par(const std::vector<double> & par) {
    this->par_ = par;
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
