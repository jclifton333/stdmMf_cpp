#include "networkRunFeatures.hpp"

namespace stdmMf {

NetworkRunFeatures::NetworkRunFeatures(
        const std::shared_ptr<const Network> & network,
        const uint32_t & run_length)
    : network_(network), num_nodes_(network->size()), run_length_(run_length),
      runs_(network->runs_of_len_cumu(run_length)),
      runs_by_node_(network->split_by_node(this->runs_)),
      num_runs_(this->runs_.size()){

    this->offset_.resize(run_length);
    uint32_t curr_offset_val = 1;
    for (uint32_t i = 0; i < run_length; ++i) {
        this->offset_.at(i) = curr_offset_val;
        curr_offset_val += 1 << (2*(i+1));
    }
    this->num_features_ = curr_offset_val;

}


std::vector<double> NetworkRunFeatures::get_features(
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) {
    std::vector<double> feat(this->num_features(), 0.0);
    feat.at(0) = 1.0; // intercept

    for (uint32_t i = 0; i < this->num_runs_; ++i) {
        const NetworkRun & nr = this->runs_.at(i);
        const uint32_t run_len = nr.nodes.size();

        boost::dynamic_bitset<> inf_mask(run_length_);
        boost::dynamic_bitset<> trt_mask(run_length_);
        for (uint32_t j = 0; j < run_len; i++) {
            if (inf_bits.check(nr.nodes.at(j))) {
                inf_mask.set(j);
            }

            if (trt_bits.check(nr.nodes.at(j))) {
                trt_mask.set(j);
            }
        }

        if (!inf_mask.all() || !trt_mask.all()) {
            // TODO: Need to figure out the indexing
        }
    }
}


std::vector<double> NetworkRunFeatures::update_features(
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits,
        const std::vector<double> & prev_feat) {
}


} // namespace stdmMf
