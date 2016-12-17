#include "networkRunFeatures.hpp"

namespace stdmMf {

NetworkRunFeatures::NetworkRunFeatures(
        const std::shared_ptr<const Network> & network,
        const uint32_t & run_length)
    : network_(network), runs_(network->runs_of_len_cumu(run_length)),
      runs_by_node_(network->split_by_node(this->runs_)),
      num_nodes_(network->size()), run_length_(run_length),
      num_runs_(this->runs_.size()){

    this->offset_.resize(run_length);
    uint32_t curr_offset_val = 1;
    for (uint32_t i = 0; i < run_length; ++i) {
        this->offset_.at(i) = curr_offset_val;
        curr_offset_val += (1 << (2*(i+1))) - 1;
    }
    this->num_features_ = curr_offset_val;

}

NetworkRunFeatures::NetworkRunFeatures(const NetworkRunFeatures & other)
    : network_(other.network_->clone()), runs_(other.runs_),
      runs_by_node_(other.runs_by_node_), num_nodes_(other.num_nodes_),
      run_length_(other.run_length_), num_runs_(other.num_runs_),
      offset_(other.offset_), num_features_(other.num_features_) {
}

std::shared_ptr<Features> NetworkRunFeatures::clone() const {
    return std::shared_ptr<Features>(new NetworkRunFeatures(*this));
}


std::vector<double> NetworkRunFeatures::get_features(
        const boost::dynamic_bitset<> & inf_bits,
        const boost::dynamic_bitset<> & trt_bits) {
    std::vector<double> feat(this->num_features(), 0.0);
    feat.at(0) = 1.0; // intercept

    for (uint32_t i = 0; i < this->num_runs_; ++i) {
        const NetworkRun & nr = this->runs_.at(i);
        const uint32_t run_len = nr.nodes.size();

        boost::dynamic_bitset<> inf_mask(run_len);
        boost::dynamic_bitset<> trt_mask(run_len);
        for (uint32_t j = 0; j < run_len; j++) {
            if (inf_bits.test(nr.nodes.at(j))) {
                inf_mask.set(j);
            }

            if (trt_bits.test(nr.nodes.at(j))) {
                trt_mask.set(j);
            }
        }

        const uint32_t max_mask = 1 << run_len;
        if (!inf_mask.all() || !trt_mask.all()) {
            const uint32_t index = offset_.at(run_len-1) +
                inf_mask.to_ulong() * max_mask +
                trt_mask.to_ulong();

            feat.at(index) += 1.0;
        }
    }

    return feat;
}


void NetworkRunFeatures::update_features(
        const uint32_t & changed_node,
        const boost::dynamic_bitset<> & inf_bits_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const boost::dynamic_bitset<> & inf_bits_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) {

    const std::vector<NetworkRun> changed_runs(runs_by_node_.at(changed_node));
    const uint32_t num_changed = changed_runs.size();

    for (uint32_t i = 0; i < num_changed; ++i) {
        const NetworkRun & nr = changed_runs.at(i);
        const uint32_t run_len = nr.nodes.size();

        boost::dynamic_bitset<> inf_mask_new(run_len);
        boost::dynamic_bitset<> trt_mask_new(run_len);
        boost::dynamic_bitset<> inf_mask_old(run_len);
        boost::dynamic_bitset<> trt_mask_old(run_len);

        for (uint32_t j = 0; j < run_len; ++j) {
            const uint32_t & node = nr.nodes.at(j);
            if (inf_bits_new.test(node)) {
                inf_mask_new.set(j);
            }
            if (trt_bits_new.test(node)) {
                trt_mask_new.set(j);
            }
            if (inf_bits_old.test(node)) {
                inf_mask_old.set(j);
            }
            if (trt_bits_old.test(node)) {
                trt_mask_old.set(j);
            }
        }

        const uint32_t max_mask = 1 << run_len;
        if (!inf_mask_new.all() || !trt_mask_new.all()) {
            const uint32_t index = offset_.at(run_len - 1) +
                inf_mask_new.to_ulong() * max_mask +
                trt_mask_new.to_ulong();
            feat.at(index) += 1.0;
        }

        if (!inf_mask_old.all() || !trt_mask_old.all()) {
            const uint32_t index = offset_.at(run_len - 1) +
                inf_mask_old.to_ulong() * max_mask +
                trt_mask_old.to_ulong();
            feat.at(index) -= 1.0;
        }
    }
}


uint32_t NetworkRunFeatures::num_features() const {
    return this->num_features_;
}


} // namespace stdmMf
