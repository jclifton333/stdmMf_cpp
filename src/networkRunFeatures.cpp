#include "networkRunFeatures.hpp"
#include <glog/logging.h>

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

    // count runs
    this->num_runs_by_len_.resize(this->run_length_,0);
    for (uint32_t i = 0; i < this->num_runs_; ++i) {
        const uint32_t run_len  = this->runs_.at(i).nodes.size();
        ++this->num_runs_by_len_.at(run_len - 1);
    }

    // set up path trt mask
    this->masks_by_node_.resize(this->network_->size());
    for (uint32_t i = 0; i < this->num_runs_; ++i) {
        const NetworkRun & nr = this->runs_.at(i);
        const uint32_t run_len = nr.nodes.size();
        CHECK_LT(run_len, 32);

        std::pair<uint32_t, uint32_t> * const
            mask(new std::pair<uint32_t, uint32_t>(0, 0));

        this->masks_.push_back(mask);

        // filter by node
        for (uint32_t j = 0; j < run_len; ++j) {
            this->masks_by_node_.at(nr.nodes.at(j)).push_back(mask);
        }
    }
}

NetworkRunFeatures::NetworkRunFeatures(const NetworkRunFeatures & other)
    : network_(other.network_->clone()), runs_(other.runs_),
      runs_by_node_(other.runs_by_node_), num_nodes_(other.num_nodes_),
      run_length_(other.run_length_), num_runs_(other.num_runs_),
      offset_(other.offset_), num_features_(other.num_features_) {
}

NetworkRunFeatures::~NetworkRunFeatures() {
    // Only need to delete masks. Masks by node are referencing the
    // same memory.
    std::for_each(masks_.begin(), masks_.end(),
            [](std::pair<uint32_t, uint32_t> * const p) {
                delete p;
            });
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
        const uint32_t & run_len = nr.len;

        uint32_t & inf_mask(this->masks_.at(i)->first);
        uint32_t & trt_mask(this->masks_.at(i)->second);

        inf_mask = 0;
        trt_mask = 0;
        for (uint32_t j = 0; j < run_len; j++) {
            if (inf_bits.test(nr.nodes.at(j))) {
                inf_mask |= (1 << j);
            }

            if (trt_bits.test(nr.nodes.at(j))) {
                trt_mask |= (1 << j);
            }
        }

        const uint32_t max_mask = 1 << run_len;
        if (inf_mask < (max_mask - 1) || trt_mask < (max_mask - 1)) {
            const uint32_t index = offset_.at(run_len-1) +
                inf_mask * max_mask +
                trt_mask;

            feat.at(index) += 1.0 / this->num_runs_by_len_.at(run_len - 1);
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

    const std::vector<NetworkRun> & changed_runs(
            runs_by_node_.at(changed_node));
    const uint32_t num_changed = changed_runs.size();

    const std::vector<std::pair<uint32_t, uint32_t> *> &
        changed_masks = this->masks_by_node_.at(changed_node);

    const bool inf_changed =
        inf_bits_new.test(changed_node) != inf_bits_old.test(changed_node);
    const bool trt_changed =
        trt_bits_new.test(changed_node) != trt_bits_old.test(changed_node);

    for (uint32_t i = 0; i < num_changed; ++i) {
        const NetworkRun & nr = changed_runs.at(i);
        const uint32_t & run_len = nr.len;
        std::pair<uint32_t, uint32_t> & cm = *changed_masks.at(i);

        const uint32_t max_mask = 1 << run_len;

        // update features for old masks
        if (cm.first < (max_mask - 1) || cm.second < (max_mask - 1)) {
            const uint32_t index = offset_.at(run_len - 1) +
                cm.first * max_mask +
                cm.second;
            feat.at(index) -= 1.0 / this->num_runs_by_len_.at(run_len - 1);
        }

        // update masks
        for (uint32_t j = 0; j < run_len; ++j) {
            const uint32_t & node = nr.nodes.at(j);
            if (node == changed_node) {
                if (inf_changed) {
                    cm.first ^= (1 << j);
                }
                if (trt_changed) {
                    cm.second ^= (1 << j);
                }
                break;
            }
        }

        // update features for new masks
        if (cm.first < (max_mask - 1) || cm.second < (max_mask - 1)) {
            const uint32_t index = offset_.at(run_len - 1) +
                cm.first * max_mask +
                cm.second;
            feat.at(index) += 1.0 / this->num_runs_by_len_.at(run_len - 1);
        }

    }
}


void NetworkRunFeatures::update_features_async(
        const uint32_t & changed_node,
        const boost::dynamic_bitset<> & inf_bits_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const boost::dynamic_bitset<> & inf_bits_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) const {

    const std::vector<NetworkRun> & changed_runs(
            runs_by_node_.at(changed_node));
    const uint32_t num_changed = changed_runs.size();

    for (uint32_t i = 0; i < num_changed; ++i) {
        const NetworkRun & nr = changed_runs.at(i);
        const uint32_t & run_len = nr.len;

        CHECK_LE(run_len, 32);
        uint32_t inf_mask_new = 0;
        uint32_t inf_mask_old = 0;
        uint32_t trt_mask_new = 0;
        uint32_t trt_mask_old = 0;

        for (uint32_t j = 0; j < run_len; ++j) {
            const uint32_t & node = nr.nodes.at(j);
            if (inf_bits_new.test(node)) {
                inf_mask_new |= (1 << j);
            }
            if (trt_bits_new.test(node)) {
                trt_mask_new |= (1 << j);
            }
            if (inf_bits_old.test(node)) {
                inf_mask_old |= (1 << j);
            }
            if (trt_bits_old.test(node)) {
                trt_mask_old |= (1 << j);
            }
        }

        const uint32_t max_mask = 1 << run_len;
        if (inf_mask_new < (max_mask - 1) || trt_mask_new < (max_mask - 1)) {
            const uint32_t index = offset_.at(run_len - 1) +
                inf_mask_new * max_mask +
                trt_mask_new;
            feat.at(index) += 1.0 / this->num_runs_by_len_.at(run_len - 1);
        }

        if (inf_mask_old < (max_mask - 1) || trt_mask_old < (max_mask - 1)) {
            const uint32_t index = offset_.at(run_len - 1) +
                inf_mask_old * max_mask +
                trt_mask_old;
            feat.at(index) -= 1.0 / this->num_runs_by_len_.at(run_len - 1);
        }
    }
}


uint32_t NetworkRunFeatures::num_features() const {
    return this->num_features_;
}


} // namespace stdmMf
