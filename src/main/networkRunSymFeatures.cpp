#include "networkRunSymFeatures.hpp"
#include "states.hpp"
#include <njm_cpp/tools/bitManip.hpp>
#include <glog/logging.h>

namespace stdmMf {


template <typename State>
NetworkRunSymFeatures<State>::NetworkRunSymFeatures(
        const std::shared_ptr<const Network> & network,
        const uint32_t & run_length)
    : network_(network), runs_(network->runs_of_len_cumu(run_length)),
      runs_by_node_(network->split_by_node(this->runs_)),
      num_nodes_(network->size()), run_length_(run_length),
      num_runs_(this->runs_.size()){

    const uint32_t tot_bits = sizeof(uint32_t(0)) * 8;
    CHECK_LE(this->bits_per_node_ * run_length_ + 1, tot_bits)
        << "Number of needed bits is "
        << (this->bits_per_node_ * run_length + 1)
        << " but number of available bits is " << tot_bits;

    uint32_t index_val = 1;
    for (uint32_t i = 0; i < this->run_length_; ++i) {
        std::vector<uint32_t> index_len_ip1;
        const uint32_t current_len = i + 1;
        const uint32_t num_bits = current_len * this->bits_per_node_;
        const uint32_t max_mask = 1 << num_bits;
        for (uint32_t mask = 0; mask < (max_mask - 1); ++mask) {
            // reverse each groups of bits then combine back together
            // e.g., reverse treatment bits, reverse infection bits,
            // reverse shielding bits
            uint32_t mask_rev = 0;
            for (uint32_t bit_group = 0; bit_group < this->bits_per_node_;
                 ++bit_group) {
                mask_rev |= njm::tools::reverse_bits(
                        mask >> (bit_group * current_len))
                    >> (tot_bits - (bit_group + 1) * current_len);
            }

            if (mask <= mask_rev) {
                index_len_ip1.push_back(index_val++);
            } else {
                index_len_ip1.push_back(index_len_ip1.at(mask_rev));
            }
        }

        this->index_by_len_.push_back(index_len_ip1);
    }
    this->num_features_ = index_val;

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
        CHECK_LE(run_len, this->run_length_);

        uint32_t * const mask(new uint32_t(0));

        this->masks_.push_back(mask);

        // filter by node
        for (uint32_t j = 0; j < run_len; ++j) {
            this->masks_by_node_.at(nr.nodes.at(j)).push_back(mask);
        }
    }
}


template <typename State>
uint32_t NetworkRunSymFeatures<State>::num_features() const {
    return this->num_features_;
}



// BEGIN: Implementation for InfState

template <>
const uint32_t NetworkRunSymFeatures<InfState>::bits_per_node_ = 2;

template<>
NetworkRunSymFeatures<InfState>::NetworkRunSymFeatures(
        const NetworkRunSymFeatures<InfState> & other)
    : network_(other.network_->clone()), runs_(other.runs_),
      runs_by_node_(other.runs_by_node_), num_nodes_(other.num_nodes_),
      run_length_(other.run_length_), num_runs_(other.num_runs_),
      index_by_len_(other.index_by_len_), num_features_(other.num_features_),
      num_runs_by_len_(other.num_runs_by_len_) {

    // set up path trt mask
    this->masks_by_node_.resize(this->network_->size());
    for (uint32_t i = 0; i < this->num_runs_; ++i) {
        const NetworkRun & nr = this->runs_.at(i);
        const uint32_t run_len = nr.nodes.size();
        CHECK_LE(run_len, this->run_length_);

        uint32_t * const mask(new uint32_t(*other.masks_.at(i)));

        this->masks_.push_back(mask);

        // filter by node
        for (uint32_t j = 0; j < run_len; ++j) {
            this->masks_by_node_.at(nr.nodes.at(j)).push_back(mask);
        }
    }
}


template<>
NetworkRunSymFeatures<InfState>::~NetworkRunSymFeatures() {
    // Only need to delete masks. Masks by node are referencing the
    // same memory.
    std::for_each(masks_.begin(), masks_.end(),
            [](uint32_t * const p) {
                delete p;
            });
}

template <>
std::shared_ptr<Features<InfState> >
NetworkRunSymFeatures<InfState>::clone() const {
    return std::shared_ptr<Features<InfState> >(
            new NetworkRunSymFeatures<InfState>(*this));
}


template <>
std::vector<double> NetworkRunSymFeatures<InfState>::get_features(
        const InfState & state,
        const boost::dynamic_bitset<> & trt_bits) {
    std::vector<double> feat(this->num_features(), 0.0);
    feat.at(0) = 1.0; // intercept

    for (uint32_t i = 0; i < this->num_runs_; ++i) {
        const NetworkRun & nr = this->runs_.at(i);
        const uint32_t & run_len = nr.len;

        uint32_t & mask(*this->masks_.at(i));

        mask = 0;
        for (uint32_t j = 0; j < run_len; j++) {
            if (state.inf_bits.test(nr.nodes.at(j))) {
                mask |= (1 << (j + run_len));
            }

            if (trt_bits.test(nr.nodes.at(j))) {
                mask |= (1 << j);
            }
        }

        const uint32_t max_mask = 1 << (run_len + run_len);
        if (mask < (max_mask - 1)) {
            feat.at(this->index_by_len_.at(run_len - 1).at(mask)) +=
                1.0 / this->num_runs_by_len_.at(run_len - 1);
        }
    }

    return feat;
}


template <>
void NetworkRunSymFeatures<InfState>::update_features(
        const uint32_t & changed_node,
        const InfState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const InfState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) {

    const std::vector<NetworkRun> & changed_runs(
            runs_by_node_.at(changed_node));
    const uint32_t num_changed = changed_runs.size();

    const std::vector<uint32_t *> &
        changed_masks = this->masks_by_node_.at(changed_node);

    const bool inf_changed =
        state_new.inf_bits.test(changed_node)
        != state_old.inf_bits.test(changed_node);
    const bool trt_changed =
        trt_bits_new.test(changed_node) != trt_bits_old.test(changed_node);

    for (uint32_t i = 0; i < num_changed; ++i) {
        const NetworkRun & nr = changed_runs.at(i);
        const uint32_t & run_len = nr.len;
        uint32_t & cm = *changed_masks.at(i);

        const uint32_t max_mask = 1 << (run_len + run_len);

        // update features for old masks
        if (cm < (max_mask - 1)) {
            feat.at(this->index_by_len_.at(run_len - 1).at(cm)) -=
                1.0 / this->num_runs_by_len_.at(run_len - 1);
        }

        // update masks
        for (uint32_t j = 0; j < run_len; ++j) {
            const uint32_t & node = nr.nodes.at(j);
            if (node == changed_node) {
                if (inf_changed) {
                    cm ^= (1 << (j + run_len));
                }
                if (trt_changed) {
                    cm ^= (1 << j);
                }
                break;
            }
        }

        // update features for new masks
        if (cm < (max_mask - 1)) {
            feat.at(this->index_by_len_.at(run_len - 1).at(cm)) +=
                1.0 / this->num_runs_by_len_.at(run_len - 1);
        }

    }
}


template<>
void NetworkRunSymFeatures<InfState>::update_features_async(
        const uint32_t & changed_node,
        const InfState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const InfState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) const {

    const std::vector<NetworkRun> & changed_runs(
            runs_by_node_.at(changed_node));
    const uint32_t num_changed = changed_runs.size();

    for (uint32_t i = 0; i < num_changed; ++i) {
        const NetworkRun & nr = changed_runs.at(i);
        const uint32_t & run_len = nr.len;

        CHECK_LE(run_len, 32);
        uint32_t mask_new = 0;
        uint32_t mask_old = 0;

        for (uint32_t j = 0; j < run_len; ++j) {
            const uint32_t & node = nr.nodes.at(j);
            if (state_new.inf_bits.test(node)) {
                mask_new |= (1 << (j + run_len));
            }
            if (trt_bits_new.test(node)) {
                mask_new |= (1 << j);
            }
            if (state_old.inf_bits.test(node)) {
                mask_old |= (1 << (j + run_len));
            }
            if (trt_bits_old.test(node)) {
                mask_old |= (1 << j);
            }
        }

        const uint32_t max_mask = 1 << (run_len + run_len);
        if (mask_new < (max_mask - 1)) {
            feat.at(this->index_by_len_.at(run_len - 1).at(mask_new)) +=
                1.0 / this->num_runs_by_len_.at(run_len - 1);
        }

        if (mask_old < (max_mask - 1)) {
            feat.at(this->index_by_len_.at(run_len - 1).at(mask_old)) -=
                1.0 / this->num_runs_by_len_.at(run_len - 1);
        }
    }
}



// END: Implementation for InfState





template class NetworkRunSymFeatures<InfState>;

} // namespace stdmMf
