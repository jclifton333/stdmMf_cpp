#include "system.hpp"

#include <math.h>

namespace stdmMf {


System::System(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model> & model)
    : network_(network), model_(model),
      num_nodes_(this->network_->size()), inf_bits_(this->num_nodes_),
      trt_bits_(this->num_nodes_), time_(0) {
}

System::System(const System & other)
    : network_(other.network_->clone()), model_(other.model_->clone()),
      num_nodes_(other.num_nodes_), inf_bits_(other.inf_bits_),
      trt_bits_(other.trt_bits_), history_(other.history_), time_(other.time_) {
}

std::shared_ptr<System> System::clone() const {
    return std::shared_ptr<System>(new System(*this));
}


uint32_t System::n_inf() const {
    return this->inf_bits_.count();
}

uint32_t System::n_not() const {
    return this->num_nodes_ - this->inf_bits_.count();
}

uint32_t System::n_trt() const {
    return this->trt_bits_.count();
}

void System::cleanse() {
    this->inf_bits_.reset();
}

void System::plague() {
    this->inf_bits_.set();
}

void System::wipe_trt() {
    this->trt_bits_.reset();
}

void System::erase_history() {
    this->history_.clear();
}

const boost::dynamic_bitset<> & System::inf_bits() const {
    return this->inf_bits_;
}

void System::inf_bits(const boost::dynamic_bitset<> & inf_bits) {
    this->inf_bits_ = inf_bits;
}

const boost::dynamic_bitset<> & System::trt_bits() const {
    return this->trt_bits_;
}

void System::trt_bits(const boost::dynamic_bitset<> & trt_bits) {
    this->trt_bits_ = trt_bits;
}

const std::vector<BitsetPair> & System::history() const {
    return this->history_;
}

void System::start() {
    this->cleanse();
    this->wipe_trt();
    this->erase_history();

    const uint32_t num_starts =
        static_cast<uint32_t>(ceil(this->num_nodes_ * 0.1));
    std::vector<int> infs = this->rng->sample_range(
            0, this->num_nodes_, num_starts);

    for (uint32_t i = 0; i < num_starts; ++i) {
        this->inf_bits_.set(infs.at(i));
    }
}

void System::update_history() {
    this->history_.push_back(
            BitsetPair(this->inf_bits_, this->trt_bits_));
}

void System::turn_clock() {
    const std::vector<double> probs = this->model_->probs(this->inf_bits_,
            this->trt_bits_);
    this->turn_clock(probs);
}

void System::turn_clock(const std::vector<double> & probs) {
    // first record the history
    this->update_history();

    // update infection status
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        double r = this->rng->runif_01();
        if (r < probs.at(i)) {
            this->inf_bits_.flip(i);
        }
    }

    // clear treatments
    this->wipe_trt();
}



} // namespace stdmMf
