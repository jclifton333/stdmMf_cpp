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
    : RngClass(other), network_(other.network_->clone()),
      model_(other.model_->clone()), num_nodes_(other.num_nodes_),
      inf_bits_(other.inf_bits_), trt_bits_(other.trt_bits_),
      history_(other.history_), time_(other.time_) {
}

std::shared_ptr<System> System::clone() const {
    return std::shared_ptr<System>(new System(*this));
}


uint32_t System::n_inf() const {
    return this->state_.inf_bits.count();
}

uint32_t System::n_not() const {
    return this->num_nodes_ - this->state_.inf_bits.count();
}

uint32_t System::n_trt() const {
    return this->trt_bits_.count();
}

void System::reset() {
    // wipe infection
    this->state_.inf_bits.reset();
    // set shield to zero
    std::fill(this->state_.shield.begin(),
            this->state_.shield.end(), 0.);
    // wipe treatments
    this->trt_bits_.reset();
    // erase history
    this->erase_history();
}

const State & System::state() const {
    return this->state_;
}

void System::state(const State & state) {
    this->state_ = state;
}

const boost::dynamic_bitset<> & System::trt_bits() const {
    return this->trt_bits_;
}

void System::trt_bits(const boost::dynamic_bitset<> & trt_bits) {
    this->trt_bits_ = trt_bits;
}

const std::vector<InfAndTrt> & System::history() const {
    return this->history_;
}

void System::start() {
    this->reset();

    // randomly select locations for infection
    const uint32_t num_starts =
        static_cast<uint32_t>(ceil(this->num_nodes_ * 0.1));
    const std::vector<int> infs = this->rng_->sample_range(
            0, this->num_nodes_, num_starts);

    for (uint32_t i = 0; i < num_starts; ++i) {
        this->state_.inf_bits.set(infs.at(i));
    }
}

void System::update_history() {
    this->history_.push_back(
            StateAndTrt(this->state_, this->trt_bits_));
}

void System::turn_clock() {
    const State next_state = this->model_->(turn_clock(this->state_,
                    this->inf_bits));

    this->turn_clock(next_state);
}

void System::turn_clock(const State & next_state) {
    // first record the history
    this->update_history();

    // assign next state
    this->state_ = next_state;

    // clear treatments
    this->trt_bits_.reset();
}



} // namespace stdmMf
