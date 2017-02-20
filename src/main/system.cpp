#include "system.hpp"

#include <math.h>

namespace stdmMf {

template <typename State>
System<State>::System(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<State> > & model)
    : network_(network), model_(model),
      num_nodes_(this->network_->size()), state_(this->num_nodes_),
      trt_bits_(this->num_nodes_), time_(0) {
    // share rng
    model_->rng(this->rng());
}

template <typename State>
System<State>::System(const System<State> & other)
    : RngClass(other), network_(other.network_->clone()),
      model_(other.model_->clone()), num_nodes_(other.num_nodes_),
      state_(other.state_), trt_bits_(other.trt_bits_),
      history_(other.history_), time_(other.time_) {
    // share rng
    model_->rng(this->rng());
}

template <typename State>
std::shared_ptr<System<State> > System<State>::clone() const {
    return std::shared_ptr<System<State> >(new System<State>(*this));
}


template <typename State>
uint32_t System<State>::num_nodes() const {
    return this->num_nodes_;
}


template <typename State>
uint32_t System<State>::n_inf() const {
    return this->state_.inf_bits.count();
}

template <typename State>
uint32_t System<State>::n_not() const {
    return this->num_nodes_ - this->state_.inf_bits.count();
}

template <typename State>
uint32_t System<State>::n_trt() const {
    return this->trt_bits_.count();
}

template <typename State>
void System<State>::reset() {
    // reset state
    this->state_.reset();
    // wipe treatments
    this->trt_bits_.reset();
    // erase history
    this->history_.clear();
}

template <typename State>
const State & System<State>::state() const {
    return this->state_;
}

template <typename State>
void System<State>::state(const State & state) {
    this->state_ = state;
}

template <typename State>
const boost::dynamic_bitset<> & System<State>::trt_bits() const {
    return this->trt_bits_;
}

template <typename State>
void System<State>::trt_bits(const boost::dynamic_bitset<> & trt_bits) {
    this->trt_bits_ = trt_bits;
}

template <typename State>
const std::vector<StateAndTrt<State> > & System<State>::history() const {
    return this->history_;
}

template <typename State>
void System<State>::start() {
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

template <typename State>
void System<State>::update_history() {
    this->history_.push_back(
            StateAndTrt<State>(this->state_, this->trt_bits_));
}

template <typename State>
void System<State>::turn_clock() {
    const State next_state = this->model_->turn_clock(this->state_,
            this->trt_bits_);

    this->turn_clock(next_state);
}

template <typename State>
void System<State>::turn_clock(const State & next_state) {
    // first record the history
    this->update_history();

    // assign next state
    this->state_ = next_state;

    // clear treatments
    this->trt_bits_.reset();
}


template class System<InfState>;
template class System<InfShieldState>;


} // namespace stdmMf
