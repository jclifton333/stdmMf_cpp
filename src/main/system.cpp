#include "system.hpp"

#include <math.h>

namespace stdmMf {

template <typename State>
System::System(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<State> > & model)
    : network_(network), model_(model),
      num_nodes_(this->network_->size()), inf_bits_(this->num_nodes_),
      trt_bits_(this->num_nodes_), time_(0) {
}

template <typename State>
System::System(const System<State> & other)
    : RngClass(other), network_(other.network_->clone()),
      model_(other.model_->clone()), num_nodes_(other.num_nodes_),
      inf_bits_(other.inf_bits_), trt_bits_(other.trt_bits_),
      history_(other.history_), time_(other.time_) {
}

template <typename State>
std::shared_ptr<System<State> > System::clone() const {
    return std::shared_ptr<System<State> >(new System<State>(*this));
}


template <typename State>
uint32_t System::n_inf() const {
    return this->state_.inf_bits.count();
}

template <typename State>
uint32_t System::n_not() const {
    return this->num_nodes_ - this->state_.inf_bits.count();
}

template <typename State>
uint32_t System::n_trt() const {
    return this->trt_bits_.count();
}

template <typename State>
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

template <typename State>
const State & System::state() const {
    return this->state_;
}

template <typename State>
void System::state(const State & state) {
    this->state_ = state;
}

template <typename State>
const boost::dynamic_bitset<> & System::trt_bits() const {
    return this->trt_bits_;
}

template <typename State>
void System::trt_bits(const boost::dynamic_bitset<> & trt_bits) {
    this->trt_bits_ = trt_bits;
}

template <typename State>
const std::vector<InfAndTrt<State> > & System::history() const {
    return this->history_;
}

template <typename State>
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

template <typename State>
void System::update_history() {
    this->history_.push_back(
            StateAndTrt<State>(this->state_, this->trt_bits_));
}

template <typename State>
void System::turn_clock() {
    const State next_state = this->model_->(turn_clock(this->state_,
                    this->inf_bits));

    this->turn_clock(next_state);
}

template <typename State>
void System::turn_clock(const State & next_state) {
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
