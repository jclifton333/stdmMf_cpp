#include "system.hpp"

#include <math.h>

namespace stdmMf {


System::System(std::shared_ptr<const Network> network,
        std::shared_ptr<Model> model)
    : RngClass(), network_(network), model_(model), num_nodes_(network->size()),
      inf_status_(num_nodes_), trt_status_(num_nodes_), time_(0) {
}


uint32_t System::n_inf() const {
    return this->inf_status_.count();
}

uint32_t System::n_not() const {
    return this->num_nodes_ - this->inf_status_.count();
}

uint32_t System::n_trt() const {
    return this->trt_status_.count();
}

std::vector<uint32_t> System::inf_nodes() const {
    std::vector<uint32_t> inds;
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        if (this->inf_status_.test(i)) {
            inds.push_back(i);
        }
    }

    return inds;
}

std::vector<uint32_t> System::not_nodes() const {
    std::vector<uint32_t> inds;
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        if (!this->inf_status_.test(i)) {
            inds.push_back(i);
        }
    }

    return inds;
}

std::vector<uint32_t> System::status() const {
    std::vector<uint32_t> status;
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        uint32_t status_i = 0;

        if (this->inf_status_.test(i)) {
            status_i = 2;
        }

        if (this->trt_status_.test(i)) {
            status_i++;
        }

        status.push_back(status_i);
    }

    return status;
}

void System::cleanse() {
    this->inf_status_.reset();
}

void System::plague() {
    this->inf_status_.set();
}

void System::wipe_trt() {
    this->trt_status_.reset();
}

void System::erase_history() {
    this->history_.clear();
}

boost::dynamic_bitset<> System::inf_status() const {
    return this->inf_status_;
}

boost::dynamic_bitset<> System::trt_status() const {
    return this->trt_status_;
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
        this->inf_status_.set(infs.at(i));
    }
}

void System::update_history() {
    this->history_.push_back(
            inf_trt_pair(this->inf_status_, this->trt_status_));
}

void System::turn_clock() {
    // TODO: need a defined model first
    std::vector<double> probs;

    this->turn_clock(probs);
}

void System::turn_clock(const std::vector<double> & probs) {
    // first record the history
    this->update_history();

    // update infection status
    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        double r = this->rng->runif_01();
        if (r < probs.at(i)) {
            this->inf_status_.flip(i);
        }
    }

    // clear treatments
    this->wipe_trt();
}



} // namespace stdmMf
