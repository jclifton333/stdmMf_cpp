#include "system.hpp"

#include <math.h>

namespace stdmMf {


System::System(std::shared_ptr<const Network> network,
        std::shared_ptr<Model> model)
    : RngClass(), network(network), model(model), num_nodes(network->size()),
      inf_status(num_nodes), trt_status(num_nodes), time(0) {
}


uint32_t System::n_inf() const {
    return this->inf_status.count();
}

uint32_t System::n_not() const {
    return this->num_nodes - this->inf_status.count();
}

uint32_t System::n_trt() const {
    return this->trt_status.count();
}

std::vector<uint32_t> System::inf_nodes() const {
    std::vector<uint32_t> inds;
    for (uint32_t i = 0; i < this->num_nodes; ++i) {
        if (this->inf_status.test(i)) {
            inds.push_back(i);
        }
    }

    return inds;
}

std::vector<uint32_t> System::not_nodes() const {
    std::vector<uint32_t> inds;
    for (uint32_t i = 0; i < this->num_nodes; ++i) {
        if (!this->inf_status.test(i)) {
            inds.push_back(i);
        }
    }

    return inds;
}

std::vector<uint32_t> System::status() const {
    std::vector<uint32_t> node_status;
    for (uint32_t i = 0; i < this->num_nodes; ++i) {
        uint32_t status_i = 0;

        if (this->inf_status.test(i)) {
            status_i = 2;
        }

        if (this->trt_status.test(i)) {
            status_i++;
        }
    }

    return node_status;
}

void System::cleanse() {
    this->inf_status.reset();
}

void System::plague() {
    this->inf_status.set();
}

void System::wipe_trt() {
    this->trt_status.reset();
}

void System::erase_history() {
    this->history.clear();
}

boost::dynamic_bitset<> System::get_inf_status() const {
    return this->inf_status;
}

boost::dynamic_bitset<> System::get_trt_status() const {
    return this->trt_status;
}

void System::start() {
    this->cleanse();
    this->wipe_trt();
    this->erase_history();

    const uint32_t num_starts =
        static_cast<uint32_t>(ceil(this->num_nodes * 0.1));
    std::vector<int> infs = this->rng->sample_range(
            0, this->num_nodes, num_starts);

    for (uint32_t i = 0; i < num_starts; ++i) {
        this->inf_status.set(infs.at(i));
    }
}

void System::update_history() {
    this->history.push_back(std::pair<
            boost::dynamic_bitset<> ,boost::dynamic_bitset<> >(
                    this->inf_status, this->trt_status));
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
    for (uint32_t i = 0; i < this->num_nodes; ++i) {
        double r = this->rng->runif_01();
        if (r < probs.at(i)) {
            this->inf_status.flip(i);
        }
    }

    // clear treatments
    this->wipe_trt();
}



} // namespace stdmMf
