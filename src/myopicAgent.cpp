#include "myopicAgent.hpp"
#include <glog/logging.h>

#include "proximalAgent.hpp"

namespace stdmMf {


MyopicAgent::MyopicAgent(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model> & model)
    : Agent(network), model_(model) {
}

MyopicAgent::MyopicAgent(const MyopicAgent & other)
    : Agent(other), model_(other.model_->clone()) {
}

std::shared_ptr<Agent> MyopicAgent::clone() const {
    return std::shared_ptr<Agent>(new MyopicAgent(*this));
}

boost::dynamic_bitset<> MyopicAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits) {
    LOG(FATAL) << "Needs history to apply treatment.";
}



boost::dynamic_bitset<> MyopicAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    boost::dynamic_bitset<> trt_bits(this->network_->size());
    if (history.size() > 0) {
        // add current infections to history for model fitting
        std::vector<BitsetPair> all_history = history;
        all_history.push_back(BitsetPair(inf_bits, trt_bits));

        // get probabilities
        this->model_->est_par(all_history);
        const std::vector<double> probs = this->model_->probs(inf_bits,
                trt_bits);

        // set up sorting
        std::vector<std::pair<double, uint32_t> > sorted;
        for (uint32_t i = 0; i < this->num_nodes_; ++i) {
            // negative probability because of ascending order
            sorted.push_back(std::pair<double, uint32_t>(- probs.at(i), i));
        }

        std::sort(sorted.begin(), sorted.end());

        trt_bits.reset();
        for (uint32_t i = 0; i < this->num_trt_; ++i) {
            trt_bits.set(sorted.at(i).second);
        }
    } else {
        // not enough data to estimate a model
        ProximalAgent pa(this->network_);
        trt_bits = pa.apply_trt(inf_bits, history);
    }
    return trt_bits;
}



} // namespace stdmMf
