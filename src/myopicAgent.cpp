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
    if (history.size() < 1) {
        // not enough data to estimate a model
        ProximalAgent pa(this->network_);
        trt_bits = pa.apply_trt(inf_bits, history);
    } else {
        // get probabilities
        this->model_->est_par(inf_bits, history);
        const std::vector<double> probs = this->model_->probs(inf_bits,
                trt_bits);

        // set up sorting
        std::vector<uint32_t> shuffled_nodes;
        for (uint32_t i = 0; i < this->num_nodes_; ++i) {
            shuffled_nodes.push_back(i);
        }
        this->rng->shuffle(shuffled_nodes);

        std::vector<std::pair<double, uint32_t> > sorted_inf;
        std::vector<std::pair<double, uint32_t> > sorted_not;
        for (uint32_t i = 0; i < this->num_nodes_; ++i) {
            const uint32_t & node = shuffled_nodes.at(i);
            if (inf_bits.test(node)) {
                // use raw probability.  sorting is ascending.  want
                // to treat infected node with smallest probability of
                // recovery
                sorted_inf.push_back(std::pair<double, uint32_t>(
                                probs.at(node), node));
            } else {
                // add negative since sorting is ascending order. want
                // to treat uninfected node with largest probability
                // of infection
                sorted_not.push_back(std::pair<double, uint32_t>(
                                -probs.at(node), node));
            }
        }

        std::sort(sorted_inf.begin(), sorted_inf.end());
        std::sort(sorted_not.begin(), sorted_not.end());

        // uint32_t num_trt_not = this->num_trt_ / 2 + 1;
        // uint32_t num_trt_inf = this->num_trt_ - num_trt_not;

        uint32_t num_trt_inf = std::min(this->num_trt_,
                static_cast<uint32_t>(inf_bits.count()));
        uint32_t num_trt_not = std::max(static_cast<uint32_t>(
                        this->num_trt_ - inf_bits.count()), uint32_t(0));

        if (num_trt_not > (this->num_nodes_ - inf_bits.count())) {
            const uint32_t diff = num_trt_not -
                (this->num_nodes_ - inf_bits.count());
            num_trt_not -= diff;
            num_trt_inf += diff;
        } else if (num_trt_inf > inf_bits.count()) {
            const uint32_t diff = num_trt_inf - inf_bits.count();
            num_trt_inf -= diff;
            num_trt_not += diff;
        }

        CHECK_LE(num_trt_not, this->num_nodes_ - inf_bits.count());
        CHECK_LE(num_trt_inf, inf_bits.count());
        CHECK_EQ(num_trt_not + num_trt_inf, this->num_trt_);

        trt_bits.reset();
        for (uint32_t i = 0; i < num_trt_not; ++i) {
            trt_bits.set(sorted_not.at(i).second);
        }
        for (uint32_t i = 0; i < num_trt_inf; ++i) {
            trt_bits.set(sorted_inf.at(i).second);
        }
    }
    return trt_bits;
}



} // namespace stdmMf
