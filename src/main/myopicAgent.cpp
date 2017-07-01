#include "myopicAgent.hpp"
#include <glog/logging.h>

#include "states.hpp"

#include "proximalAgent.hpp"

namespace stdmMf {


template <typename State>
MyopicAgent<State>::MyopicAgent(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<State> > & model)
    : Agent<State>(network), model_(model) {
    // share rng
    this->model_->rng(this->rng());
}


template <typename State>
MyopicAgent<State>::MyopicAgent(const MyopicAgent<State> & other)
    : Agent<State>(other), model_(other.model_->clone()) {
    // share rng
    this->model_->rng(this->rng());
}


template <typename State>
std::shared_ptr<Agent<State> > MyopicAgent<State>::clone() const {
    return std::shared_ptr<Agent<State> >(new MyopicAgent<State>(*this));
}


template <typename State>
boost::dynamic_bitset<> MyopicAgent<State>::apply_trt(
        const State & curr_state,
        const std::vector<StateAndTrt<State> > & history) {
    boost::dynamic_bitset<> trt_bits(this->network_->size());
    if (history.size() < 1) {
        // not enough data to estimate a model
        ProximalAgent<State> pa(this->network_);
        pa.rng(this->rng());
        trt_bits = pa.apply_trt(curr_state, history);
    } else {
        // get probabilities
        this->model_->est_par(history, curr_state);
        const std::vector<double> probs = this->model_->probs(curr_state,
                trt_bits);

        // set up sorting
        std::vector<uint32_t> shuffled_nodes;
        for (uint32_t i = 0; i < this->num_nodes_; ++i) {
            shuffled_nodes.push_back(i);
        }
        this->rng_->shuffle(shuffled_nodes);

        std::vector<std::pair<double, uint32_t> > sorted_inf;
        std::vector<std::pair<double, uint32_t> > sorted_not;
        for (uint32_t i = 0; i < this->num_nodes_; ++i) {
            const uint32_t & node = shuffled_nodes.at(i);
            if (curr_state.inf_bits.test(node)) {
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

        uint32_t num_trt_not = this->num_trt_ / 2 + 1;
        uint32_t num_trt_inf = this->num_trt_ - num_trt_not;

        if (num_trt_not > (this->num_nodes_ - curr_state.inf_bits.count())) {
            const uint32_t diff = num_trt_not -
                (this->num_nodes_ - curr_state.inf_bits.count());
            num_trt_not -= diff;
            num_trt_inf += diff;
        } else if (num_trt_inf > curr_state.inf_bits.count()) {
            const uint32_t diff = num_trt_inf - curr_state.inf_bits.count();
            num_trt_inf -= diff;
            num_trt_not += diff;
        }

        CHECK_LE(num_trt_not, this->num_nodes_ - curr_state.inf_bits.count());
        CHECK_LE(num_trt_inf, curr_state.inf_bits.count());
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


template <>
boost::dynamic_bitset<> MyopicAgent<EbolaState>::apply_trt(
        const EbolaState & curr_state,
        const std::vector<StateAndTrt<EbolaState> > & history) {
    boost::dynamic_bitset<> trt_bits(this->network_->size());
    if (history.size() < 1) {
        // not enough data to estimate a model
        ProximalAgent<EbolaState> pa(this->network_);
        pa.rng(this->rng());
        trt_bits = pa.apply_trt(curr_state, history);
    } else {
        // get probabilities
        this->model_->est_par(history, curr_state);
        const std::vector<double> probs = this->model_->probs(curr_state,
                trt_bits);

        // set up sorting
        std::vector<uint32_t> shuffled_nodes;
        for (uint32_t i = 0; i < this->num_nodes_; ++i) {
            shuffled_nodes.push_back(i);
        }
        this->rng_->shuffle(shuffled_nodes);

        std::vector<std::pair<double, uint32_t> > sorted_inf;
        std::vector<std::pair<double, uint32_t> > sorted_not;
        for (uint32_t i = 0; i < this->num_nodes_; ++i) {
            const uint32_t & node = shuffled_nodes.at(i);
            if (curr_state.inf_bits.test(node)) {
                // use probability of other locations becoming
                // infected weighted by distance, want to treat large
                // values so add negative because its sorted in
                // ascending order
                double weight_prob(0.0);
                for (uint32_t j = 0; j < this->num_nodes_; ++j) {
                    if (j != node && !curr_state.inf_bits.test(j)) {
                        weight_prob += probs.at(j)
                            / this->network_->dist().at(node).at(j);
                    }
                }
                sorted_inf.push_back(std::pair<double, uint32_t>(
                                -weight_prob, node));
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

        uint32_t num_trt_not = this->num_trt_ / 2 + 1;
        uint32_t num_trt_inf = this->num_trt_ - num_trt_not;

        if (num_trt_not > (this->num_nodes_ - curr_state.inf_bits.count())) {
            const uint32_t diff = num_trt_not -
                (this->num_nodes_ - curr_state.inf_bits.count());
            num_trt_not -= diff;
            num_trt_inf += diff;
        } else if (num_trt_inf > curr_state.inf_bits.count()) {
            const uint32_t diff = num_trt_inf - curr_state.inf_bits.count();
            num_trt_inf -= diff;
            num_trt_not += diff;
        }

        CHECK_LE(num_trt_not, this->num_nodes_ - curr_state.inf_bits.count());
        CHECK_LE(num_trt_inf, curr_state.inf_bits.count());
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


template<typename State>
void MyopicAgent<State>::rng(const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
    this->model_->rng(rng);
}



template class MyopicAgent<InfState>;
template class MyopicAgent<InfShieldState>;
template class MyopicAgent<EbolaState>;

} // namespace stdmMf
