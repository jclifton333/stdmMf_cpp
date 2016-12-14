#include <glog/logging.h>
#include <set>
#include <limits>
#include "stepAgent.hpp"
#include "utilities.hpp"

namespace stdmMf {


StepAgent::StepAgent(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features> & features,
        const std::vector<double> & coef,
        const uint32_t & max_sweeps)
    : Agent(network), features_(features), max_sweeps_(max_sweeps),
      coef_(coef) {
    CHECK_EQ(this->coef_.size(), this->features_->num_features());
}


boost::dynamic_bitset<> StepAgent::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<BitsetPair> & history) {
    boost::dynamic_bitset<> trt_bits(this->num_nodes_);

    // sets of treated and not treated
    std::set<uint32_t> not_trt;
    std::set<uint32_t> has_trt;

    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        not_trt.insert(i);
    }

    std::vector<double> f(this->features_->get_features(inf_bits, trt_bits));

    double best_val = dot_a_and_b(this->coef_, f);

    // initialize first treatment bits
    for (uint32_t i = 0; i < this->num_trt_; ++i) {

    }
}

void StepAgent::set_new_treatment(
        boost::dynamic_bitset<> & trt_bits,
        const boost::dynamic_bitset<> & inf_bits,
        const std::set<uint32_t> & not_trt) const {

    std::set<uint32_t>::const_iterator it, end;
    end = not_trt.end();

    double best_val = std::numeric_limits<double>::lowest();
    std::vector<uint32_t> best_nodes;
    for (it = not_trt.begin(); it != end; ++it) {
        CHECK(!trt_bits.test(*it)) << "bit is already set";
        trt_bits.set(*it); // set new bit

        const std::vector<double> f = this->features_->get_features(inf_bits,
                trt_bits);

        const double val = dot_a_and_b(this->coef_, f);

        trt_bits.reset(*it); // reset new bit

        if (val > best_val) {
            best_val = val;
            best_nodes.clear();
            best_nodes.push_back(*it);
        } else if (val == best_val) {
            best_nodes.push_back(*it);
        }
    }

    CHECK_GT(best_nodes.size(), 0);
    if (best_nodes.size() == 1) {
        // unique best node
        trt_bits.set(best_nodes.at(0));
    } else {
        // multiple best nodes
        const uint32_t index = this->rng->rint(0, best_nodes.size());
        trt_bits.set(best_nodes.at(index));
    }
}


bool StepAgent::sweep_treatments(
        boost::dynamic_bitset<> & trt_bits,
        double & best_val,
        std::set<uint32_t> & not_trt,
        std::set<uint32_t> & has_trt,
        const boost::dynamic_bitset<> & inf_bits) const {
    std::set<uint32_t>::const_iterator has_it, not_it, has_end, not_end;
    has_end = has_trt.end();

    bool changed = false;

    std::set<uint32_t> new_trt;

    // loop over all current treatments
    for (has_it = has_trt.begin(); has_it != has_end; ++has_it) {
        CHECK(trt_bits.test(*has_it)) << "bit is not set";

        trt_bits.reset(*has_it); // reset

        std::vector<uint32_t> better_nodes;

        // see if any non-treated are better
        not_end = not_trt.end();
        for (not_it = not_trt.begin(); not_it != not_end; ++not_it) {
            CHECK(!trt_bits.test(*not_it)) << "bit is already set";

            trt_bits.set(*not_it);

            const std::vector<double> f = this->features_->get_features(
                    inf_bits, trt_bits);

            const double val = dot_a_and_b(this->coef_, f);

            trt_bits.reset(*not_it);

            if (val > best_val) {
                best_val = val;
                better_nodes.clear();
                better_nodes.push_back(*not_it);
            } else if (val == best_val) {
                better_nodes.push_back(*not_it);
            }
        }

        const uint32_t num_better = better_nodes.size();
        if (num_better == 0) {
            // original node was best
            trt_bits.set(*has_it);
        } else if (num_better == 1) {
            // unique better node
            const uint32_t better_node = better_nodes.at(0);
            trt_bits.set(better_node);
            new_trt.insert(better_node);
            not_trt.erase(better_node);
            changed = true;
            // add *has_it to set of not_treated
            not_trt.insert(*has_it);
        } else {
            // multiple better nodes
            const uint32_t index = this->rng->rint(0, num_better);
            const uint32_t better_node = better_nodes.at(index);
            trt_bits.set(better_node);
            new_trt.insert(better_node);
            not_trt.erase(better_node);
            changed = true;
            // add *has_it to set of not_treated
            not_trt.insert(*has_it);
        }
    }

    // add new_trt to has_trt
    has_trt.insert(new_trt.begin(), new_trt.end());

    return changed;
}






} // namespace stdmMf
