#include <glog/logging.h>
#include <set>
#include <limits>
#include "sweepAgentSlow.hpp"
#include "utilities.hpp"

namespace stdmMf {


SweepAgentSlow::SweepAgentSlow(const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Features> & features,
        const std::vector<double> & coef,
        const uint32_t & max_sweeps)
    : Agent(network), features_(features), max_sweeps_(max_sweeps),
      coef_(coef) {
    CHECK_EQ(this->coef_.size(), this->features_->num_features());
}

SweepAgentSlow::SweepAgentSlow(const SweepAgentSlow & other)
    : Agent(other), features_(other.features_->clone()),
      max_sweeps_(other.max_sweeps_), coef_(other.coef_) {
}

std::shared_ptr<Agent> SweepAgentSlow::clone() const{
    return std::shared_ptr<Agent>(new SweepAgentSlow(*this));
}

boost::dynamic_bitset<> SweepAgentSlow::apply_trt(
        const boost::dynamic_bitset<> & inf_bits,
        const std::vector<InfAndTrt> & history) {
    return this->apply_trt(inf_bits);
}


boost::dynamic_bitset<> SweepAgentSlow::apply_trt(
        const boost::dynamic_bitset<> & inf_bits) {
    boost::dynamic_bitset<> trt_bits(this->num_nodes_);

    // sets of treated and not treated
    std::set<uint32_t> not_trt;
    std::set<uint32_t> has_trt;

    for (uint32_t i = 0; i < this->num_nodes_; ++i) {
        not_trt.insert(i);
    }

    // initialize first treatment bits
    for (uint32_t i = 0; i < this->num_trt_; ++i) {
        this->set_new_treatment(trt_bits, not_trt, has_trt, inf_bits);
    }

    std::vector<double> f = this->features_->get_features(inf_bits, trt_bits);
    double best_val = dot_a_and_b(this->coef_, f);

    // sweep treatments
    if (this->max_sweeps_ > 0) {
        for (uint32_t i = 0; i < this->max_sweeps_; ++i) {
            const bool changed = this->sweep_treatments(trt_bits, best_val,
                    not_trt, has_trt, inf_bits);

            if (!changed)
                break;
        }
    } else {
        bool changed = true;
        uint32_t i = 0;
        while (changed) {
            changed = this->sweep_treatments(trt_bits, best_val,
                    not_trt, has_trt, inf_bits);
        }
    }

    CHECK_EQ(trt_bits.count(), this->num_trt_);

    return trt_bits;
}

void SweepAgentSlow::set_new_treatment(
        boost::dynamic_bitset<> & trt_bits,
        std::set<uint32_t> & not_trt,
        std::set<uint32_t> & has_trt,
        const boost::dynamic_bitset<> & inf_bits) const {

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
        const uint32_t best_node = best_nodes.at(0);
        trt_bits.set(best_node);
        // update sets
        not_trt.erase(best_node);
        has_trt.insert(best_node);
    } else {
        // multiple best nodes
        const uint32_t index = this->rng->rint(0, best_nodes.size());
        const uint32_t best_node = best_nodes.at(index);
        trt_bits.set(best_node);
        // update sets
        not_trt.erase(best_node);
        has_trt.insert(best_node);
    }
}


bool SweepAgentSlow::sweep_treatments(
        boost::dynamic_bitset<> & trt_bits,
        double & best_val,
        std::set<uint32_t> & not_trt,
        std::set<uint32_t> & has_trt,
        const boost::dynamic_bitset<> & inf_bits) const {


    std::set<uint32_t>::const_iterator has_it, not_it, has_end, not_end;
    has_end = has_trt.end();

    bool changed = false;

    std::set<uint32_t> new_not_trt;
    std::set<uint32_t> new_has_trt;

    // loop over all current treatments
    for (has_it = has_trt.begin(); has_it != has_end; ++has_it) {
        CHECK_EQ(has_trt.size(), this->num_trt());
        CHECK(trt_bits.test(*has_it)) << "bit is not set";

        trt_bits.reset(*has_it); // reset

        std::vector<uint32_t> better_nodes;
        better_nodes.push_back(*has_it);

        // see if any non-treated are better
        not_end = not_trt.end();

        bool can_tie_orig = true;

        for (not_it = not_trt.begin(); not_it != not_end; ++not_it) {
            CHECK(!trt_bits.test(*not_it)) << "bit is already set";

            trt_bits.set(*not_it);

            const std::vector<double> f = this->features_->get_features(
                    inf_bits, trt_bits);

            const double val = dot_a_and_b(this->coef_, f);

            trt_bits.reset(*not_it);

            if (val > best_val) {
                can_tie_orig = false;
                best_val = val;
                better_nodes.clear();
                better_nodes.push_back(*not_it);
            } else if (val == best_val) {
                better_nodes.push_back(*not_it);
            }
        }

        const uint32_t num_better = better_nodes.size();
        if (num_better == 1 && can_tie_orig) {
            // original node was best
            CHECK(!trt_bits.test(*has_it));
            trt_bits.set(*has_it);
        } else if (num_better == 1) {
            // unique better node
            const uint32_t better_node = better_nodes.at(0);
            CHECK(!trt_bits.test(better_node));
            trt_bits.set(better_node);
            not_trt.erase(better_node);
            changed = !can_tie_orig; // only has changed if value improved
            // add *has_it to set of not_treated
            not_trt.insert(*has_it);
            // records for updating has_trt;
            new_has_trt.insert(better_node);
            new_not_trt.insert(*has_it);
        } else {
            // multiple better nodes
            const uint32_t index = this->rng->rint(0, num_better);
            const uint32_t better_node = better_nodes.at(index);
            CHECK(!trt_bits.test(better_node));
            trt_bits.set(better_node);
            if (better_node != *has_it) { // if it is the original
                not_trt.erase(better_node);
                changed = !can_tie_orig; // only has changed if value improved
                // add *has_it to set of not_treated
                not_trt.insert(*has_it);
                // records for updating has_trt;
                new_has_trt.insert(better_node);
                new_not_trt.insert(*has_it);
            }
        }
    }

    // add new_trt to has_trt
    std::for_each(new_not_trt.begin(), new_not_trt.end(),
            [&has_trt] (const uint32_t & x) {
                has_trt.erase(x);
            });
    has_trt.insert(new_has_trt.begin(), new_has_trt.end());

    return changed;
}






} // namespace stdmMf
