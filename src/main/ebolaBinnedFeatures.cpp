#include "ebolaBinnedFeatures.hpp"

#include "ebolaData.hpp"

#include <njm_cpp/tools/stats.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

#include <glog/logging.h>

namespace stdmMf {


const uint32_t EbolaBinnedFeatures::num_features_per_bin_ = 2;


EbolaBinnedFeatures::EbolaBinnedFeatures(
        const std::shared_ptr<const Network> & network,
        const uint32_t num_bins,
        const uint32_t num_neigh)
    : network_(network), num_bins_(num_bins), num_neigh_(num_neigh),
      bins_(this->get_bins()), neigh_(this->get_neigh()),
      terms_(this->get_terms()) {
}


EbolaBinnedFeatures::EbolaBinnedFeatures(const EbolaBinnedFeatures & other)
    : network_(other.network_), num_bins_(other.num_bins_),
      num_neigh_(other.num_neigh_), bins_(other.bins_),
      neigh_(other.neigh_), terms_(other.terms_) {
}


std::shared_ptr<Features<EbolaState> > EbolaBinnedFeatures::clone() const {
    return std::shared_ptr<Features<EbolaState> >(
            new EbolaBinnedFeatures(*this));
}


std::vector<double> EbolaBinnedFeatures::get_features(
        const EbolaState & state,
        const boost::dynamic_bitset<> & trt_bits) {
    std::vector<double> feat(this->num_features(), 0.0);
    feat.at(0) = 1.0;
    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        const std::vector<Term> & t(this->terms_.at(i));
        const uint32_t offset(state.inf_bits.test(i)
                + 2 * trt_bits.test(i));
        std::vector<Term>::const_iterator it,end(t.end());
        for (it = t.begin(); it != end; ++it) {
            feat.at(it->index + offset) += it->weight;
        }
    }
    return feat;
}


void EbolaBinnedFeatures::update_features(
        const uint32_t & changed_node,
        const EbolaState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const EbolaState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) {
    const std::vector<Term> & t(this->terms_.at(changed_node));
    const uint32_t offset_old(state_old.inf_bits.test(changed_node)
            + 2 * trt_bits_old.test(changed_node));
    const uint32_t offset_new(state_new.inf_bits.test(changed_node)
            + 2 * trt_bits_new.test(changed_node));
    std::vector<Term>::const_iterator it,end(t.end());
    for (it = t.begin(); it != end; ++it) {
        feat.at(it->index + offset_old) -= it->weight;
        feat.at(it->index + offset_new) += it->weight;
    }
}


void EbolaBinnedFeatures::update_features_async(
        const uint32_t & changed_node,
        const EbolaState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const EbolaState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
}


std::vector<uint32_t> EbolaBinnedFeatures::get_bins() const {
    // calcualate centrality for each location
    std::vector<std::pair<double, uint32_t> > centrality;
    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        double c(0.0);
        for (uint32_t j = 0; j < this->network_->size(); ++j) {
            if (j != i) {
                c += 1.0 / this->network_->dist().at(i).at(j);
            }
        }
        CHECK(std::isfinite(c));

        centrality.emplace_back(c, i);
    }
    // sort by centrality
    std::sort(centrality.begin(), centrality.end());

    const uint32_t num_per_bin(this->network_->size() / this->num_bins_);
    const uint32_t num_extra(this->network_->size() % this->num_bins_);
    CHECK_EQ(num_per_bin * this->num_bins_ + num_extra, this->network_->size());

    std::vector<uint32_t> bins(this->network_->size(), this->num_bins_);
    uint32_t beg(0);
    uint32_t end(num_per_bin);
    for (uint32_t bin = 0; bin < this->num_bins_; ++bin, end += num_per_bin) {
        // fill in the extras in the early bins
        if (bin < num_extra) {
            ++end;
        }
        for (uint32_t node(bin); node < end; ++node) {
            bins.at(centrality.at(node).second) = bin;
        }
        beg = end;
    }
    CHECK_EQ(beg, this->network_->size());
    std::for_each(bins.begin(), bins.end(),
            [this] (const uint32_t & x_) {
                CHECK_LT(x_, this->num_bins_);
            });
    return bins;
}


std::vector<std::vector<uint32_t> > EbolaBinnedFeatures::get_neigh() const {
    std::vector<std::vector<uint32_t> > neigh(this->network_->size());
    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        // sort locations by distance
        std::vector<std::pair<double, uint32_t > > pairs;
        for (uint32_t j = 0; j < this->network_->size(); ++j) {
            if (j != i) {
                pairs.emplace_back(this->network_->dist().at(i).at(j), j);
            }
        }
        std::sort(pairs.begin(), pairs.end());
        // pick off closest
        for (uint32_t j = 0; j < this->num_neigh_; ++j) {
            neigh.at(i).push_back(pairs.at(j).second);
        }
    }
    return neigh;
}


std::vector<std::vector<EbolaBinnedFeatures::Term> >
EbolaBinnedFeatures::get_terms() const {
    // prep data
    std::vector<double> norm_pop(EbolaData::population());
    const double mean(njm::tools::mean_and_var(norm_pop).first);
    njm::linalg::mult_b_to_a(norm_pop, 1.0 / mean);

    std::vector<double> unique_dist;
    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        for (uint32_t j = i + 1; j < this->network_->size(); ++j) {
            unique_dist.push_back(this->network_->dist().at(i).at(j));
        }
    }
    const double min_dist(*std::min_element(
                    unique_dist.begin(), unique_dist.end()));

    const double mean_dist(njm::tools::mean_and_var(unique_dist).first);


    std::vector<std::vector<Term> > terms(this->network_->size());
    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        uint32_t index(this->num_features_per_bin_ * this->bins_.at(i) * 4);
        // counts
        terms.at(i).emplace_back(Term{index, 1.0});
        index += 4;

        // population
        terms.at(i).emplace_back(Term{index, norm_pop.at(i)});
        index += 4;

        // neighbor counts
        for (uint32_t j = 0; j < this->num_neigh_; ++j) {
            const uint32_t neigh(this->neigh_.at(i).at(j));
            const double prox_weight(
                    std::exp(-this->network_->dist().at(i).at(neigh)
                            / min_dist));
            terms.at(neigh).emplace_back(Term{index, prox_weight});
        }
        index += 4;

        // neighbor populations
        for (uint32_t j = 0; j < this->num_neigh_; ++j) {
            const uint32_t neigh(this->neigh_.at(i).at(j));
            const double prox_weight(
                    std::exp(-this->network_->dist().at(i).at(neigh)
                            / min_dist));
            const double weight(prox_weight * norm_pop.at(neigh));
            terms.at(neigh).emplace_back(
                    Term{index, weight / this->num_neigh_});
        }
        index += 4;

        // grav
        for (uint32_t j = 0; j < this->num_neigh_; ++j) {
            const uint32_t neigh(this->neigh_.at(i).at(j));
            const double dist(this->network_->dist().at(i).at(neigh));
            const double prox_weight(std::exp(-dist / min_dist));
            const double weight(prox_weight * (dist / mean_dist)
                    / (norm_pop.at(i) * norm_pop.at(neigh)));
            terms.at(neigh).emplace_back(
                    Term{index, weight / this->num_neigh_});
        }
        index += 4;
    }

    return terms;
}


uint32_t EbolaBinnedFeatures::num_features() const {
    return this->num_bins_ * num_features_per_bin_ * 4;
}


} // namespace stdmMf
