#include "ebolaFeatures.hpp"

#include "ebolaData.hpp"

#include <njm_cpp/tools/stats.hpp>

#include <glog/logging.h>


namespace stdmMf {

const uint32_t EbolaFeatures::num_solo_ = 2;
const uint32_t EbolaFeatures::num_joint_ = 4;


EbolaFeatures::EbolaFeatures(const std::shared_ptr<const Network> & network,
        const uint32_t & num_base_locs, const uint32_t & num_neigh)
    : Features<EbolaState>(), network_(network), num_base_locs_(num_base_locs),
      num_neigh_(num_neigh) {
    CHECK_GT(this->num_base_locs_, 0);
    CHECK_LE(this->num_base_locs_, network->size());
    CHECK_LE(this->num_neigh_, network->size());

    // calc centrality
    std::vector<std::pair<double, uint32_t> > centrality;
    centrality.reserve(network->size());
    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        double c(0.0);
        for (uint32_t j = 0; j < this->network_->size(); ++j) {
            c += 1.0 / network->dist().at(i).at(j);
        }
        // want to sort in decreasing order wrt c, so multiply c by -1
        centrality.emplace_back(-c, i);
    }
    std::sort(centrality.begin(), centrality.end());

    // get unique distance values
    std::vector<double> unique_dist;
    unique_dist.reserve((this->network_->size() - 1)
            * this->network_->size() / 2);
    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        for (uint32_t j = i + 1; j < this->network_->size(); ++j) {
            unique_dist.push_back(this->network_->dist().at(i).at(j));
        }
    }
    // mean and variance of distance
    const std::pair<double, double> dist_mean_var (
            njm::tools::mean_and_var(unique_dist));

    // get base and neighbor locations
    std::vector<uint32_t> base_locs;
    std::vector<std::vector<std::pair<uint32_t, double> > > neigh_locs;
    for (uint32_t i = 0; i < num_base_locs; ++i) {
        const uint32_t index(i * network->size() / num_base_locs);
        base_locs.push_back(index);

        const std::vector<double> & dist(this->network_->dist().at(index));
        std::vector<std::pair<double, uint32_t> > weights;
        for (uint32_t j = 0; j < this->network_->size(); ++j) {
            if (j != index) {
                const double & d(dist.at(j));
                const double adj_d((d - dist_mean_var.first)
                        / std::sqrt(2.0 * dist_mean_var.second));

                // add negative to sort in ascending order
                weights.emplace_back(- 0.5 * (1.0 + std::erf(adj_d)), j);
            }
        }
        std::sort(weights.begin(), weights.end());
        std::vector<std::pair<uint32_t, double> > add_to_neigh;
        for (uint32_t j = 0; j < this->num_neigh_; ++j) {
            // correct sign on weights because it was previously
            // multiplied by -1 for sorting reasons
            add_to_neigh.emplace_back(- weights.at(j).second,
                    weights.at(j).first);
        }

        neigh_locs.push_back(std::move(add_to_neigh));
    }


    // calculate statistics
    std::vector<std::vector<double> > solo_stats(this->num_solo_);
    std::vector<std::vector<double> > joint_stats(this->num_joint_);
    for (uint32_t i = 0; i < this->num_base_locs_; ++i) {
        const uint32_t & base(base_locs.at(i));

        uint32_t solo_index(0);
        // constant
        solo_stats.at(solo_index++).push_back(1.0);
        // population
        solo_stats.at(solo_index++).push_back(EbolaData::population().at(base));

        CHECK_EQ(solo_index, this->num_solo_);

        for (uint32_t j = 0; j < this->num_neigh_; ++j) {
            const uint32_t & neigh(neigh_locs.at(i).at(j).first);
            uint32_t joint_index(0);
            // constant
            joint_stats.at(joint_index++).push_back(1.0);
            // distance
            joint_stats.at(joint_index++).push_back(
                    this->network_->dist().at(base).at(neigh));
            // population product
            joint_stats.at(joint_index++).push_back(
                    EbolaData::population().at(base)
                    * EbolaData::population().at(neigh));
            // gravity
            joint_stats.at(joint_index++).push_back(
                    this->network_->dist().at(base).at(neigh)
                    / (EbolaData::population().at(base)
                            * EbolaData::population().at(neigh)));

            CHECK_EQ(joint_index, this->num_joint_);
        }
    }

    // get mean and variance for statistics
    std::vector<std::pair<double, double> > solo_mean_var(solo_stats.size());
    std::vector<std::pair<double, double> > joint_mean_var(joint_stats.size());
    std::transform(solo_stats.begin(), solo_stats.end(),
            solo_mean_var.begin(), njm::tools::mean_and_var);
    std::transform(joint_stats.begin(), joint_stats.end(),
            joint_mean_var.begin(), njm::tools::mean_and_var);


    // create terms
    this->neither_.reserve(this->network_->size());
    this->only_inf_.reserve(this->network_->size());
    this->only_trt_.reserve(this->network_->size());
    this->both_.reserve(this->network_->size());

    uint32_t term_index = 1; // zero is the intercept
    for (uint32_t i = 0; i < this->num_base_locs_; ++i) {
        const uint32_t & base(base_locs.at(i));
        // solo terms
        for (uint32_t m = 0; m < solo_stats.size(); ++m) {
            Term t;
            t.index = term_index++;

            const auto & mean_var(solo_mean_var.at(m));
            if (mean_var.second > 0.0) {
                t.weight = (solo_stats.at(m).at(base) - mean_var.first)
                    / mean_var.second;
            } else {
                CHECK_EQ(solo_stats.at(m).at(base), 1.0);
                t.weight = 1.0;
            }

            // add term
            this->neither_.at(base).push_back(t);
            this->only_inf_.at(base).push_back(t);
            this->only_trt_.at(base).push_back(t);
            this->both_.at(base).push_back(t);
        }

        // joint terms
        for (uint32_t m = 0; m < joint_stats.size(); ++m) {
            for (uint32_t j = 0; j < this->num_neigh_; ++j) {
                const uint32_t & neigh(neigh_locs.at(i).at(j).first);
                Term t;
                t.index = term_index; // don't increment

                const double & dist_weight(neigh_locs.at(i).at(j).second);

                const auto & mean_var(joint_mean_var.at(m));
                const uint32_t joint_index(i * this->num_neigh_ + j);
                if (mean_var.second > 0.0) {
                    t.weight = dist_weight
                        * (joint_stats.at(m).at(joint_index)
                                - mean_var.first)
                        / mean_var.second;
                } else {
                    t.weight = dist_weight;
                }

                // add term
                this->neither_.at(neigh).push_back(t);
                this->only_inf_.at(neigh).push_back(t);
                this->only_trt_.at(neigh).push_back(t);
                this->both_.at(neigh).push_back(t);
            }
            // increment for next feature
            ++term_index;
        }
    }

    CHECK_EQ(term_index, this->num_features());
}


EbolaFeatures::EbolaFeatures(const EbolaFeatures & other)
    : Features<EbolaState>(other), network_(other.network_),
      num_base_locs_(other.num_base_locs_), num_neigh_(other.num_neigh_),
      neither_(other.neither_), only_inf_(other.only_inf_),
      only_trt_(other.only_trt_), both_(other.both_) {
}


std::shared_ptr<Features<EbolaState> > EbolaFeatures::clone() const {
    return std::shared_ptr<Features<EbolaState> > (new EbolaFeatures(*this));
}


std::vector<double> EbolaFeatures::get_features(
        const EbolaState & state,
        const boost::dynamic_bitset<> & trt_bits) {
}


void EbolaFeatures::update_features(
        const uint32_t & changed_node,
        const EbolaState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const EbolaState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) {
}


void EbolaFeatures::update_features_async(
        const uint32_t & changed_node,
        const EbolaState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const EbolaState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) const {
}


uint32_t EbolaFeatures::num_features() const {
    return 1 + this->num_base_locs_ * this->num_solo_
        + this->num_base_locs_ * this->num_joint_;
}


} // namespace stdmMf
