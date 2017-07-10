#include "ebolaTransProbFeatures.hpp"

#include "ebolaData.hpp"

#include "ebolaStateModel.hpp"

#include <njm_cpp/tools/stats.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/tools/bitManip.hpp>

#include <glog/logging.h>

#include <armadillo>

namespace stdmMf {


EbolaTransProbFeatures::EbolaTransProbFeatures(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<EbolaState> > & model)
    : network_(network), model_(model) {
}


EbolaTransProbFeatures::EbolaTransProbFeatures(
        const EbolaTransProbFeatures & other)
    : network_(other.network_), model_(other.model_->clone()),
      all_probs_(other.all_probs_), to_probs_(other.to_probs_),
      from_probs_(other.from_probs_) {
}


std::shared_ptr<Features<EbolaState> > EbolaTransProbFeatures::clone() const {
    return std::shared_ptr<Features<EbolaState> >(
            new EbolaTransProbFeatures(*this));
}


void EbolaTransProbFeatures::update(const EbolaState & curr_state,
        const std::vector<StateAndTrt<EbolaState> > & history,
        const uint32_t & num_trt) {
    const std::vector<Transition<EbolaState> > all_history(
            Transition<EbolaState>::from_sequence(history, curr_state));

    // fit model
    this->model_->est_par(all_history);

    // // thompson sampling
    // // get information matrix and take inverse sqrt
    // std::vector<double> hess = this->model_->ll_hess(all_history);
    // njm::linalg::mult_b_to_a(hess, -1.0 * all_history.size());

    // const arma::mat hess_mat(hess.data(), this->model_->par_size(),
    //         this->model_->par_size());
    // arma::mat eigvec;
    // arma::vec eigval;
    // arma::eig_sym(eigval, eigvec, hess_mat);
    // for (uint32_t i = 0; i < this->model_->par_size(); ++i) {
    //     if (eigval(i) > 0.1)
    //         eigval(i) = std::sqrt(1.0 / eigval(i));
    //     else
    //         eigval(i) = 0.0;
    // }
    // // threshold eigen vectors
    // for (auto it = eigvec.begin(); it != eigvec.end(); ++it) {
    //     if (std::abs(*it) < 1e-3) {
    //         *it = 0.0;
    //     }
    // }
    // arma::mat var_sqrt = eigvec * arma::diagmat(eigval) * eigvec.t();
    // // threshold sqrt matrix
    // for (auto it = var_sqrt.begin(); it != var_sqrt.end(); ++it) {
    //     if (*it < 1e-3) {
    //         *it = 0.0;
    //     }
    // }

    // // sample new parameters
    // arma::vec std_norm(this->model_->par_size());
    // for (uint32_t i = 0; i < this->model_->par_size(); ++i) {
    //     std_norm(i) = this->rng_->rnorm_01();
    // }
    // const std::vector<double> par_samp(
    //         njm::linalg::add_a_and_b(this->model_->par(),
    //                 arma::conv_to<std::vector<double> >::from(
    //                         var_sqrt * std_norm)));
    // // check for finite values
    // std::for_each(par_samp.begin(), par_samp.end(),
    //         [] (const double & x_) {
    //             LOG_IF(FATAL, !std::isfinite(x_));
    //         });

    // // set new parameters
    // this->model_->par(par_samp);

    this->update_all_probs(curr_state);
}

void EbolaTransProbFeatures::update_all_probs(
        const EbolaState & curr_state) {
    std::shared_ptr<EbolaStateModel> mod(
            std::static_pointer_cast<EbolaStateModel>(this->model_));

        const uint32_t num_nodes(this->network_->size());

        this->all_probs_.resize(num_nodes,
                std::vector<std::vector<double> >(num_nodes,
                        std::vector<double>(4, 0.0)));

        const boost::dynamic_bitset<> blank_trt(num_nodes);

        for (uint32_t not_idx = 0; not_idx < num_nodes; ++not_idx) {
            for (uint32_t inf_idx = 0; inf_idx < num_nodes; ++inf_idx) {
                const double neither(mod->a_inf_b(inf_idx, not_idx,
                                false, false, curr_state, blank_trt));

                const double not_trt(mod->a_inf_b(inf_idx, not_idx,
                                false, true, curr_state, blank_trt));

                const double inf_trt(mod->a_inf_b(inf_idx, not_idx,
                                true, false, curr_state, blank_trt));

                const double both(mod->a_inf_b(inf_idx, not_idx,
                                true, true, curr_state, blank_trt));

                this->all_probs_.at(not_idx).at(inf_idx) =
                    {neither, not_trt, inf_trt, both};
            }
        }
    }


std::vector<double> EbolaTransProbFeatures::get_features(
        const EbolaState & state,
        const boost::dynamic_bitset<> & trt_bits) {
    std::vector<double> feat(this->num_features(), 0.0);
    CHECK_EQ(this->num_features(), 3);
    feat.at(0) = 1.0;
    const uint32_t num_nodes(this->network_->size());

    const auto sets(njm::tools::both_sets(state.inf_bits));
    const std::vector<uint32_t> & inf_vec(sets.first);
    const std::vector<uint32_t> & not_vec(sets.second);

    if (inf_vec.size())
        CHECK(state.inf_bits.test(inf_vec.at(0)));
    if (not_vec.size())
        CHECK(!state.inf_bits.test(not_vec.at(0)));

    std::vector<uint32_t>::const_iterator inf_it, inf_end, not_it, not_end;
    inf_end = inf_vec.end();
    not_end = not_vec.end();

    // clear containers
    this->to_probs_.resize(num_nodes);
    this->from_probs_.resize(num_nodes);

    for (inf_it = inf_vec.begin(); inf_it != inf_end; ++inf_it) {
        double to_prob(0.0);
        const uint32_t inf_trt_bits(trt_bits.test(*inf_it) ? 2 : 0);
        for (not_it = not_vec.begin(); not_it != not_end; ++not_it) {
            const uint32_t not_trt_bits(trt_bits.test(*not_it) ? 1 : 0);

            to_prob += std::log(1.0 - this->all_probs_.at(*not_it).at(*inf_it)
                    .at(inf_trt_bits | not_trt_bits));
        }
        this->to_probs_.at(*inf_it) = std::exp(to_prob);
        feat.at(1) += std::exp(to_prob);
    }


    for (not_it = not_vec.begin(); not_it != not_end; ++not_it) {
        double from_prob(0.0);
        const uint32_t not_trt_bits(trt_bits.test(*not_it) ? 1 : 0);
        for (inf_it = inf_vec.begin(); inf_it != inf_end; ++inf_it) {
            const uint32_t inf_trt_bits(trt_bits.test(*inf_it) ? 2 : 0);

            from_prob += std::log(1.0 - this->all_probs_.at(*not_it).at(*inf_it)
                    .at(inf_trt_bits | not_trt_bits));
        }
        this->from_probs_.at(*not_it) = std::exp(from_prob);
        feat.at(2) += std::exp(from_prob);
    }
    return feat;
}


void EbolaTransProbFeatures::update_features(
        const uint32_t & changed_node,
        const EbolaState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const EbolaState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) {

    // treatment status
    const bool trt_now(trt_bits_new.test(changed_node));
    const bool trt_before(trt_bits_old.test(changed_node));

    CHECK_NE(trt_now, trt_before)
        << "Treatment status did not change for " << changed_node;

    CHECK_EQ(state_new.inf_bits.test(changed_node),
            state_old.inf_bits.test(changed_node))
        << "Infection status changed for " << changed_node;

    // update terms
    if (state_new.inf_bits.test(changed_node)) {
        // infected

        const uint32_t inf_trt_bits_new(trt_now * 2);
        const uint32_t inf_trt_bits_old(trt_before * 2);

        // update to_prob
        // remove old
        feat.at(1) -= this->to_probs_.at(changed_node);

        std::vector<uint32_t> not_vec(njm::tools::inactive_set(
                        state_new.inf_bits));
        std::vector<uint32_t>::iterator not_it, not_end;
        not_end = not_vec.end();

        double to_prob(0.0);
        for (not_it = not_vec.begin(); not_it != not_end; ++not_it) {
            const uint32_t not_trt_bits(trt_bits_new.test(*not_it) ? 1 : 0);

            to_prob += std::log(1.0 - this->all_probs_.at(*not_it)
                    .at(changed_node).at(inf_trt_bits_new | not_trt_bits));
        }
        this->to_probs_.at(changed_node) = std::exp(to_prob);
        // add new
        feat.at(1) += std::exp(to_prob);

        // update from_prob
        feat.at(2) = 0.0;
        for (not_it = not_vec.begin(); not_it != not_end; ++not_it) {
            const uint32_t not_trt_bits(trt_bits_new.test(*not_it) ? 1 : 0);

            // update
            this->from_probs_.at(*not_it) *= (1.0 - this->all_probs_.at(*not_it)
                    .at(changed_node).at(inf_trt_bits_new | not_trt_bits));
            this->from_probs_.at(*not_it) /= (1.0 - this->all_probs_.at(*not_it)
                    .at(changed_node).at(inf_trt_bits_old | not_trt_bits));

            feat.at(2) += this->from_probs_.at(*not_it);
        }
    } else {
        // not infected

        const uint32_t not_trt_bits_new(trt_now);
        const uint32_t not_trt_bits_old(trt_before);

        // update from_prob
        // remove old
        feat.at(2) -= this->from_probs_.at(changed_node);

        std::vector<uint32_t> inf_vec(njm::tools::active_set(
                        state_new.inf_bits));
        std::vector<uint32_t>::iterator inf_it, inf_end;
        inf_end = inf_vec.end();

        double from_prob(0.0);
        for (inf_it = inf_vec.begin(); inf_it != inf_end; ++inf_it) {
            const uint32_t inf_trt_bits(trt_bits_new.test(*inf_it) ? 2 : 0);

            from_prob += std::log(1.0 - this->all_probs_.at(changed_node)
                    .at(*inf_it).at(inf_trt_bits | not_trt_bits_new));
        }
        this->from_probs_.at(changed_node) = std::exp(from_prob);
        // add new
        feat.at(2) += std::exp(from_prob);

        // update to_prob
        feat.at(1) = 0.0;
        for (inf_it = inf_vec.begin(); inf_it != inf_end; ++inf_it) {
            const uint32_t inf_trt_bits(trt_bits_new.test(*inf_it) ? 2 : 0);

            // update
            this->to_probs_.at(*inf_it) *= (1.0 - this->all_probs_
                    .at(changed_node).at(*inf_it)
                    .at(inf_trt_bits | not_trt_bits_new));
            this->to_probs_.at(*inf_it) /= (1.0 - this->all_probs_
                    .at(changed_node).at(*inf_it)
                    .at(inf_trt_bits | not_trt_bits_old));

            feat.at(1) += this->to_probs_.at(*inf_it);
        }
    }
}


void EbolaTransProbFeatures::update_features_async(
        const uint32_t & changed_node,
        const EbolaState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const EbolaState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
}


uint32_t EbolaTransProbFeatures::num_features() const {
    return 1 + 2;
}


} // namespace stdmMf
