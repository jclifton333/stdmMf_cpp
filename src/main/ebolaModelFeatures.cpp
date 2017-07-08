#include "ebolaModelFeatures.hpp"

#include "ebolaData.hpp"

#include "ebolaStateModel.hpp"

#include <njm_cpp/tools/stats.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

#include <glog/logging.h>

#include <armadillo>

namespace stdmMf {


const uint32_t EbolaModelFeatures::num_inf_features_ = 1;
const uint32_t EbolaModelFeatures::num_not_features_ = 1;


EbolaModelFeatures::EbolaModelFeatures(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<EbolaState> > & model)
    : network_(network), model_(model) {
    this->dist_mean_ = 0.0;
    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        for (uint32_t j = (i + 1); j < this->network_->size(); ++j) {
            this->dist_mean_ += this->network_->dist().at(i).at(j);
        }
    }
    this->dist_mean_ /=
        (this->network_->size() * (this->network_->size() - 1)) / 2;
}


EbolaModelFeatures::EbolaModelFeatures(const EbolaModelFeatures & other)
    : network_(other.network_), model_(other.model_->clone()),
      terms_(other.terms_) {
}


std::shared_ptr<Features<EbolaState> > EbolaModelFeatures::clone() const {
    return std::shared_ptr<Features<EbolaState> >(
            new EbolaModelFeatures(*this));
}


void EbolaModelFeatures::update(const EbolaState & curr_state,
        const std::vector<StateAndTrt<EbolaState> > & history,
        const uint32_t & num_trt) {
    const std::vector<Transition<EbolaState> > all_history(
            Transition<EbolaState>::from_sequence(history, curr_state));

    // fit model
    this->model_->est_par(all_history);

    // thompson sampling
    // get information matrix and take inverse sqrt
    std::vector<double> hess = this->model_->ll_hess(all_history);
    njm::linalg::mult_b_to_a(hess, -1.0 * all_history.size());

    const arma::mat hess_mat(hess.data(), this->model_->par_size(),
            this->model_->par_size());
    arma::mat eigvec;
    arma::vec eigval;
    arma::eig_sym(eigval, eigvec, hess_mat);
    for (uint32_t i = 0; i < this->model_->par_size(); ++i) {
        if (eigval(i) > 0.1)
            eigval(i) = std::sqrt(1.0 / eigval(i));
        else
            eigval(i) = 0.0;
    }
    // threshold eigen vectors
    for (auto it = eigvec.begin(); it != eigvec.end(); ++it) {
        if (std::abs(*it) < 1e-3) {
            *it = 0.0;
        }
    }
    arma::mat var_sqrt = eigvec * arma::diagmat(eigval) * eigvec.t();
    // threshold sqrt matrix
    for (auto it = var_sqrt.begin(); it != var_sqrt.end(); ++it) {
        if (*it < 1e-3) {
            *it = 0.0;
        }
    }

    // sample new parameters
    arma::vec std_norm(this->model_->par_size());
    for (uint32_t i = 0; i < this->model_->par_size(); ++i) {
        std_norm(i) = this->rng_->rnorm_01();
    }
    const std::vector<double> par_samp(
            njm::linalg::add_a_and_b(this->model_->par(),
                    arma::conv_to<std::vector<double> >::from(
                            var_sqrt * std_norm)));
    // check for finite values
    std::for_each(par_samp.begin(), par_samp.end(),
            [] (const double & x_) {
                LOG_IF(FATAL, !std::isfinite(x_));
            });

    // set new parameters
    this->model_->par(par_samp);
}


std::vector<double> EbolaModelFeatures::get_features(
        const EbolaState & state,
        const boost::dynamic_bitset<> & trt_bits) {
    // prep terms
    std::shared_ptr<EbolaStateModel> mod(
            std::static_pointer_cast<EbolaStateModel>(this->model_));

    // probs with no treatment
    const std::vector<double> probs(mod->probs(state,
                    boost::dynamic_bitset<>(this->network_->size())));
    // treat all inot infected locations
    boost::dynamic_bitset<> not_trt_bits(state.inf_bits);
    not_trt_bits.flip();
    const std::vector<double> probs_not_trt(mod->probs(state, not_trt_bits));

    this->terms_.resize(this->network_->size());

    const uint32_t num_not(this->network_->size() - state.inf_bits.count());

    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        this->terms_.at(i).clear();
        if (state.inf_bits.test(i)) {
            // double one_to_one(0.0);
            // double one_to_one_diff(0.0);
            double prob_dist(0.0);
            for (uint32_t j = 0; j < this->network_->size(); ++j) {
                if (!state.inf_bits.test(j) && i != j) {
                    // const double a_inf_b_no(mod->a_inf_b(i, j, false, false,
                    //                 state, trt_bits // these don't matter
                    //                 ));
                    // const double a_inf_b_yes(mod->a_inf_b(i, j, true, false,
                    //                 state, trt_bits // these don't matter
                    //                 ));
                    // one_to_one += a_inf_b_no;
                    // one_to_one_diff += a_inf_b_no - a_inf_b_yes;

                    prob_dist += probs.at(j) * this->dist_mean_
                        / this->network_->dist().at(i).at(j);
                }
            }
            if (num_not > 0) {
                // one_to_one /= num_not;
                // one_to_one_diff /= num_not;
                prob_dist /= num_not;
            }

            // this->terms_.at(i).emplace_back(Term{1, one_to_one});
            // this->terms_.at(i).emplace_back(Term{3, one_to_one_diff});
            // this->terms_.at(i).emplace_back(Term{5, prob_dist});

            this->terms_.at(i).emplace_back(Term{1, prob_dist});
        } else {
            // this->terms_.at(i).emplace_back(Term{7, probs.at(i)});
            // // benefit of treating
            // this->terms_.at(i).emplace_back(
            //         Term{9, probs.at(i) - probs_not_trt.at(i)});

            // double inf_effect(0.0);
            // for (uint32_t j = 0; j < this->network_->size(); ++j) {
            //     if (!state.inf_bits.test(j) && i != j) {
            //         const double a_inf_b_no(mod->a_inf_b(i, j, false, false,
            //                         state, trt_bits // these don't matter
            //                         ));
            //         inf_effect += a_inf_b_no * (1.0 - probs.at(j));
            //     }
            // }
            // if (num_not > 0) {
            //     inf_effect /= num_not;
            // }

            // this->terms_.at(i).emplace_back(Term{11, inf_effect});

            this->terms_.at(i).emplace_back(Term{3, probs.at(i)});
        }
    }

    // calculate features
    std::vector<double> feat(this->num_features(), 0.0);
    feat.at(0) = 1.0;

    for (uint32_t i = 0; i < this->network_->size(); ++i) {
        const std::vector<Term> & t(this->terms_.at(i));
        std::vector<Term>::const_iterator it,end(t.end());
        if (trt_bits.test(i)) {
            for (it = t.begin(); it != end; ++it) {
                feat.at(it->index + 1) += it->weight;
            }
        } else {
            for (it = t.begin(); it != end; ++it) {
                feat.at(it->index) += it->weight;
            }
        }
    }

    return feat;
}


void EbolaModelFeatures::update_features(
        const uint32_t & changed_node,
        const EbolaState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const EbolaState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) {

    const bool trt_now(trt_bits_new.test(changed_node));
    const bool trt_before(trt_bits_old.test(changed_node));

    const std::vector<Term> & t(this->terms_.at(changed_node));
    std::vector<Term>::const_iterator it,end(t.end());
    if (trt_now && !trt_before) {
        for (it = t.begin(); it != end; ++it) {
            feat.at(it->index) -= it->weight;
            feat.at(it->index + 1) += it->weight;
        }
    } else if (!trt_now && trt_before) {
        for (it = t.begin(); it != end; ++it) {
            feat.at(it->index + 1) -= it->weight;
            feat.at(it->index) += it->weight;
        }
    } else {
        LOG(FATAL) << "no trt change for node: " << changed_node
                   << "trt (" << trt_now << ", " << trt_before
                   << " inf (" << state_new.inf_bits.test(changed_node)
                   << ", " << state_old.inf_bits.test(changed_node);
    }
}


void EbolaModelFeatures::update_features_async(
        const uint32_t & changed_node,
        const EbolaState & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const EbolaState & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) const {
    LOG(FATAL) << "NOT IMPLEMENTED";
}


uint32_t EbolaModelFeatures::num_features() const {
    return 1 + this->num_inf_features_ * 2 + this->num_not_features_ * 2;
}


} // namespace stdmMf
