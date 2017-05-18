#include "finiteQfnFeatures.hpp"

#include <njm_cpp/info/project.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

#include "system.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "sweepAgent.hpp"

#include "utilities.hpp"

#include <cmath>
#include <glog/logging.h>
#include <armadillo>

#include <glog/logging.h>

namespace stdmMf {


template <typename State>
FiniteQfnFeatures<State>::FiniteQfnFeatures(
        const std::shared_ptr<const Network> & network,
        const std::vector<std::shared_ptr<Model<State> > > & models,
        const std::shared_ptr<Features<State> > & features,
        const uint32_t & look_ahead)
    : network_(network), num_nodes_(this->network_->size()), models_(models),
      num_models_(this->models_.size()), features_(features),
      look_ahead_(look_ahead), coef_(this->num_models_) {

    CHECK_GT(this->look_ahead_, 0);

    std::for_each(this->models_.begin(), this->models_.end(),
            [this] (const std::shared_ptr<Model<State> > & m_) {
                m_->rng(this->rng());
            });
    this->features_->rng(this->rng());

    // resize coefficients
    std::for_each(this->coef_.begin(), this->coef_.end(),
            [this] (std::vector<std::vector<double> > & c_) {
                c_.resize(this->look_ahead_,
                        std::vector<double>(0.0,
                                this->features_->num_features()));
            });
}


template <typename State>
FiniteQfnFeatures<State>::FiniteQfnFeatures(
        const FiniteQfnFeatures<State> & other)
    : network_(other.network_),  num_nodes_(other.num_nodes_),
      models_(clone_vec(other.models_)), num_models_(other.num_models_),
      features_(other.features_->clone()),
      look_ahead_(other.look_ahead_), coef_(other.coef_),
      last_feat_(other.last_feat_) {

    std::for_each(this->models_.begin(), this->models_.end(),
            [this] (const std::shared_ptr<Model<State> > & m_) {
                m_->rng(this->rng());
            });
    this->features_->rng(this->rng());
}


template <typename State>
std::shared_ptr<Features<State> >FiniteQfnFeatures<State>::clone() const {
    return std::shared_ptr<Features<State> >(new FiniteQfnFeatures(*this));
}


template <typename State>
void FiniteQfnFeatures<State>::update(const State & curr_state,
        const std::vector<StateAndTrt<State> > & history,
        const uint32_t & num_trt) {
    const std::vector<Transition<State> > all_history(
            Transition<State>::from_sequence(history, curr_state));
    for (uint32_t m = 0; m < this->num_models_; ++m) {
        // fit model
        this->models_.at(m)->est_par(all_history);

        // thompson sampling
        // get information matrix and take inverse sqrt
        std::vector<double> hess = this->models_.at(m)->ll_hess(all_history);
        njm::linalg::mult_b_to_a(hess, -1.0 * all_history.size());

        const arma::mat hess_mat(hess.data(), this->models_.at(m)->par_size(),
                this->models_.at(m)->par_size());
        arma::mat eigvec;
        arma::vec eigval;
        arma::eig_sym(eigval, eigvec, hess_mat);
        for (uint32_t i = 0; i < this->models_.at(m)->par_size(); ++i) {
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
        arma::vec std_norm(this->models_.at(m)->par_size());
        for (uint32_t i = 0; i < this->models_.at(m)->par_size(); ++i) {
            std_norm(i) = this->rng_->rnorm_01();
        }
        const std::vector<double> par_samp(
                njm::linalg::add_a_and_b(this->models_.at(m)->par(),
                        arma::conv_to<std::vector<double> >::from(
                                var_sqrt * std_norm)));
        // check for finite values
        std::for_each(par_samp.begin(), par_samp.end(),
                [] (const double & x_) {
                    LOG_IF(FATAL, !std::isfinite(x_));
                });

        // set new parameters
        this->models_.at(m)->par(par_samp);
    }

    const std::vector<std::vector<Transition<State> > > sim_data(
            this->generate_data(100, 100));

    this->fit_q_functions(sim_data);
}


template <typename State>
std::vector<double> FiniteQfnFeatures<State>::get_features(
        const State & state,
        const boost::dynamic_bitset<> & trt_bits) {
    std::vector<double> features;
    features.reserve(this->num_features());
    features.push_back(1.0);

    this->last_feat_ = this->features_->get_features(state, trt_bits);

    for (uint32_t m = 0; m < this->num_models_; ++m) {
        for (uint32_t i = 0; i < this->look_ahead_; ++i) {
            features.push_back(njm::linalg::dot_a_and_b(
                            this->last_feat_, this->coef_.at(m).at(i)));
        }
    }
    return features;
}


template <typename State>
void FiniteQfnFeatures<State>::update_features(
        const uint32_t & changed_node,
        const State & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const State & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) {
    this->features_->update_features(changed_node, state_new, trt_bits_new,
            state_old, trt_bits_old, this->last_feat_);
    for (uint32_t m = 0; m < this->num_models_; ++m) {
        for (uint32_t i = 0; i < this->look_ahead_; ++i) {
            feat.at(i + 1) = njm::linalg::dot_a_and_b(
                    this->last_feat_, this->coef_.at(m).at(i));
        }
    }
}


template <typename State>
void FiniteQfnFeatures<State>::update_features_async(
        const uint32_t & changed_node,
        const State & state_new,
        const boost::dynamic_bitset<> & trt_bits_new,
        const State & state_old,
        const boost::dynamic_bitset<> & trt_bits_old,
        std::vector<double> & feat) const {
    LOG(FATAL) << "this is not implemented";
}


template <typename State>
std::vector<std::vector<Transition<State> > >
FiniteQfnFeatures<State>::generate_data(
        const uint32_t & num_episodes, const uint32_t & num_obs_per_episode) {
    // setup return container
    std::vector<std::vector<Transition<State> > > transitions(
            this->num_models_);
    std::for_each(transitions.begin(), transitions.end(),
            [&] (std::vector<Transition<State> > & t_) {
                t_.reserve(num_episodes * num_obs_per_episode);
            });

    for (uint32_t m = 0; m < this->num_models_; ++m) {
        // system
        System<State> s(this->network_, this->models_.at(m));
        s.rng(this->rng());

        // agents
        ProximalAgent<State> proximal_agent(this->network_);
        proximal_agent.rng(this->rng());
        RandomAgent<State> random_agent(this->network_);
        proximal_agent.rng(this->rng());

        // simualate data
        for (uint32_t i = 0; i < num_episodes; ++i) {
            s.start();
            for (uint32_t j = 0; j < num_obs_per_episode; ++j) {
                // save initial state
                const State curr_state(s.state());

                // randomly pick between proximal and random
                boost::dynamic_bitset<> trt_bits;
                const auto draw = this->rng()->rint(0, 2);
                if (draw == 0) {
                    trt_bits = proximal_agent.apply_trt(s.state(), s.history());
                } else if (draw == 1) {
                    trt_bits = random_agent.apply_trt(s.state(), s.history());
                }

                // simualate
                s.trt_bits(trt_bits);
                s.turn_clock();

                // save next state
                const State next_state(s.state());

                // save transition
                transitions.at(m).emplace_back(curr_state, trt_bits,
                        next_state);
            }
        }
    }
    return transitions;
}


template <typename State>
void FiniteQfnFeatures<State>::fit_q_functions(
        const std::vector<std::vector<Transition<State> > > & obs) {
    for (uint32_t m = 0; m < this->num_models_; ++m) {
        const uint32_t num_obs(obs.at(m).size());
        std::vector<uint32_t> train_index, test_index;
        train_index.reserve(num_obs);
        test_index.reserve(num_obs);
        std::vector<StateAndTrt<State> > state_trt_train, state_trt_test;
        state_trt_train.reserve(num_obs);
        state_trt_test.reserve(num_obs);
        std::vector<double> outcomes_train, outcomes_test;
        outcomes_train.reserve(num_obs);
        outcomes_test.reserve(num_obs);

        // outcomes from history except first observation
        for (uint32_t i = 0; i < num_obs; ++i) {
            if (this->rng()->runif_01() < 0.2) {
                // test sets
                test_index.push_back(i);
                state_trt_test.emplace_back(obs.at(m).at(i).curr_state,
                        obs.at(m).at(i).curr_trt_bits);
                const uint32_t num_inf(
                        obs.at(m).at(i).next_state.inf_bits.count());
                outcomes_test.push_back(
                        - static_cast<float>(num_inf)
                        / static_cast<float>(this->num_nodes_));
            } else {
                // train sets
                train_index.push_back(i);
                state_trt_train.emplace_back(obs.at(m).at(i).curr_state,
                        obs.at(m).at(i).curr_trt_bits);
                const uint32_t num_inf(
                        obs.at(m).at(i).next_state.inf_bits.count());
                outcomes_train.push_back(
                        - static_cast<float>(num_inf)
                        / static_cast<float>(this->num_nodes_));
            }
        }
        const uint32_t num_train(state_trt_train.size());
        const uint32_t num_test(state_trt_test.size());

        // train first neural network using outcomes
        this->fit_q_function(0, m, state_trt_train, outcomes_train,
                state_trt_test, outcomes_test);

        // train remaining neural networks using outcome plus arg max of
        // previous networks
        for (uint32_t i = 1; i < this->look_ahead_; ++i) {
            std::vector<double> outcome_plus_max_train, outcome_plus_max_test;
            outcome_plus_max_train.reserve(num_obs);
            outcome_plus_max_test.reserve(num_obs);

            SweepAgent<State> sa(this->network_, this->features_,
                    this->coef_.at(m).at(i - 1), njm::linalg::dot_a_and_b,
                    2, true);

            // find max for next states and add to outcomes
            for (uint32_t j = 0; j < num_train; ++j) {
                const auto & next_state(
                        obs.at(m).at(train_index.at(j)).next_state);
                const boost::dynamic_bitset<> trt_bits(
                        sa.apply_trt(next_state));

                const std::vector<double> feat(
                        this->features_->get_features(next_state, trt_bits));

                outcome_plus_max_train.push_back(outcomes_train.at(j)
                        + njm::linalg::dot_a_and_b(feat,
                                this->coef_.at(m).at(i - 1)));
            }

            // find max for next states and add to outcomes
            for (uint32_t j = 0; j < num_test; ++j) {
                const auto & next_state(
                        obs.at(m).at(test_index.at(j)).next_state);
                const boost::dynamic_bitset<> trt_bits(
                        sa.apply_trt(next_state));

                const std::vector<double> feat(
                        this->features_->get_features(next_state, trt_bits));

                outcome_plus_max_test.push_back(outcomes_test.at(j)
                        + njm::linalg::dot_a_and_b(feat,
                                this->coef_.at(m).at(i - 1)));
            }

            // train neural network
            this->fit_q_function(i, m, state_trt_train, outcome_plus_max_train,
                    state_trt_test, outcome_plus_max_test);
        }
    }
}


template <typename State>
void FiniteQfnFeatures<State>::fit_q_function(const uint32_t & qfn_index,
        const uint32_t & model_index,
        const std::vector<StateAndTrt<State> > & state_trt_train,
        const std::vector<double> & outcomes_train,
        const std::vector<StateAndTrt<State> > & state_trt_test,
        const std::vector<double> & outcomes_test) {
    const uint32_t num_train(state_trt_train.size());
    CHECK_EQ(num_train, outcomes_train.size());
    const uint32_t num_test(state_trt_test.size());
    CHECK_EQ(num_test, outcomes_test.size());
    const uint32_t num_features(this->features_->num_features());

    arma::mat xt_train(num_features, num_train),
        xt_test(num_features, num_test);
    arma::vec y_train(num_train), y_test(num_test);

    // train
    for (uint32_t i = 0; i < num_train; ++i) {
        // calculate features
        const std::vector<double> feat(this->features_->get_features(
                        state_trt_train.at(i).state,
                        state_trt_train.at(i).trt_bits));

        // set column in x matrix
        auto xt_it(xt_train.begin_col(i));
        auto f_it(feat.begin());
        for (uint32_t j = 0; j < num_features; ++j, ++xt_it, ++f_it) {
            *xt_it = *f_it;
        }

        // set y
        y_train(i) = outcomes_train.at(i);
    }

    // test
    for (uint32_t i = 0; i < num_test; ++i) {
        // calculate features
        const std::vector<double> feat(this->features_->get_features(
                        state_trt_test.at(i).state,
                        state_trt_test.at(i).trt_bits));

        // set column in x matrix
        auto xt_it(xt_test.begin_col(i));
        auto f_it(feat.begin());
        for (uint32_t j = 0; j < num_features; ++j, ++xt_it, ++f_it) {
            *xt_it = *f_it;
        }

        // set y
        y_test(i) = outcomes_test.at(i);
    }

    const arma::mat xtx_train(xt_train * xt_train.t());
    const arma::vec xty_train(xt_train * y_train);

    const std::vector<double> lambda_vals({1.0, 10.0, 25.0, 50.0, 100.0,
                                           250.0, 500.0, 1000.0});
    double best_ss = std::numeric_limits<double>::infinity();
    arma::vec best_beta;

    arma::mat pen_eye(arma::eye(num_features, num_features));
    pen_eye(0, 0) = 0.0;

    for (uint32_t i = 0; i < lambda_vals.size(); ++i) {
        const double & lambda(lambda_vals.at(i));

        // cholesky on xtx + lamba * pen_eye, where pen_eye is the
        // identity with a zero in the top left element
        const arma::mat r_train(arma::chol(xtx_train + pen_eye * lambda));

        // forward solve then backward solve
        const arma::vec beta(arma::solve(arma::trimatu(r_train),
                        arma::solve(arma::trimatl(r_train.t()),
                                xty_train)));

        const double ss_test(arma::norm(y_test - (beta.t() * xt_test).t(), 2));

        if (best_ss > ss_test) {
            best_ss = ss_test;
            best_beta = beta;
        }
    }

    CHECK(std::isfinite(best_ss));

    { // assign coefficient values
        coef_.at(model_index).at(qfn_index).resize(num_features);
        auto coef_it(coef_.at(model_index).at(qfn_index).begin());
        auto beta_it(best_beta.begin());
        for (uint32_t i = 0; i < num_features; ++i, ++coef_it, ++beta_it) {
            *coef_it = *beta_it;
        }
    }
}


template <typename State>
uint32_t FiniteQfnFeatures<State>::num_features() const {
    return this->look_ahead_ * this->num_models_ + 1;
}


template <typename State>
void FiniteQfnFeatures<State>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
    std::for_each(this->models_.begin(), this->models_.end(),
            [this] (const std::shared_ptr<Model<State> > & m_) {
                m_->rng(this->rng());
            });
    this->features_->rng(rng);
}


template class FiniteQfnFeatures<InfState>;
template class FiniteQfnFeatures<InfShieldState>;

} // namespace stdmMf
