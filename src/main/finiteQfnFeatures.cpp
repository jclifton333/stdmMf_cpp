#include "finiteQfnFeatures.hpp"

#include <njm_cpp/info/project.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>

#include "system.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "sweepAgent.hpp"

#include <cmath>
#include <glog/logging.h>
#include <armadillo>

#include <glog/logging.h>

namespace stdmMf {


template <typename State>
FiniteQfnFeatures<State>::FiniteQfnFeatures(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<State> > & model,
        const std::shared_ptr<Features<State> > & features,
        const uint32_t & look_ahead)
    : network_(network), num_nodes_(this->network_->size()), model_(model),
      features_(features), look_ahead_(look_ahead),
      coef_(look_ahead_,
              std::vector<double>(this->features_->num_features(), 0.0)){

    CHECK_GT(this->look_ahead_, 0);

    this->model_->rng(this->rng());
    this->features_->rng(this->rng());
}


template <typename State>
FiniteQfnFeatures<State>::FiniteQfnFeatures(
        const FiniteQfnFeatures<State> & other)
    : network_(other.network_->clone()),  num_nodes_(other.num_nodes_),
      model_(other.model_->clone()), features_(other.features_->clone()),
      look_ahead_(other.look_ahead_), coef_(other.coef_),
      last_feat_(other.last_feat_) {

    this->model_->rng(this->rng());
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
    this->model_->est_par(history, curr_state);
    const std::vector<Transition<State> > sim_data(
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

    for (uint32_t i = 0; i < this->look_ahead_; ++i) {
        features.push_back(njm::linalg::dot_a_and_b(
                        this->last_feat_, this->coef_.at(i)));
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
    for (uint32_t i = 0; i < this->look_ahead_; ++i) {
        feat.at(i + 1) = njm::linalg::dot_a_and_b(
                this->last_feat_, this->coef_.at(i));
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
std::vector<Transition<State> > FiniteQfnFeatures<State>::generate_data(
        const uint32_t & num_episodes, const uint32_t & num_obs_per_episode) {
    // setup return container
    std::vector<Transition<State> > transitions;
    transitions.reserve(num_episodes * num_obs_per_episode);

    // system
    System<State> s(this->network_, this->model_);
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
            transitions.emplace_back(curr_state, trt_bits, next_state);
        }
    }

    return transitions;
}


template <typename State>
void FiniteQfnFeatures<State>::fit_q_functions(
        const std::vector<Transition<State> > & obs) {
    const uint32_t num_obs(obs.size());
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
            state_trt_test.emplace_back(obs.at(i).curr_state,
                    obs.at(i).curr_trt_bits);
            outcomes_test.push_back(
                    - static_cast<float>(obs.at(i).next_state.inf_bits.count())
                    / static_cast<float>(this->num_nodes_));
        } else {
            // train sets
            train_index.push_back(i);
            state_trt_train.emplace_back(obs.at(i).curr_state,
                    obs.at(i).curr_trt_bits);
            outcomes_train.push_back(
                    - static_cast<float>(obs.at(i).next_state.inf_bits.count())
                    / static_cast<float>(this->num_nodes_));
        }
    }
    const uint32_t num_train(state_trt_train.size());
    const uint32_t num_test(state_trt_test.size());

    // train first neural network using outcomes
    this->fit_model(0, state_trt_train, outcomes_train,
            state_trt_test, outcomes_test);

    // train remaining neural networks using outcome plus arg max of
    // previous networks
    for (uint32_t i = 1; i < this->look_ahead_; ++i) {
        std::vector<double> outcome_plus_max_train, outcome_plus_max_test;
        outcome_plus_max_train.reserve(num_obs);
        outcome_plus_max_test.reserve(num_obs);

        SweepAgent<State> sa(this->network_, this->features_,
                this->coef_.at(i - 1), njm::linalg::dot_a_and_b, 2, true);

        // find max for next states and add to outcomes
        for (uint32_t j = 0; j < num_train; ++j) {
            const auto & next_state(obs.at(train_index.at(j)).next_state);
            const boost::dynamic_bitset<> trt_bits(sa.apply_trt(next_state));

            const std::vector<double> feat(
                    this->features_->get_features(next_state, trt_bits));

            outcome_plus_max_train.push_back(outcomes_train.at(j)
                    + njm::linalg::dot_a_and_b(feat, this->coef_.at(i - 1)));
        }

        // find max for next states and add to outcomes
        for (uint32_t j = 0; j < num_test; ++j) {
            const auto & next_state(obs.at(test_index.at(j)).next_state);
            const boost::dynamic_bitset<> trt_bits(sa.apply_trt(next_state));

            const std::vector<double> feat(
                    this->features_->get_features(next_state, trt_bits));

            outcome_plus_max_test.push_back(outcomes_test.at(j)
                    + njm::linalg::dot_a_and_b(feat, this->coef_.at(i - 1)));
        }

        // train neural network
        this->fit_model(i, state_trt_train, outcome_plus_max_train,
                state_trt_test, outcome_plus_max_test);
    }
}


template <typename State>
void FiniteQfnFeatures<State>::fit_model(const uint32_t & model_index,
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
    for (uint32_t i = 0; i < lambda_vals.size(); ++i) {
        const double & lambda(lambda_vals.at(i));

        // cholesky on xtx + lamba * I
        const arma::mat r_train(arma::chol(xtx_train
                        + arma::eye(num_features, num_features) * lambda));

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
        coef_.at(model_index).resize(num_features);
        auto coef_it(coef_.at(model_index).begin());
        auto beta_it(best_beta.begin());
        for (uint32_t i = 0; i < num_features; ++i, ++coef_it, ++beta_it) {
            *coef_it = *beta_it;
        }
    }
}


template <typename State>
uint32_t FiniteQfnFeatures<State>::num_features() const {
    return this->look_ahead_ + 1;
}


template <typename State>
void FiniteQfnFeatures<State>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->njm::tools::RngClass::rng(rng);
    this->model_->rng(rng);
    this->features_->rng(rng);
}


template class FiniteQfnFeatures<InfState>;
template class FiniteQfnFeatures<InfShieldState>;

} // namespace stdmMf
