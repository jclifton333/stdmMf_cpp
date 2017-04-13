#include "finiteQfnFeatures.hpp"

#include <njm_cpp/info/project.hpp>

#include "system.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"

namespace stdmMf {


template <typename State>
FiniteQfnFeatures<State>::FiniteQfnFeatures(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<State> > & model,
        const uint32_t & look_ahead)
    : network_(network), model_(model), num_nodes_(this->network_->size()),
      look_ahead_(look_ahead) {
    CHECK_GT(this->look_ahead_, 0);

    for (uint32_t i = 0; i < this->look_ahead_; ++i) {
        this->nn_.emplace_back(
                (njm::info::project::PROJECT_ROOT_DIR
                        + "/src/prototxt/neural_network_model.protoxt"),
                (njm::info::project::PROJECT_ROOT_DIR
                        + "/src/prototxt/neural_network_solver.prototxt"),
                100, this->network_->size());
    }

    std::for_each(this->nn_.begin(), this->nn_.end(),
            [this] (NeuralNetwork<State> & nn) {
                nn.rng(this->rng());
            });
    this->model_->rng(this->rng());
}


template <typename State>
FiniteQfnFeatures<State>::FiniteQfnFeatures(
        const FiniteQfnFeatures<State> & other)
    : network_(other.network_->clone()), model_(other.model_->clone()),
      num_nodes_(other.num_nodes_), look_ahead_(other.look_ahead_),
      nn_(other.nn_) {

    std::for_each(this->nn_.begin(), this->nn_.end(),
            [this] (NeuralNetwork<State> & nn) {
                nn.rng(this->rng());
            });
    this->model_->rng(this->rng());
}


template <typename State>
void FiniteQfnFeatures<State>::update(const State & curr_state,
        const std::vector<StateAndTrt<State> > & history,
        const uint32_t & num_trt) {
    this->model_->est_par(history, curr_state);
    const std::vector<Transition<State> > sim_data(
            this->generate_data(100, 100));

    this->fit_q_functions(sim_data, num_trt);
}


template <typename State>
std::vector<double> FiniteQfnFeatures<State>::get_features(const State & state,
        const boost::dynamic_bitset<> & trt_bits) {
    std::vector<double> features;
    features.reserve(this->num_features());
    features.push_back(1.0);
    for (uint32_t i = 0; i < this->look_ahead_; ++i) {
        const double nn_value(this->nn_.at(i).eval(
                        StateAndTrt<State>(state, trt_bits)));

        features.push_back(nn_value);
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
    feat = this->get_features(state_new, trt_bits_new);
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
        const std::vector<Transition<State> > & obs,
        const uint32_t & num_trt) {
    const uint32_t num_obs(obs.size());
    std::vector<StateAndTrt<State> > state_trt_data;
    state_trt_data.reserve(num_obs);
    std::vector<double> outcomes;
    outcomes.reserve(num_obs);

    // outcomes from history except first observation
    for (uint32_t i = 0; i < num_obs; ++i) {
        state_trt_data.emplace_back(obs.at(i).curr_state,
                obs.at(i).curr_trt_bits);
        outcomes.push_back(
                static_cast<float>(obs.at(i).next_state.inf_bits.count())
                / -static_cast<float>(this->num_nodes_));
    }

    // train first neural network using outcomes
    this->nn_.at(0).train_data(state_trt_data, outcomes);

    // train remaining neural networks using outcome plus arg max of
    // previous networks
    for (uint32_t i = 1; i < this->look_ahead_; ++i) {
        std::vector<double> outcome_plus_max;
        outcome_plus_max.reserve(num_obs);
        // find max for next states and add to outcomes
        for (uint32_t j = 0; j < num_obs; ++j) {
            // TODO: need to calculate arg max of the neural network
            const std::pair<boost::dynamic_bitset<>, double> arg_max(
                    this->nn_.at(i - 1).sweep_max(
                            obs.at(j).next_state, num_trt));

            outcome_plus_max.push_back(outcomes.at(j) + arg_max.second);
        }

        // train neural network
        this->nn_.at(i).train_data(state_trt_data, outcome_plus_max);
    }

}


template <typename State>
void FiniteQfnFeatures<State>::rng(
        const std::shared_ptr<njm::tools::Rng> & rng) {
    this->RngClass::rng(rng);
    std::for_each(this->nn_.begin(), this->nn_.end(),
            [this] (NeuralNetwork<State> & nn) {
                nn.rng(this->rng());
            });
    this->model_->rng(rng);
}


template class FiniteQfnFeatures<InfState>;
template class FiniteQfnFeatures<InfShieldState>;

} // namespace stdmMf
