#include "polValIteration.hpp"

#include <glog/logging.h>

#include <numeric>
#include <iterator>


namespace stdmMf {


template <typename ret_val_t>
ret_val_t StateLookup<InfState, ret_val_t>::get(
        const boost::dynamic_bitset<> & inf_bits) const {
    return this->lookup_.at(inf_bits);
}

template <typename ret_val_t>
void StateLookup<InfState, ret_val_t>::put(
        const boost::dynamic_bitset<> & inf_bits,
        const ret_val_t & ret_val) {
    this->lookup_[inf_bits] = ret_val;
}


template class StateLookup<InfState, boost::dynamic_bitset<> >;
template class StateLookup<InfState, double >;


std::pair<std::vector<double>, std::vector<double> > trans_and_reward(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<InfState> > & model,
        const StateLookup<InfState, boost::dynamic_bitset<> > & policy) {
    CHECK_LT(network->size(), 32);
    const uint32_t dim = (1 << network->size());
    std::vector<double> trans(dim * dim, 0.0);
    std::vector<double> reward(dim, 0.0);
    for (uint32_t i = 0; i < dim; ++i) { // starting infection
        const boost::dynamic_bitset<> curr_inf_bits(network->size(), i);

        const std::vector<double> probs = model->probs(curr_inf_bits,
                policy.get(curr_inf_bits));

        double sum_inf = 0.0;
        for (uint32_t k = 0; k < network->size(); ++k) {
            if (curr_inf_bits.test(k)) {
                sum_inf -= 1.0 - probs.at(k);
            } else {
                sum_inf -= probs.at(k);
            }
        }
        reward.at(i) = sum_inf / network->size();

        for (uint32_t j = 0; j < dim; ++j) { // ending infection
            const boost::dynamic_bitset<> next_inf_bits(network->size(), j);

            const boost::dynamic_bitset<> diff = curr_inf_bits ^ next_inf_bits;

                        double log_prob = 0.0;

            for (uint32_t k = 0; k < network->size(); ++k) {
                if (diff.test(k)) {
                    log_prob += std::log(probs.at(k));
                } else {
                    log_prob += std::log(1.0 - probs.at(k));
                }
            }

            CHECK(std::isfinite(log_prob));
            trans.at(i * dim + j) = std::exp(log_prob);
        }
    }

    return std::make_pair(trans, reward);
}


std::pair<std::vector<double>, double> trans_and_reward(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<InfState> > & model,
        const boost::dynamic_bitset<> & curr_inf_bits,
        const boost::dynamic_bitset<> & trt_bits) {
    CHECK_LT(network->size(), 32);
    const uint32_t dim = (1 << network->size());
    std::vector<double> trans(dim, 0.0);
    double reward(0.0);
    const std::vector<double> probs = model->probs(curr_inf_bits,
            trt_bits);

    double sum_inf = 0.0;
    for (uint32_t k = 0; k < network->size(); ++k) {
        if (curr_inf_bits.test(k)) {
            sum_inf -= 1.0 - probs.at(k);
        } else {
            sum_inf -= probs.at(k);
        }
    }
    reward = sum_inf / network->size();

    for (uint32_t j = 0; j < dim; ++j) { // ending infection
        const boost::dynamic_bitset<> next_inf_bits(network->size(), j);

        const boost::dynamic_bitset<> diff = curr_inf_bits ^ next_inf_bits;

        double log_prob = 0.0;

        for (uint32_t k = 0; k < network->size(); ++k) {
            if (diff.test(k)) {
                log_prob += std::log(probs.at(k));
            } else {
                log_prob += std::log(1.0 - probs.at(k));
            }
        }

        CHECK(std::isfinite(log_prob));
        trans.at(j) = std::exp(log_prob);
    }

    return std::make_pair(trans, reward);
}


std::vector<double> value_iteration(
        const StateLookup<InfState, boost::dynamic_bitset<> > & policy,
        const double & gamma,
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<InfState> > & model) {

    const uint32_t max_inf_bits = (1 << network->size());
    std::vector<double> value(max_inf_bits, 0.0);

    const auto tr = trans_and_reward(network, model, policy);
    const std::vector<double> & trans = tr.first;
    const std::vector<double> & reward = tr.second;

    CHECK_EQ(trans.size(), max_inf_bits * max_inf_bits);
    CHECK_EQ(reward.size(), max_inf_bits);

    std::vector<double>::const_iterator it;

    double diff, new_val;

    bool converged = false;
    while (!converged) {
        diff = 0.0;
        for (uint32_t i = 0; i < max_inf_bits; ++i) {
            it = trans.begin();
            std::advance(it, i * network->size());

            new_val = reward.at(i) + gamma * std::inner_product(
                    value.begin(), value.end(), it, 0.0);
            diff += (value.at(i) - new_val) * (value.at(i) - new_val);
            value.at(i) = new_val;
        }

        diff = std::sqrt(diff);
        if (diff < 1e-8) {
            converged = true;
        }
    }

    return value;
}



StateLookup<InfState, boost::dynamic_bitset<> > policy_iteration(
        const std::shared_ptr<const Network> network,
        const std::shared_ptr<Model<InfState> > model,
        const double & gamma,
        const uint32_t & num_trt) {
    StateLookup<InfState, boost::dynamic_bitset<> > policy_lookup;

    CHECK_LE(num_trt, network->size());
    CHECK_GT(num_trt, 0);

    // initialize lookups
    CHECK_LT(network->size() * 2, 32);
    const uint32_t max_inf_bits = (1 << network->size());
    for (uint32_t i = 0; i < max_inf_bits; ++i) {
        const boost::dynamic_bitset<> inf_bits(network->size(), i);
        const boost::dynamic_bitset<> trt_bits(
                network->size(), 0);
        policy_lookup.put(inf_bits, trt_bits);
    }

    bool converged = false;
    while (!converged) {
        converged = true;

        const std::vector<double> val = value_iteration(policy_lookup, gamma,
                network, model);

        for (uint32_t i = 0; i < max_inf_bits; ++i) {
            const boost::dynamic_bitset<> curr_inf_bits(network->size(), i);

            std::string trt_bits_str(num_trt, '1');
            trt_bits_str.resize(network->size(), '0');
            do {
                const boost::dynamic_bitset<> trt_bits(trt_bits_str);

                const auto tr = trans_and_reward(network, model, curr_inf_bits,
                        trt_bits);
                const std::vector<double> & trans = tr.first;
                const double & reward = tr.second;

                const double new_value = reward + gamma * std::inner_product(
                        trans.begin(), trans.end(), val.begin(), 0.0);

                if (new_value > (val.at(i) + 1e-6)) {
                    policy_lookup.put(curr_inf_bits, trt_bits);
                    converged = false;
                }

            } while(std::prev_permutation(trt_bits_str.begin(),
                            trt_bits_str.end()));

        }
    }

    return policy_lookup;
}



} // namespace stdmMf
