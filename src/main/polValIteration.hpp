#ifndef POL_VAL_ITERATION_HPP
#define POL_VAL_ITERATION_HPP

#include "states.hpp"
#include "network.hpp"
#include "model.hpp"

#include <map>
#include <boost/dynamic_bitset.hpp>

namespace stdmMf {

template <typename State, typename ret_val_t>
class StateLookup;

template <typename ret_val_t>
class StateLookup<InfState, ret_val_t> {
protected:
    std::map<boost::dynamic_bitset<>, ret_val_t > lookup_;

public:
    ret_val_t get(const boost::dynamic_bitset<> & inf_bits) const ;

    void put(const boost::dynamic_bitset<> & inf_bits,
            const ret_val_t & ret_val);
};


std::pair<std::vector<double>, std::vector<double> > trans_and_reward(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<InfState> > & model,
        const StateLookup<InfState, boost::dynamic_bitset<> > & policy);

std::pair<std::vector<double>, double> trans_and_reward(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<InfState> > & model,
        const boost::dynamic_bitset<> & curr_inf_bits,
        const boost::dynamic_bitset<> & trt_bits);

std::vector<double> value_iteration(
        const StateLookup<InfState, boost::dynamic_bitset<> > & policy,
        const double & gamma,
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<InfState> > & model);

StateLookup<InfState, boost::dynamic_bitset<> >
policy_iteration(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<const Network> & model);


} // namespace stdmMf


#endif // POL_VAL_ITERATION_HPP
