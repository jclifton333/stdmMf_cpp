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
    ret_val_t get(const boost::dynamic_bitset<> & inf_bits);

    void put(const boost::dynamic_bitset<> & inf_bits,
            const ret_val_t & ret_val);
};

StateLookup<InfState, boost::dynamic_bitset<> >
policy_iteration(
        const std::shared_ptr<const Network> & network,
        const std::shared_ptr<const Network> & model);


} // namespace stdmMf


#endif // POL_VAL_ITERATION_HPP
