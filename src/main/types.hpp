#ifndef TYPES_HPP
#define TYPES_HPP

#include <boost/dynamic_bitset.hpp>
#include <boost/serialization/strong_typedef.hpp>

namespace stdmMf {

struct InfAndTrt {
    boost::dynamic_bitset<> inf_bits;
    boost::dynamic_bitset<> trt_bits;

    InfAndTrt(const boost::dynamic_bitset<> & inf_bits,
            const boost::dynamic_bitset<> & trt_bits);
};

struct Transition {
    boost::dynamic_bitset<> curr_inf_bits;
    boost::dynamic_bitset<> curr_trt_bits;
    boost::dynamic_bitset<> next_inf_bits;

    Transition(const boost::dynamic_bitset<> & curr_inf_bits,
            const boost::dynamic_bitset<> & curr_trt_bits,
            const boost::dynamic_bitset<> & next_inf_bits);

    static std::vector<Transition> from_sequence(
            const std::vector<InfAndTrt> & sequence);
    static std::vector<Transition> from_sequence(
            const std::vector<InfAndTrt> & sequence,
            const boost::dynamic_bitset<> & final_inf);
};


} // namespace stdmMf


#endif // TYPES_HPP
