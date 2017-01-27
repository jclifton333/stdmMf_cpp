#ifndef TYPES_HPP
#define TYPES_HPP

#include <boost/dynamic_bitset.hpp>
#include <boost/serialization/strong_typedef.hpp>

namespace stdmMf {


using BitsetPair = std::pair<boost::dynamic_bitset<>,
                             boost::dynamic_bitset<> >;


} // namespace stdmMf


#endif // TYPES_HPP
