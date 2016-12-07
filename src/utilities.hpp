#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <boost/dynamic_bitset.hpp>

namespace stdmMf {

std::vector<uint32_t> active_set(const boost::dynamic_bitset<> & bs);

std::vector<uint32_t> inactive_set(const boost::dynamic_bitset<> & bs);

std::pair<std::vector<uint32_t>,
          std::vector<uint32_t>
          > both_sets(const boost::dynamic_bitset<> & bs);




} // namespace stdmMf


#endif // UTILITIES_HPP
