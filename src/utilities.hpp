#ifndef UTILITIES_HPP
#define UTILITIES_HPP

#include <boost/dynamic_bitset.hpp>

namespace stdmMf {

std::vector<uint32_t> active_set(const boost::dynamic_bitset<> & bs);

std::vector<uint32_t> inactive_set(const boost::dynamic_bitset<> & bs);

std::pair<std::vector<uint32_t>,
          std::vector<uint32_t>
          > both_sets(const boost::dynamic_bitset<> & bs);

std::vector<uint32_t> combine_sets(const boost::dynamic_bitset<> & one,
        const boost::dynamic_bitset<> & two);

std::vector<double> add_a_and_b(const std::vector<double> & a,
        const std::vector<double> & b);

std::vector<double> add_a_and_b(const std::vector<double> & a,
        const double & b);

void add_b_to_a(std::vector<double> & a, const std::vector<double> & b);

void add_b_to_a(std::vector<double> & a, const double & b);

std::vector<double> mult_a_and_b(const std::vector<double> & a,
        const std::vector<double> & b);

std::vector<double> mult_a_and_b(const std::vector<double> & a,
        const double & b);

void mult_b_to_a(std::vector<double> & a, const std::vector<double> & b);

void mult_b_to_a(std::vector<double> & a, const double & b);

} // namespace stdmMf


#endif // UTILITIES_HPP
