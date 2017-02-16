#ifndef STATE_HPP
#define STATE_HPP

#include <boost/dyanmic_bitset.hpp>

namespace stdmMf {


struct State {
    boost::dynamci_bitset<> inf_bits;
    std::vector<double> resistance;
};


} // namespace stdmMf


#endif // STATE_HPP
