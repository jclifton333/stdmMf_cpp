#include "utilities.hpp"
#include <glog/logging.h>


namespace stdmMf {


std::vector<uint32_t> active_set(const boost::dynamic_bitset<> & bs) {
    std::vector<uint32_t> active;
    const uint32_t size = bs.size();
    uint32_t active_ind = bs.find_first();
    while (active_ind < size) {
        active.push_back(active_ind);
        active_ind = bs.find_next(active_ind);
    }
    return active;
}

std::vector<uint32_t> inactive_set(const boost::dynamic_bitset<> & bs) {
    std::vector<uint32_t> inactive;
    const uint32_t size = bs.size();
    uint32_t active_ind = bs.find_first();
    for (uint32_t i = 0; i < size; ++i) {
        if (i == active_ind) {
            active_ind = bs.find_next(active_ind);
        } else {
            inactive.push_back(i);
        }
    }
    return inactive;
}

std::pair<std::vector<uint32_t>,
          std::vector<uint32_t>
          > both_sets(const boost::dynamic_bitset<> & bs) {
    std::pair<std::vector<uint32_t>,
              std::vector<uint32_t> > both;
    const uint32_t size = bs.size();
    uint32_t active_ind = bs.find_first();
    for (uint32_t i = 0; i < size; ++i) {
        if (i == active_ind) {
            both.first.push_back(i);
            active_ind = bs.find_next(active_ind);
        } else {
            both.second.push_back(i);
        }
    }
    return both;
}

std::vector<uint32_t> combine_sets(const boost::dynamic_bitset<> & one,
        const boost::dynamic_bitset<> & two) {
    const uint32_t size = one.size();
    CHECK_EQ(size, two.size());
    std::vector<uint32_t> combined;
    uint32_t one_active = one.find_first();
    uint32_t two_active = two.find_first();
    for (uint32_t i = 0; i < size; ++i) {
        uint32_t combined_val;
        if (i == one_active) {
            combined_val = 2;
            one_active = one.find_next(one_active);
        } else {
            combined_val = 0;
        }

        if (i == two_active) {
            ++combined_val;
            two_active = two.find_next(two_active);
        }
    }
    return combined;
}



} // namespace stdmMf
