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

        combined.push_back(combined_val);
    }
    return combined;
}


std::vector<double> add_a_and_b(const std::vector<double> & a,
        const std::vector<double> & b) {
    const uint32_t a_size = a.size();
    CHECK_EQ(a_size, b.size());
    std::vector<double> res;
    res.reserve(a_size);
    for (uint32_t i = 0; i < a_size; ++i) {
        res.push_back(a.at(i) + b.at(i));
    }
    return res;
}

std::vector<double> add_a_and_b(const std::vector<double> & a,
        const double & b) {
    const uint32_t a_size = a.size();
    std::vector<double> res;
    res.reserve(a_size);
    for (uint32_t i = 0; i < a_size; ++i) {
        res.push_back(a.at(i) + b);
    }
    return res;
}

void add_b_to_a(std::vector<double> & a, const std::vector<double> & b) {
    const uint32_t a_size = a.size();
    CHECK_EQ(a_size, b.size());
    for (uint32_t i = 0; i < a_size; ++i) {
        a.at(i) += b.at(i);
    }
}

void add_b_to_a(std::vector<double> & a, const double & b) {
    const uint32_t a_size = a.size();
    for (uint32_t i = 0; i < a_size; ++i) {
        a.at(i) += b;
    }
}

std::vector<double> mult_a_and_b(const std::vector<double> & a,
        const std::vector<double> & b) {
    const uint32_t a_size = a.size();
    CHECK_EQ(a_size, b.size());
    std::vector<double> res;
    res.reserve(a_size);
    for (uint32_t i = 0; i < a_size; ++i) {
        res.push_back(a.at(i) * b.at(i));
    }
    return res;
}

std::vector<double> mult_a_and_b(const std::vector<double> & a,
        const double & b) {
    const uint32_t a_size = a.size();
    std::vector<double> res;
    res.reserve(a_size);
    for (uint32_t i = 0; i < a_size; ++i) {
        res.push_back(a.at(i) * b);
    }
    return res;
}

void mult_b_to_a(std::vector<double> & a, const std::vector<double> & b) {
    const uint32_t a_size = a.size();
    CHECK_EQ(a_size, b.size());
    for (uint32_t i = 0; i < a_size; ++i) {
        a.at(i) *= b.at(i);
    }
}

void mult_b_to_a(std::vector<double> & a, const double & b) {
    const uint32_t a_size = a.size();
    for (uint32_t i = 0; i < a_size; ++i) {
        a.at(i) *= b;
    }
}

double dot_a_and_b(const std::vector<double> & a,
        const std::vector<double> & b) {
    const uint32_t size = a.size();
    CHECK_EQ(size, b.size());

    double dot = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
        dot += a.at(i) * b.at(i);
    }

    return dot;
}

double l2_norm_sq(const std::vector<double> & a) {
    const uint32_t size = a.size();

    double norm = 0.0;
    for (uint32_t i = 0; i < size; ++i) {
        norm += a.at(i) * a.at(i);
    }

    return norm;
}

double l2_norm(const std::vector<double> & a) {
    return std::sqrt(l2_norm_sq(a));
}

void recip(std::vector<double> & a) {
    std::for_each(a.begin(), a.end(),
            [] (double & x) {
                x = 1.0 / x;
            });
}

std::vector<double> recip_of(const std::vector<double> & a) {
    std::vector<double> a_recip;
    const uint32_t size = a.size();
    a_recip.reserve(size);
    for (uint32_t i = 0; i < size; ++i) {
        a_recip.push_back(1.0 / a.at(i));
    }
}



} // namespace stdmMf
