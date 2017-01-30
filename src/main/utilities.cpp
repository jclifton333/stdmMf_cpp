#include "utilities.hpp"
#include <glog/logging.h>

#include <cmath>
#include <cblas.h>

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

    return cblas_ddot(size, a.data(), 1, b.data(), 1);
}

std::vector<double> outer_a_and_b(const std::vector<double> & a,
        const std::vector<double> & b) {
    std::vector<double> outer;
    const uint32_t a_size = a.size();
    const uint32_t b_size = b.size();
    outer.reserve(a_size * b_size);
    for (uint32_t i = 0; i < a_size; ++i) {
        for (uint32_t j = 0; j < b_size; ++j) {
            outer.push_back(a.at(i) * b.at(j));
        }
    }
    return outer;
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
    return a_recip;
}

std::pair<double, double> mean_and_var(const std::vector<double> & vec) {
    const uint32_t size = vec.size();
    std::pair<double, double> stats(0., 0.);
    for (uint32_t i = 0; i < size; ++i) {
        const double temp_mean = stats.first;
        stats.first += (vec.at(i) - temp_mean) / (i + 1);
        stats.second += (vec.at(i) - temp_mean) * (vec.at(i) - stats.first);
    }

    if (size > 1) {
        stats.second = stats.second / (size - 1);
    }

    return stats;
}

static const unsigned char BitReverseTable256[] =
{
  0x00, 0x80, 0x40, 0xC0, 0x20, 0xA0, 0x60, 0xE0, 0x10, 0x90, 0x50, 0xD0, 0x30,
  0xB0, 0x70, 0xF0, 0x08, 0x88, 0x48, 0xC8, 0x28, 0xA8, 0x68, 0xE8, 0x18, 0x98,
  0x58, 0xD8, 0x38, 0xB8, 0x78, 0xF8, 0x04, 0x84, 0x44, 0xC4, 0x24, 0xA4, 0x64,
  0xE4, 0x14, 0x94, 0x54, 0xD4, 0x34, 0xB4, 0x74, 0xF4, 0x0C, 0x8C, 0x4C, 0xCC,
  0x2C, 0xAC, 0x6C, 0xEC, 0x1C, 0x9C, 0x5C, 0xDC, 0x3C, 0xBC, 0x7C, 0xFC, 0x02,
  0x82, 0x42, 0xC2, 0x22, 0xA2, 0x62, 0xE2, 0x12, 0x92, 0x52, 0xD2, 0x32, 0xB2,
  0x72, 0xF2, 0x0A, 0x8A, 0x4A, 0xCA, 0x2A, 0xAA, 0x6A, 0xEA, 0x1A, 0x9A, 0x5A,
  0xDA, 0x3A, 0xBA, 0x7A, 0xFA, 0x06, 0x86, 0x46, 0xC6, 0x26, 0xA6, 0x66, 0xE6,
  0x16, 0x96, 0x56, 0xD6, 0x36, 0xB6, 0x76, 0xF6, 0x0E, 0x8E, 0x4E, 0xCE, 0x2E,
  0xAE, 0x6E, 0xEE, 0x1E, 0x9E, 0x5E, 0xDE, 0x3E, 0xBE, 0x7E, 0xFE, 0x01, 0x81,
  0x41, 0xC1, 0x21, 0xA1, 0x61, 0xE1, 0x11, 0x91, 0x51, 0xD1, 0x31, 0xB1, 0x71,
  0xF1, 0x09, 0x89, 0x49, 0xC9, 0x29, 0xA9, 0x69, 0xE9, 0x19, 0x99, 0x59, 0xD9,
  0x39, 0xB9, 0x79, 0xF9, 0x05, 0x85, 0x45, 0xC5, 0x25, 0xA5, 0x65, 0xE5, 0x15,
  0x95, 0x55, 0xD5, 0x35, 0xB5, 0x75, 0xF5, 0x0D, 0x8D, 0x4D, 0xCD, 0x2D, 0xAD,
  0x6D, 0xED, 0x1D, 0x9D, 0x5D, 0xDD, 0x3D, 0xBD, 0x7D, 0xFD, 0x03, 0x83, 0x43,
  0xC3, 0x23, 0xA3, 0x63, 0xE3, 0x13, 0x93, 0x53, 0xD3, 0x33, 0xB3, 0x73, 0xF3,
  0x0B, 0x8B, 0x4B, 0xCB, 0x2B, 0xAB, 0x6B, 0xEB, 0x1B, 0x9B, 0x5B, 0xDB, 0x3B,
  0xBB, 0x7B, 0xFB, 0x07, 0x87, 0x47, 0xC7, 0x27, 0xA7, 0x67, 0xE7, 0x17, 0x97,
  0x57, 0xD7, 0x37, 0xB7, 0x77, 0xF7, 0x0F, 0x8F, 0x4F, 0xCF, 0x2F, 0xAF, 0x6F,
  0xEF, 0x1F, 0x9F, 0x5F, 0xDF, 0x3F, 0xBF, 0x7F, 0xFF
};


uint32_t reverse_bits(const uint32_t & x) {
    return (BitReverseTable256[x & 0xff] << 24)
        | (BitReverseTable256[(x >> 8) & 0xff] << 16)
        | (BitReverseTable256[(x >> 16) & 0xff] << 8)
        | (BitReverseTable256[(x >> 24) & 0xff]);
}

} // namespace stdmMf
