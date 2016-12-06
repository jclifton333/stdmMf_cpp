#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>
#include <memory>
#include <cstdint>

namespace stdmMf {


class Rng {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis;
    uint32_t seed;

public:
    Rng();

    // set the random seed
    void set_seed(const uint32_t seed);

    // get the random seed
    uint32_t get_seed() const;

    // generate random uniform between [0,1)
    double runif_01();

    // generate random uniform between [a,b)
    double runif_ab(const double a, const double b);

    // generate random uniform integer between [a,b) inclusive
    int rint_ab(const int a, const int b);

    // random sample from range without replacement
    std::vector<int> sample_range(const int a, const int b, const int n);
};


class RngClass {
protected:
    std::shared_ptr<Rng> rng;

    RngClass();

public:
    void set_rng(std::shared_ptr<Rng> rng);
    void set_seed(uint32_t seed);
    uint32_t get_seed() const;
};


} // namespace stdmMf


#endif // RANDOM_HPP
