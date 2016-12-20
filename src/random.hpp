#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>
#include <memory>
#include <cstdint>

namespace stdmMf {


class Rng {
private:
    std::mt19937 gen;
    std::uniform_real_distribution<double> dis_runif_01;
    std::normal_distribution<double> dis_rnorm_01;
    uint32_t seed;

public:
    Rng();

    // set the random seed
    void set_seed(const uint32_t seed);

    // get the random seed
    uint32_t get_seed() const;

    // get the generator
    std::mt19937 & get_gen();

    // set the generator
    void set_gen(const std::mt19937 & gen);

    // generate random uniform between [0,1)
    double runif_01();

    // generate random normal
    double rnorm_01();

    // generate random normal N(mu, sigma^2)
    double rnorm(const double mu, const double sigma);

    // generate random uniform between [a,b)
    double runif(const double a, const double b);

    // generate random uniform integer between [a,b) inclusive
    int rint(const int a, const int b);

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
