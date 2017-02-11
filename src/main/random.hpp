#ifndef RANDOM_HPP
#define RANDOM_HPP

#include <random>
#include <memory>
#include <cstdint>
#include <mutex>

namespace stdmMf {


class Rng {
private:
    std::mt19937 gen_;
    std::uniform_real_distribution<double> dis_runif_01_;
    std::normal_distribution<double> dis_rnorm_01_;
    uint32_t seed_;

    mutable std::mutex gen_mutex_;

public:
    Rng();

    // set the random seed
    void seed(const uint32_t seed);

    // get the random seed
    uint32_t seed() const;

    // get the generator
    std::mt19937 & gen();

    // set the generator
    void gen(const std::mt19937 & gen);

    // generate random uniform between [0,1)
    double runif_01();

    // generate random normal
    double rnorm_01();

    // generate random normal N(mu, sigma^2)
    double rnorm(const double mu, const double sigma);

    // generate random uniform between [a,b)
    double runif(const double a, const double b);

    // generate random uniform integer between [a,b)
    int rint(const int a, const int b);

    // random sample from range without replacement
    std::vector<int> sample_range(const int a, const int b, const int n);

    // shuffle
    void shuffle(std::vector<uint32_t> & x);
};


class RngClass {
protected:
    std::shared_ptr<Rng> rng_;

    RngClass();

    RngClass(const RngClass & other);

public:
    void rng(std::shared_ptr<Rng> rng);
    std::shared_ptr<Rng> rng();
    void seed(uint32_t seed);
    uint32_t seed() const;
};


} // namespace stdmMf


#endif // RANDOM_HPP
