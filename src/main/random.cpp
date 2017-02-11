#include "random.hpp"
#include <glog/logging.h>
#include <algorithm>

namespace stdmMf {


Rng::Rng()
    : gen_(0), seed_(0), dis_runif_01_(0., 1.), dis_rnorm_01_(0., 1.) {
}

void Rng::seed(const uint32_t seed) {
    std::lock_guard<std::mutex> lock(this->gen_mutex_);
    this->seed_ = seed;
    this->gen_.seed(seed);
}


uint32_t Rng::seed() const {
    std::lock_guard<std::mutex> lock(this->gen_mutex_);
    return this->seed_;
}

std::mt19937 & Rng::gen() {
    std::lock_guard<std::mutex> lock(this->gen_mutex_);
    return this->gen_;
}

void Rng::gen(const std::mt19937 & gen) {
    std::lock_guard<std::mutex> lock(this->gen_mutex_);
    this->gen_ = gen;
}

double Rng::runif_01() {
    std::lock_guard<std::mutex> lock(this->gen_mutex_);
    return this->dis_runif_01_(this->gen_);
}


double Rng::rnorm_01() {
    std::lock_guard<std::mutex> lock(this->gen_mutex_);
    return this->dis_rnorm_01_(this->gen_);
}

double Rng::rnorm(const double mu, const double sigma) {
    return this->rnorm_01() * sigma + mu;
}

double Rng::runif(const double a, const double b) {
    CHECK_LT(a, b);
    return this->runif_01() * (b - a) + a;
}


int Rng::rint(const int a, const int b) {
    CHECK_LT(a, b);
    return static_cast<int>(this->runif_01() * (b - a)) + a;
}


std::vector<int> Rng::sample_range(const int a, const int b, const int n) {
    CHECK_LT(a, b) << "a = " << a << ", b = " << b;
    const uint32_t num_vals = b - a;
    CHECK_LT(n, num_vals); // can't sample more than what's there
    std::vector<int> choices;
    for (int i = a; i < b; ++i) {
        choices.push_back(i);
    }

    std::vector<int> values;
    for (uint32_t i = 0; i < n; ++i) {
        // sample an index
        const uint32_t ind = this->rint(0, num_vals - i);

        // record the value
        values.push_back(choices.at(ind));

        // swap values
        choices.at(ind) = choices.at(num_vals - i - 1);
    }

    return values;
}

void Rng::shuffle(std::vector<uint32_t> & x) {
    std::lock_guard<std::mutex> lock(this->gen_mutex_);
    std::shuffle(x.begin(), x.end(), this->gen_);
}


RngClass::RngClass()
    : rng_(new Rng()) {
}

RngClass::RngClass(const RngClass & other)
    : RngClass() {
    this->seed(other.seed());
}

void RngClass::rng(std::shared_ptr<Rng> rng) {
    this->rng_ = rng;
}

std::shared_ptr<Rng> RngClass::rng() {
    return this->rng_;
}

void RngClass::seed(const uint32_t seed) {
    this->rng_->seed(seed);
}

uint32_t RngClass::seed() const {
    return this->rng_->seed();
}


} // namespace stdmMf
