#include <glog/logging.h>
#include "random.hpp"

namespace stdmMf {


Rng::Rng()
    : gen(0), seed(0), dis_runif_01(0., 1.), dis_rnorm_01(0., 1.) {
}

void Rng::set_seed(const uint32_t seed) {
    this->gen.seed(seed);
}


uint32_t Rng::get_seed() const {
    return this->seed;
}

std::mt19937 & Rng::get_gen() {
    return this->gen;
}

void Rng::set_gen(const std::mt19937 & gen) {
    this->gen = gen;
}

double Rng::runif_01() {
    return dis_runif_01(gen);
}


double Rng::rnorm_01() {
    return dis_rnorm_01(gen);
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
    CHECK_LT(a, b);
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


RngClass::RngClass()
    : rng(new Rng()) {
}

void RngClass::set_rng(std::shared_ptr<Rng> rng) {
    this->rng = rng;
}

void RngClass::set_seed(const uint32_t seed) {
    this->rng->set_seed(seed);
}

uint32_t RngClass::get_seed() const {
    return this->rng->get_seed();
}


} // namespace stdmMf
