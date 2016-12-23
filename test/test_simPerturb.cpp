#include <gtest/gtest.h>
#include <glog/logging.h>

#include "simPerturb.hpp"
#include "random.hpp"

#include <chrono>
#include <thread>

namespace stdmMf {

TEST(TestSimPerturb, IndependentQuadratic) {
    Rng rng;
    const uint32_t n = 5;
    const std::vector<double> truth = {1., 2., 3., 4., 5.};
    auto f = [&n, &rng, &truth] (const std::vector<double> & par,
            void * const data)
        {
            double res = 0.0;
            for (uint32_t i = 0; i < n; ++i) {
                const double rand = rng.rnorm(truth.at(i), 1);
                res += (rand - par.at(i))*(rand - par.at(i));
            }
            return res;
        };

    std::vector<double> start;
    for (uint32_t i = 0; i < n; ++i) {
        start.push_back(rng.rnorm(-10, 1));
    }
    SimPerturb sp(f, start, NULL, 2.0, 0.15, 1.0, 1, 0.8, 0.0001);

    Optim::ErrorCode ec;
    do {
        ec = sp.step();
    } while(ec == Optim::ErrorCode::CONTINUE);
    CHECK_EQ(ec, Optim::ErrorCode::SUCCESS);

    const std::vector<double> par = sp.par();
    for (uint32_t i = 0; i < n; ++i) {
        EXPECT_LT(std::abs((par.at(i) - truth.at(i)) / truth.at(i)) , 0.1);
    }
}



} // namespace stdmMf

int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
