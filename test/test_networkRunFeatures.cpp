#include <gtest/gtest.h>
#include <glog/logging.h>

#include <random>

#include <njm_cpp/tools/random.hpp>
#include "network.hpp"
#include "states.hpp"
#include "networkRunFeatures.hpp"

namespace stdmMf {


TEST(TestNetworkRunFeatures, TestFeaturesSimpleLen1) {
    // generate network
    NetworkInit init;
    init.set_dim_x(1);
    init.set_dim_y(1);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    NetworkRunFeatures<InfState> nrf(n, 1);

    ASSERT_EQ(nrf.num_features(), 1 + 3);

    boost::dynamic_bitset<> inf_bits(1);
    boost::dynamic_bitset<> trt_bits(1);

    std::vector<double> f;
    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 1);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);

    inf_bits.reset();
    trt_bits.reset();

    trt_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1);
    EXPECT_EQ(f.at(3), 0);

    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1);

    inf_bits.reset();
    trt_bits.reset();

    trt_bits.set(0);
    inf_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
}


TEST(TestNetworkRunFeatures, TestFeaturesSimpleLen2) {
    // generate network
    NetworkInit init;
    init.set_dim_x(1);
    init.set_dim_y(2);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    NetworkRunFeatures<InfState> nrf(n, 2);

    ASSERT_EQ(nrf.num_features(), 1 + 3 + 15);

    boost::dynamic_bitset<> inf_bits(2);
    boost::dynamic_bitset<> trt_bits(2);

    std::vector<double> f;
    // trt: (0, 0), inf: (0, 0)
    inf_bits.reset();
    trt_bits.reset();

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 2 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 1);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (1, 0), inf: (0, 0)
    inf_bits.reset();
    trt_bits.reset();

    trt_bits.set(0);


    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (0, 1), inf: (0, 0)
    inf_bits.reset();
    trt_bits.reset();

    trt_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (1, 1), inf: (0, 0)
    inf_bits.reset();
    trt_bits.reset();

    trt_bits.set();

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 2 / 2.);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (0, 0), inf: (1, 0)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 1);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (1, 0), inf: (1, 0)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set(0);
    trt_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 1);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (0, 1), inf: (1, 0)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set(0);
    trt_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 1 / 2.);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 1);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);


    // trt: (1, 1), inf: (1, 0)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set(0);
    trt_bits.set();

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 1);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);


    // trt: (0, 0), inf: (0, 1)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 1);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (1, 0), inf: (0, 1)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set(1);
    trt_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 1 / 2.);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 1);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (0, 1), inf: (0, 1)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set(1);
    trt_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 1);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (1, 1), inf: (0, 1)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set(1);
    trt_bits.set();

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 1);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    // trt: (0, 0), inf: (1, 1)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set();

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 2 / 2.);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 1);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);


    // trt: (1, 0), inf: (1, 1)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set();
    trt_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 1);
    EXPECT_EQ(f.at(18), 0);

    // trt: (0, 1), inf: (1, 1)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set();
    trt_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 1);

    // trt: (1, 1), inf: (1, 1)
    inf_bits.reset();
    trt_bits.reset();

    inf_bits.set();
    trt_bits.set();

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);
}


TEST(TestNetworkRunFeatures, TestFeaturesLen1) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    NetworkRunFeatures<InfState> nrf(n, 1);

    ASSERT_EQ(nrf.num_features(), 1 + 3);

    boost::dynamic_bitset<> inf_bits(9);
    boost::dynamic_bitset<> trt_bits(9);

    std::vector<double> f;
    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_NEAR(f.at(1), 9 / 9., 1e-12);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);

    inf_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_NEAR(f.at(1), 8 / 9., 1e-12);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_NEAR(f.at(3), 1 / 9., 1e-12);

    trt_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_NEAR(f.at(1), 7 / 9., 1e-12);
    EXPECT_NEAR(f.at(2), 1 / 9., 1e-12);
    EXPECT_NEAR(f.at(3), 1 / 9., 1e-12);

    inf_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_NEAR(f.at(1), 7 / 9., 1e-12);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_NEAR(f.at(3), 1 / 9., 1e-12);

    inf_bits.set();
    trt_bits.set();

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
}

TEST(TestNetworkRunFeatures, TestFeaturesLen2) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    NetworkRunFeatures<InfState> nrf(n, 2);

    ASSERT_EQ(nrf.num_features(), 1 + 3 + 15);

    boost::dynamic_bitset<> inf_bits(9);
    boost::dynamic_bitset<> trt_bits(9);

    std::vector<double> f;
    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_NEAR(f.at(1), 9 / 9., 1e-12);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_NEAR(f.at(4), 12 / 12., 1e-12);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    inf_bits.reset();
    trt_bits.reset();
    inf_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_NEAR(f.at(1), 8 / 9., 1e-12);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_NEAR(f.at(3), 1 / 9., 1e-12);
    // len 2
    EXPECT_NEAR(f.at(4), 10 / 12., 1e-12);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_NEAR(f.at(8), 2 / 12., 12e-12);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    inf_bits.reset();
    trt_bits.reset();
    trt_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_NEAR(f.at(1), 8 / 9., 1e-12);
    EXPECT_NEAR(f.at(2), 1 / 9., 1e-12);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_NEAR(f.at(4), 10 / 12., 1e-12);
    EXPECT_NEAR(f.at(5), 2 / 12., 1e-12);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    inf_bits.reset();
    trt_bits.reset();
    inf_bits.set(0);
    trt_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_NEAR(f.at(1), 8 / 9., 1e-12);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_NEAR(f.at(4), 10 / 12., 1e-12);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_NEAR(f.at(9), 2 / 12., 1e-12);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    inf_bits.reset();
    trt_bits.reset();
    inf_bits.set(0);
    trt_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_NEAR(f.at(1), 7 / 9., 1e-12);
    EXPECT_NEAR(f.at(2), 1 / 9., 1e-12);
    EXPECT_NEAR(f.at(3), 1 / 9., 1e-12);
    // len 2
    EXPECT_NEAR(f.at(4), 8 / 12., 1e-12);
    EXPECT_NEAR(f.at(5), 2 / 12., 1e-12);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_NEAR(f.at(8), 1 / 12., 1e-12);
    EXPECT_EQ(f.at(9), 0);
    EXPECT_NEAR(f.at(10), 1 / 12., 1e-12);
    EXPECT_EQ(f.at(11), 0);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);

    inf_bits.reset();
    trt_bits.reset();
    inf_bits.set(0);
    trt_bits.set(0);
    trt_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_NEAR(f.at(1), 7 / 9., 1e-12);
    EXPECT_NEAR(f.at(2), 1 / 9., 1e-12);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_NEAR(f.at(4), 8 / 12., 1e-12);
    EXPECT_NEAR(f.at(5), 2 / 12., 1e-12);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_NEAR(f.at(9), 1 / 12., 1e-12);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_NEAR(f.at(11), 1 / 12., 1e-12);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);
}


TEST(TestNetworkRunFeatures, UpdateFeatures) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    std::random_device device;
    const uint32_t seed = device();
    njm::tools::Rng rng;
    rng.seed(seed);

    NetworkRunFeatures<InfState> nrf_get(n, 3);
    NetworkRunFeatures<InfState> nrf_update(n, 3);
    for (uint32_t reps = 0; reps < 100; ++reps) {
        const uint32_t num_inf = rng.rint(0, n->size());
        const std::vector<int> inf_list =
            rng.sample_range(0, n->size(), num_inf);

        const uint32_t num_trt = rng.rint(0, n->size());
        const std::vector<int> trt_list =
            rng.sample_range(0, n->size(), num_trt);

        boost::dynamic_bitset<> inf_bits(n->size()), trt_bits(n->size());
        for (uint32_t i = 0; i < num_inf; ++i) {
            inf_bits.set(inf_list.at(i));
        }
        for (uint32_t i = 0; i < num_trt; ++i) {
            trt_bits.set(trt_list.at(i));
        }

        std::string inf_string, trt_string;
        boost::to_string(inf_bits, inf_string);
        boost::to_string(trt_bits, trt_string);


        std::vector<double> f_orig;

        // flip inf
        boost::dynamic_bitset<> inf_bits_flipped;
        for (uint32_t i = 0; i < n->size(); ++i) {
            inf_bits_flipped = inf_bits;
            inf_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(
                    inf_bits_flipped, trt_bits);


            // get features to reset masks properly
            f_orig = nrf_update.get_features(inf_bits,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, inf_bits_flipped, trt_bits,
                    inf_bits, trt_bits, f_upd_async);

            nrf_update.update_features(i, inf_bits_flipped, trt_bits, inf_bits,
                    trt_bits, f_upd);

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd.at(j), f_new.at(j), 1e-14)
                    << "Flipping inf failed for node " << i <<
                    " and feature " << j << " with seed " << seed << ".";
            }

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd_async.at(j), f_new.at(j), 1e-14)
                    << "Async flipping inf failed for node " << i <<
                    " and feature " << j << " with seed " << seed << ".";
            }
        }

        // get features again to reset paths properly
        f_orig = nrf_update.get_features(inf_bits,
                trt_bits);


        // flip trt
        boost::dynamic_bitset<> trt_bits_flipped;
        for (uint32_t i = 0; i < n->size(); ++i) {
            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(inf_bits,
                    trt_bits_flipped);

            // get features to reset masks properly
            f_orig = nrf_update.get_features(inf_bits,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, inf_bits, trt_bits_flipped,
                    inf_bits, trt_bits, f_upd_async);

            nrf_update.update_features(i, inf_bits, trt_bits_flipped, inf_bits,
                    trt_bits, f_upd);

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd.at(j), f_new.at(j), 1e-14)
                    << "Flipping inf failed for node " << i <<
                    " and feature " << j << " with seed " << seed << ".";
            }

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd_async.at(j), f_new.at(j), 1e-14)
                    << "Async flipping inf failed for node " << i <<
                    " and feature " << j << " with seed " << seed << ".";
            }
        }

        // flip both
        for (uint32_t i = 0; i < n->size(); ++i) {
            inf_bits_flipped = inf_bits;
            inf_bits_flipped.flip(i);

            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(
                    inf_bits_flipped, trt_bits_flipped);

            // get features to reset masks properly
            f_orig = nrf_update.get_features(inf_bits,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, inf_bits_flipped,
                    trt_bits_flipped, inf_bits, trt_bits, f_upd_async);

            nrf_update.update_features(i, inf_bits_flipped, trt_bits_flipped,
                    inf_bits, trt_bits, f_upd);

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd.at(j), f_new.at(j), 1e-14)
                    << "Flipping inf failed for node " << i <<
                    " and feature " << j << " with seed " << seed << ".";
            }

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd_async.at(j), f_new.at(j), 1e-14)
                    << "Async flipping inf failed for node " << i <<
                    " and feature " << j << " with seed " << seed << ".";
            }
        }
    }
}




TEST(TestNetworkRunFeatures, Copy) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    std::random_device device;
    const uint32_t seed = device();
    njm::tools::Rng rng;
    rng.seed(seed);

    NetworkRunFeatures<InfState> nrf_get(n, 3);
    NetworkRunFeatures<InfState> nrf_update(n, 3);

    NetworkRunFeatures<InfState> nrf_get_cpy(nrf_get);
    NetworkRunFeatures<InfState> nrf_update_cpy(nrf_update);

    EXPECT_EQ(nrf_get.num_features(), nrf_get_cpy.num_features());
    EXPECT_EQ(nrf_update.num_features(),
            nrf_update_cpy.num_features());

    for (uint32_t reps = 0; reps < 100; ++reps) {
        const uint32_t num_inf = rng.rint(0, n->size());
        const std::vector<int> inf_list =
            rng.sample_range(0, n->size(), num_inf);

        const uint32_t num_trt = rng.rint(0, n->size());
        const std::vector<int> trt_list =
            rng.sample_range(0, n->size(), num_trt);

        boost::dynamic_bitset<> inf_bits(n->size()), trt_bits(n->size());
        for (uint32_t i = 0; i < num_inf; ++i) {
            inf_bits.set(inf_list.at(i));
        }
        for (uint32_t i = 0; i < num_trt; ++i) {
            trt_bits.set(trt_list.at(i));
        }

        std::string inf_string, trt_string;
        boost::to_string(inf_bits, inf_string);
        boost::to_string(trt_bits, trt_string);


        std::vector<double> f_orig;
        std::vector<double> f_orig_cpy;

        // flip inf
        boost::dynamic_bitset<> inf_bits_flipped;
        for (uint32_t i = 0; i < n->size(); ++i) {
            inf_bits_flipped = inf_bits;
            inf_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(
                    inf_bits_flipped, trt_bits);
            const std::vector<double> f_new_cpy = nrf_get_cpy.get_features(
                    inf_bits_flipped, trt_bits);


            // get features to reset masks properly
            f_orig = nrf_update.get_features(inf_bits,
                    trt_bits);
            f_orig_cpy = nrf_update_cpy.get_features(inf_bits,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);
            std::vector<double> f_upd_cpy(f_orig_cpy);
            std::vector<double> f_upd_async_cpy(f_orig_cpy);

            nrf_update.update_features_async(i, inf_bits_flipped, trt_bits,
                    inf_bits, trt_bits, f_upd_async);

            nrf_update.update_features(i, inf_bits_flipped, trt_bits, inf_bits,
                    trt_bits, f_upd);

            nrf_update_cpy.update_features_async(i, inf_bits_flipped, trt_bits,
                    inf_bits, trt_bits, f_upd_async_cpy);

            nrf_update_cpy.update_features(i, inf_bits_flipped, trt_bits,
                    inf_bits, trt_bits, f_upd_cpy);

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_new.at(j), f_new_cpy.at(j), 1e-14)
                    << "Copy failed for new";
            }

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd.at(j), f_upd_cpy.at(j), 1e-14)
                    << "Copy failed for synchronous update";
            }

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd_async.at(j), f_upd_async_cpy.at(j), 1e-14)
                    << "Copy failed for synchronous update";
            }
        }

        // get features again to reset paths properly
        f_orig = nrf_update.get_features(inf_bits,
                trt_bits);

        f_orig_cpy = nrf_update_cpy.get_features(inf_bits,
                trt_bits);


        // flip trt
        boost::dynamic_bitset<> trt_bits_flipped;
        for (uint32_t i = 0; i < n->size(); ++i) {
            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(inf_bits,
                    trt_bits_flipped);

            const std::vector<double> f_new_cpy = nrf_get_cpy.get_features(
                    inf_bits, trt_bits_flipped);

            // get features to reset masks properly
            f_orig = nrf_update.get_features(inf_bits,
                    trt_bits);
            f_orig_cpy = nrf_update_cpy.get_features(inf_bits,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            std::vector<double> f_upd_cpy(f_orig_cpy);
            std::vector<double> f_upd_async_cpy(f_orig_cpy);

            nrf_update.update_features_async(i, inf_bits, trt_bits_flipped,
                    inf_bits, trt_bits, f_upd_async);

            nrf_update.update_features(i, inf_bits, trt_bits_flipped, inf_bits,
                    trt_bits, f_upd);

            nrf_update_cpy.update_features_async(i, inf_bits, trt_bits_flipped,
                    inf_bits, trt_bits, f_upd_async_cpy);

            nrf_update_cpy.update_features(i, inf_bits, trt_bits_flipped,
                    inf_bits, trt_bits, f_upd_cpy);

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_new.at(j), f_new_cpy.at(j), 1e-14)
                    << "Copy failed for new";
            }

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd.at(j), f_upd_cpy.at(j), 1e-14)
                    << "Copy failed for synchronous update";
            }

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd_async.at(j), f_upd_async_cpy.at(j), 1e-14)
                    << "Copy failed for synchronous update";
            }
        }

        // flip both
        for (uint32_t i = 0; i < n->size(); ++i) {
            inf_bits_flipped = inf_bits;
            inf_bits_flipped.flip(i);

            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(
                    inf_bits_flipped, trt_bits_flipped);

            const std::vector<double> f_new_cpy = nrf_get_cpy.get_features(
                    inf_bits_flipped, trt_bits_flipped);

            // get features to reset masks properly
            f_orig = nrf_update.get_features(inf_bits,
                    trt_bits);

            f_orig_cpy = nrf_update_cpy.get_features(inf_bits,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            std::vector<double> f_upd_cpy(f_orig_cpy);
            std::vector<double> f_upd_async_cpy(f_orig_cpy);

            nrf_update.update_features_async(i, inf_bits_flipped,
                    trt_bits_flipped, inf_bits, trt_bits, f_upd_async);

            nrf_update.update_features(i, inf_bits_flipped, trt_bits_flipped,
                    inf_bits, trt_bits, f_upd);

            nrf_update_cpy.update_features_async(i, inf_bits_flipped,
                    trt_bits_flipped, inf_bits, trt_bits, f_upd_async_cpy);

            nrf_update_cpy.update_features(i, inf_bits_flipped,
                    trt_bits_flipped, inf_bits, trt_bits, f_upd_cpy);

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_new.at(j), f_new_cpy.at(j), 1e-14)
                    << "Copy failed for new";
            }

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd.at(j), f_upd_cpy.at(j), 1e-14)
                    << "Copy failed for synchronous update";
            }

            for (uint32_t j = 0; j < nrf_get.num_features(); ++j) {
                EXPECT_NEAR(f_upd_async.at(j), f_upd_async_cpy.at(j), 1e-14)
                    << "Copy failed for synchronous update";
            }
        }
    }
}



} // namespace stdmMf


int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
