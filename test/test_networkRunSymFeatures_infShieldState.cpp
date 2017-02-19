#include <gtest/gtest.h>
#include <glog/logging.h>

#include <random>

#include <njm_cpp/tools/random.hpp>
#include "network.hpp"
#include "states.hpp"
#include "networkRunSymFeatures.hpp"

namespace stdmMf {

TEST(TestNetworkRunSymFeatures, TestFeaturesSimpleLen1InfShieldState) {
    // generate network
    NetworkInit init;
    init.set_dim_x(1);
    init.set_dim_y(1);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    NetworkRunSymFeatures<InfShieldState> nrf(n, 1);

    ASSERT_EQ(nrf.num_features(), 1 + 7);

    InfShieldState state(1);
    boost::dynamic_bitset<> trt_bits(1);

    std::vector<double> f;
    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 1);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);

    state.reset();
    trt_bits.reset();

    trt_bits.set(0);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);

    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);

    state.reset();
    trt_bits.reset();

    trt_bits.set(0);
    state.inf_bits.set(0);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);


    state.reset();
    trt_bits.reset();

    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);



    state.reset();
    trt_bits.reset();

    trt_bits.set(0);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1);
    EXPECT_EQ(f.at(7), 0);

    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1);

    state.reset();
    trt_bits.reset();

    trt_bits.set(0);
    state.inf_bits.set(0);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);

}


TEST(TestNetworkRunSymFeatures, TestFeaturesSimpleLen2InfShieldState) {
    // generate network
    NetworkInit init;
    init.set_dim_x(1);
    init.set_dim_y(2);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    NetworkRunSymFeatures<InfShieldState> nrf(n, 2);

    ASSERT_EQ(nrf.num_features(), 1 + 7 + 35);

    InfShieldState state(2);
    boost::dynamic_bitset<> trt_bits(2);

    std::vector<double> f;

    // shield: (0, 0), inf: (0, 0), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 2 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (0, 0), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    trt_bits.set(0);


    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (0, 0), inf: (0, 0), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    trt_bits.set(1);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (0, 0), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    trt_bits.set();

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 2 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (0, 1), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (0, 1), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set(0);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (0, 1), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set(1);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (0, 0), inf: (0, 1), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set();

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (0, 0), inf: (1, 0), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (1, 0), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set(0);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (1, 0), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set(1);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (1, 0), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set();

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (1, 1), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 2 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (0, 0), inf: (1, 1), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set(0);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (1, 1), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set(1);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 0), inf: (1, 1), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set();

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 2 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);



    // shield: (0, 1), inf: (0, 0), trt: (0, 0)
    state.reset();
    trt_bits.reset();
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (0, 0), trt: (0, 1)
    state.reset();
    trt_bits.reset();
    state.shield.at(0) = 1;

    trt_bits.set(0);


    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 1);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (0, 1), inf: (0, 0), trt: (1, 0)
    state.reset();
    trt_bits.reset();
    state.shield.at(0) = 1;

    trt_bits.set(1);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 1);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (0, 0), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    trt_bits.set();
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 1);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (0, 1), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 1);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (0, 1), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set(0);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 1);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (0, 1), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set(1);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 1);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (0, 1), inf: (0, 1), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set();
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 1);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (0, 1), inf: (1, 0), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 1);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (1, 0), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set(0);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 1);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (1, 0), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set(1);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 1);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (1, 0), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set();
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 1);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (1, 1), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 1);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (0, 1), inf: (1, 1), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set(0);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 1);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (1, 1), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set(1);
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 1);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (0, 1), inf: (1, 1), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set();
    state.shield.at(0) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 1);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);




    // shield: (1, 0), inf: (0, 0), trt: (0, 0)
    state.reset();
    trt_bits.reset();
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (0, 0), trt: (0, 1)
    state.reset();
    trt_bits.reset();
    state.shield.at(1) = 1;

    trt_bits.set(0);


    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 1);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (1, 0), inf: (0, 0), trt: (1, 0)
    state.reset();
    trt_bits.reset();
    state.shield.at(1) = 1;

    trt_bits.set(1);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 1);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (0, 0), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    trt_bits.set();
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 1);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (0, 1), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 1);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (0, 1), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set(0);
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 1);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (0, 1), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set(1);
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 1);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (1, 0), inf: (0, 1), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set();
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 1);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (1, 0), inf: (1, 0), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 1);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (1, 0), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set(0);
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 1);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (1, 0), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set(1);
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 1 / 2.);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 1);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (1, 0), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set();
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 1 / 2.);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 1);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (1, 1), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 1);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (1, 0), inf: (1, 1), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set(0);
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 1);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (1, 1), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set(1);
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1 / 2.);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 1);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 0), inf: (1, 1), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set();
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 1 / 2.);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 1);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);





    // shield: (1, 1), inf: (0, 0), trt: (0, 0)
    state.reset();
    trt_bits.reset();
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 2 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 1);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 1), inf: (0, 0), trt: (0, 1)
    state.reset();
    trt_bits.reset();
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    trt_bits.set(0);


    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 1);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (1, 1), inf: (0, 0), trt: (1, 0)
    state.reset();
    trt_bits.reset();
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    trt_bits.set(1);

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 1);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 1), inf: (0, 0), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    trt_bits.set();
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 2 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 1);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 1), inf: (0, 1), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 1);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 1), inf: (0, 1), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set(0);
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 1);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 1), inf: (0, 1), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set(1);
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 1);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (1, 1), inf: (0, 1), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(0);
    trt_bits.set();
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 1);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);


    // shield: (1, 1), inf: (1, 0), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 1);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 1), inf: (1, 0), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set(0);
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 1);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 1), inf: (1, 0), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set(1);
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 1 / 2.);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 1);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 1), inf: (1, 0), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set(1);
    trt_bits.set();
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 1 / 2.);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 1);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

    // shield: (1, 1), inf: (1, 1), trt: (0, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 2 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 1);
    EXPECT_EQ(f.at(42), 0);


    // shield: (1, 1), inf: (1, 1), trt: (0, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set(0);
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 1);

    // shield: (1, 1), inf: (1, 1), trt: (1, 0)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set(1);
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 1 / 2.);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 1);

    // shield: (1, 1), inf: (1, 1), trt: (1, 1)
    state.reset();
    trt_bits.reset();

    state.inf_bits.set();
    trt_bits.set();
    state.shield.at(0) = 1;
    state.shield.at(1) = 1;

    f = nrf.get_features(state, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());

    // intercept
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 0);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    EXPECT_EQ(f.at(4), 0);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    // len 2
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
    EXPECT_EQ(f.at(19), 0);
    EXPECT_EQ(f.at(20), 0);
    EXPECT_EQ(f.at(21), 0);
    EXPECT_EQ(f.at(22), 0);
    EXPECT_EQ(f.at(23), 0);
    EXPECT_EQ(f.at(24), 0);
    EXPECT_EQ(f.at(25), 0);
    EXPECT_EQ(f.at(26), 0);
    EXPECT_EQ(f.at(27), 0);
    EXPECT_EQ(f.at(28), 0);
    EXPECT_EQ(f.at(29), 0);
    EXPECT_EQ(f.at(30), 0);
    EXPECT_EQ(f.at(31), 0);
    EXPECT_EQ(f.at(32), 0);
    EXPECT_EQ(f.at(33), 0);
    EXPECT_EQ(f.at(34), 0);
    EXPECT_EQ(f.at(35), 0);
    EXPECT_EQ(f.at(36), 0);
    EXPECT_EQ(f.at(37), 0);
    EXPECT_EQ(f.at(38), 0);
    EXPECT_EQ(f.at(39), 0);
    EXPECT_EQ(f.at(40), 0);
    EXPECT_EQ(f.at(41), 0);
    EXPECT_EQ(f.at(42), 0);

}


TEST(TestNetworkRunSymFeatures, UpdateFeaturesInfShieldState) {
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

    NetworkRunSymFeatures<InfShieldState> nrf_get(n, 3);
    NetworkRunSymFeatures<InfShieldState> nrf_update(n, 3);
    for (uint32_t reps = 0; reps < 100; ++reps) {
        const uint32_t num_inf = rng.rint(0, n->size());
        const std::vector<int> inf_list =
            rng.sample_range(0, n->size(), num_inf);

        const uint32_t num_trt = rng.rint(0, n->size());
        const std::vector<int> trt_list =
            rng.sample_range(0, n->size(), num_trt);

        InfShieldState state(n->size());
        boost::dynamic_bitset<> trt_bits(n->size());
        for (uint32_t i = 0; i < num_inf; ++i) {
            state.inf_bits.set(inf_list.at(i));
        }
        for (uint32_t i = 0; i < num_trt; ++i) {
            trt_bits.set(trt_list.at(i));
        }
        for (uint32_t i = 0; i < n->size(); ++i) {
            state.shield.at(i) = rng.rnorm_01();
        }


        std::vector<double> f_orig;

        // flip inf
        InfShieldState state_inf_bits_flipped(n->size());
        for (uint32_t i = 0; i < n->size(); ++i) {
            state_inf_bits_flipped = state;
            state_inf_bits_flipped.inf_bits.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(
                    state_inf_bits_flipped, trt_bits);


            // get features to reset masks properly
            f_orig = nrf_update.get_features(state,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, state_inf_bits_flipped,
                    trt_bits, state, trt_bits, f_upd_async);

            nrf_update.update_features(i, state_inf_bits_flipped, trt_bits,
                    state, trt_bits, f_upd);

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

        // flip trt
        boost::dynamic_bitset<> trt_bits_flipped;
        for (uint32_t i = 0; i < n->size(); ++i) {
            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(state,
                    trt_bits_flipped);

            // get features to reset masks properly
            f_orig = nrf_update.get_features(state,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, state, trt_bits_flipped,
                    state, trt_bits, f_upd_async);

            nrf_update.update_features(i, state, trt_bits_flipped, state,
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


        // flip inf and trt
        for (uint32_t i = 0; i < n->size(); ++i) {
            state_inf_bits_flipped = state;
            state_inf_bits_flipped.inf_bits.flip(i);

            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(
                    state_inf_bits_flipped, trt_bits_flipped);

            // get features to reset masks properly
            f_orig = nrf_update.get_features(state,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, state_inf_bits_flipped,
                    trt_bits_flipped, state, trt_bits, f_upd_async);

            nrf_update.update_features(i, state_inf_bits_flipped,
                    trt_bits_flipped, state, trt_bits, f_upd);

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


        // flip shield
        InfShieldState state_shield_flipped(n->size());
        for (uint32_t i = 0; i < n->size(); ++i) {
            state_shield_flipped = state;
            state_shield_flipped.shield.at(i) *= -1;
            const std::vector<double> f_new = nrf_get.get_features(
                    state_shield_flipped, trt_bits);


            // get features to reset masks properly
            f_orig = nrf_update.get_features(state,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, state_shield_flipped,
                    trt_bits, state, trt_bits, f_upd_async);

            nrf_update.update_features(i, state_shield_flipped, trt_bits,
                    state, trt_bits, f_upd);

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


        // flip inf and shield
        InfShieldState state_inf_bits_shield_flipped(n->size());
        for (uint32_t i = 0; i < n->size(); ++i) {
            state_inf_bits_shield_flipped = state;
            state_inf_bits_shield_flipped.inf_bits.flip(i);
            state_inf_bits_shield_flipped.shield.at(i) *= -1;
            const std::vector<double> f_new = nrf_get.get_features(
                    state_inf_bits_shield_flipped, trt_bits);


            // get features to reset masks properly
            f_orig = nrf_update.get_features(state,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, state_inf_bits_shield_flipped,
                    trt_bits, state, trt_bits, f_upd_async);

            nrf_update.update_features(i, state_inf_bits_shield_flipped,
                    trt_bits, state, trt_bits, f_upd);

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




        // flip shield and trt
        for (uint32_t i = 0; i < n->size(); ++i) {
            state_shield_flipped = state;
            state_shield_flipped.shield.at(i) *= -1;

            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(
                    state_shield_flipped, trt_bits_flipped);

            // get features to reset masks properly
            f_orig = nrf_update.get_features(state,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, state_shield_flipped,
                    trt_bits_flipped, state, trt_bits, f_upd_async);

            nrf_update.update_features(i, state_shield_flipped,
                    trt_bits_flipped, state, trt_bits, f_upd);

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


        // flip inf, shield, and trt
        for (uint32_t i = 0; i < n->size(); ++i) {
            state_inf_bits_shield_flipped = state;
            state_inf_bits_shield_flipped.inf_bits.flip(i);
            state_inf_bits_shield_flipped.shield.at(i) *= -1;

            trt_bits_flipped = trt_bits;
            trt_bits_flipped.flip(i);
            const std::vector<double> f_new = nrf_get.get_features(
                    state_inf_bits_shield_flipped, trt_bits_flipped);

            // get features to reset masks properly
            f_orig = nrf_update.get_features(state,
                    trt_bits);

            std::vector<double> f_upd(f_orig);
            std::vector<double> f_upd_async(f_orig);

            nrf_update.update_features_async(i, state_inf_bits_shield_flipped,
                    trt_bits_flipped, state, trt_bits, f_upd_async);

            nrf_update.update_features(i, state_inf_bits_shield_flipped,
                    trt_bits_flipped, state, trt_bits, f_upd);

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


} // namespace stdmMf


int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
