#include <gtest/gtest.h>
#include <glog/logging.h>

#include "network.hpp"
#include "networkRunFeatures.hpp"

namespace stdmMf {


TEST(TestNetworkRunFeatures, TestFeaturesLen1) {
    // generate network
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> n = Network::gen_network(init);

    NetworkRunFeatures nrf(n, 1);

    ASSERT_EQ(nrf.num_features(), 1 + 3);

    boost::dynamic_bitset<> inf_bits(9);
    boost::dynamic_bitset<> trt_bits(9);

    std::vector<double> f;
    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 9);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);

    inf_bits.set(0);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 8);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1);

    trt_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 7);
    EXPECT_EQ(f.at(2), 1);
    EXPECT_EQ(f.at(3), 1);

    inf_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    EXPECT_EQ(f.at(1), 7);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1);

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

    NetworkRunFeatures nrf(n, 2);

    ASSERT_EQ(nrf.num_features(), 1 + 3 + 15);

    boost::dynamic_bitset<> inf_bits(9);
    boost::dynamic_bitset<> trt_bits(9);

    std::vector<double> f;
    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 9);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 12);
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
    EXPECT_EQ(f.at(1), 8);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 1);
    // len 2
    EXPECT_EQ(f.at(4), 10);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 2);
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
    EXPECT_EQ(f.at(1), 8);
    EXPECT_EQ(f.at(2), 1);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 10);
    EXPECT_EQ(f.at(5), 2);
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
    EXPECT_EQ(f.at(1), 8);
    EXPECT_EQ(f.at(2), 0);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 10);
    EXPECT_EQ(f.at(5), 0);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 2);
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
    EXPECT_EQ(f.at(1), 7);
    EXPECT_EQ(f.at(2), 1);
    EXPECT_EQ(f.at(3), 1);
    // len 2
    EXPECT_EQ(f.at(4), 8);
    EXPECT_EQ(f.at(5), 2);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 1);
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

    inf_bits.reset();
    trt_bits.reset();
    inf_bits.set(0);
    trt_bits.set(0);
    trt_bits.set(1);

    f = nrf.get_features(inf_bits, trt_bits);
    ASSERT_EQ(f.size(), nrf.num_features());
    EXPECT_EQ(f.at(0), 1);
    // len 1
    EXPECT_EQ(f.at(1), 7);
    EXPECT_EQ(f.at(2), 1);
    EXPECT_EQ(f.at(3), 0);
    // len 2
    EXPECT_EQ(f.at(4), 8);
    EXPECT_EQ(f.at(5), 2);
    EXPECT_EQ(f.at(6), 0);
    EXPECT_EQ(f.at(7), 0);
    EXPECT_EQ(f.at(8), 0);
    EXPECT_EQ(f.at(9), 1);
    EXPECT_EQ(f.at(10), 0);
    EXPECT_EQ(f.at(11), 1);
    EXPECT_EQ(f.at(12), 0);
    EXPECT_EQ(f.at(13), 0);
    EXPECT_EQ(f.at(14), 0);
    EXPECT_EQ(f.at(15), 0);
    EXPECT_EQ(f.at(16), 0);
    EXPECT_EQ(f.at(17), 0);
    EXPECT_EQ(f.at(18), 0);
}



} // namespace stdmMf


int main(int argc, char **argv)
{
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc, argv);
    return RUN_ALL_TESTS();
}
