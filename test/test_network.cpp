#include <gtest/gtest.h>
#include <glog/logging.h>
#include "network.hpp"

namespace stdmMf {


TEST(TestNetwork,TestGridNetworkNoWrap) {
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> net = Network::gen_network(init);

    ASSERT_EQ(net->size(),9);

    // node 0
    EXPECT_EQ(net->get_node(0).index(), 0);
    EXPECT_NEAR(net->get_node(0).x(), 0., 1e-6);
    EXPECT_NEAR(net->get_node(0).y(), 0., 1e-6);
    EXPECT_EQ(net->get_node(0).neigh(0), 0);
    EXPECT_EQ(net->get_node(0).neigh(1), 1);
    EXPECT_EQ(net->get_node(0).neigh(2), 0);
    EXPECT_EQ(net->get_node(0).neigh(3), 3);

    // node 1
    EXPECT_EQ(net->get_node(1).index(), 1);
    EXPECT_NEAR(net->get_node(1).x(), 0., 1e-6);
    EXPECT_NEAR(net->get_node(1).y(), 0.5, 1e-6);
    EXPECT_EQ(net->get_node(1).neigh(0), 0);
    EXPECT_EQ(net->get_node(1).neigh(1), 2);
    EXPECT_EQ(net->get_node(1).neigh(2), 1);
    EXPECT_EQ(net->get_node(1).neigh(3), 4);

    // node 2
    EXPECT_EQ(net->get_node(2).index(), 2);
    EXPECT_NEAR(net->get_node(2).x(), 0., 1e-6);
    EXPECT_NEAR(net->get_node(2).y(), 1., 1e-6);
    EXPECT_EQ(net->get_node(2).neigh(0), 1);
    EXPECT_EQ(net->get_node(2).neigh(1), 2);
    EXPECT_EQ(net->get_node(2).neigh(2), 2);
    EXPECT_EQ(net->get_node(2).neigh(3), 5);

    // node 3
    EXPECT_EQ(net->get_node(3).index(), 3);
    EXPECT_NEAR(net->get_node(3).x(), 0.5, 1e-6);
    EXPECT_NEAR(net->get_node(3).y(), 0., 1e-6);
    EXPECT_EQ(net->get_node(3).neigh(0), 3);
    EXPECT_EQ(net->get_node(3).neigh(1), 4);
    EXPECT_EQ(net->get_node(3).neigh(2), 0);
    EXPECT_EQ(net->get_node(3).neigh(3), 6);

    // node 4
    EXPECT_EQ(net->get_node(4).index(), 4);
    EXPECT_NEAR(net->get_node(4).x(), 0.5, 1e-6);
    EXPECT_NEAR(net->get_node(4).y(), 0.5, 1e-6);
    EXPECT_EQ(net->get_node(4).neigh(0), 3);
    EXPECT_EQ(net->get_node(4).neigh(1), 5);
    EXPECT_EQ(net->get_node(4).neigh(2), 1);
    EXPECT_EQ(net->get_node(4).neigh(3), 7);

    // node 5
    EXPECT_EQ(net->get_node(5).index(), 5);
    EXPECT_NEAR(net->get_node(5).x(), 0.5, 1e-6);
    EXPECT_NEAR(net->get_node(5).y(), 1., 1e-6);
    EXPECT_EQ(net->get_node(5).neigh(0), 4);
    EXPECT_EQ(net->get_node(5).neigh(1), 5);
    EXPECT_EQ(net->get_node(5).neigh(2), 2);
    EXPECT_EQ(net->get_node(5).neigh(3), 8);

    // node 6
    EXPECT_EQ(net->get_node(6).index(), 6);
    EXPECT_NEAR(net->get_node(6).x(), 1., 1e-6);
    EXPECT_NEAR(net->get_node(6).y(), 0., 1e-6);
    EXPECT_EQ(net->get_node(6).neigh(0), 6);
    EXPECT_EQ(net->get_node(6).neigh(1), 7);
    EXPECT_EQ(net->get_node(6).neigh(2), 3);
    EXPECT_EQ(net->get_node(6).neigh(3), 6);

    // node 7
    EXPECT_EQ(net->get_node(7).index(), 7);
    EXPECT_NEAR(net->get_node(7).x(), 1., 1e-6);
    EXPECT_NEAR(net->get_node(7).y(), 0.5, 1e-6);
    EXPECT_EQ(net->get_node(7).neigh(0), 6);
    EXPECT_EQ(net->get_node(7).neigh(1), 8);
    EXPECT_EQ(net->get_node(7).neigh(2), 4);
    EXPECT_EQ(net->get_node(7).neigh(3), 7);

    // node 8
    EXPECT_EQ(net->get_node(8).index(), 8);
    EXPECT_NEAR(net->get_node(8).x(), 1., 1e-6);
    EXPECT_NEAR(net->get_node(8).y(), 1., 1e-6);
    EXPECT_EQ(net->get_node(8).neigh(0), 7);
    EXPECT_EQ(net->get_node(8).neigh(1), 8);
    EXPECT_EQ(net->get_node(8).neigh(2), 5);
    EXPECT_EQ(net->get_node(8).neigh(3), 8);
}



TEST(TestNetwork,TestGridNetworkWrap) {
    NetworkInit init;
    init.set_dim_x(3);
    init.set_dim_y(3);
    init.set_wrap(true);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> net = Network::gen_network(init);

    ASSERT_EQ(net->size(),9);

    // node 0
    EXPECT_EQ(net->get_node(0).index(), 0);
    EXPECT_NEAR(net->get_node(0).x(), 0., 1e-6);
    EXPECT_NEAR(net->get_node(0).y(), 0., 1e-6);
    EXPECT_EQ(net->get_node(0).neigh(0), 2);
    EXPECT_EQ(net->get_node(0).neigh(1), 1);
    EXPECT_EQ(net->get_node(0).neigh(2), 6);
    EXPECT_EQ(net->get_node(0).neigh(3), 3);

    // node 1
    EXPECT_EQ(net->get_node(1).index(), 1);
    EXPECT_NEAR(net->get_node(1).x(), 0., 1e-6);
    EXPECT_NEAR(net->get_node(1).y(), 0.5, 1e-6);
    EXPECT_EQ(net->get_node(1).neigh(0), 0);
    EXPECT_EQ(net->get_node(1).neigh(1), 2);
    EXPECT_EQ(net->get_node(1).neigh(2), 7);
    EXPECT_EQ(net->get_node(1).neigh(3), 4);

    // node 2
    EXPECT_EQ(net->get_node(2).index(), 2);
    EXPECT_NEAR(net->get_node(2).x(), 0., 1e-6);
    EXPECT_NEAR(net->get_node(2).y(), 1., 1e-6);
    EXPECT_EQ(net->get_node(2).neigh(0), 1);
    EXPECT_EQ(net->get_node(2).neigh(1), 0);
    EXPECT_EQ(net->get_node(2).neigh(2), 8);
    EXPECT_EQ(net->get_node(2).neigh(3), 5);

    // node 3
    EXPECT_EQ(net->get_node(3).index(), 3);
    EXPECT_NEAR(net->get_node(3).x(), 0.5, 1e-6);
    EXPECT_NEAR(net->get_node(3).y(), 0., 1e-6);
    EXPECT_EQ(net->get_node(3).neigh(0), 5);
    EXPECT_EQ(net->get_node(3).neigh(1), 4);
    EXPECT_EQ(net->get_node(3).neigh(2), 0);
    EXPECT_EQ(net->get_node(3).neigh(3), 6);

    // node 4
    EXPECT_EQ(net->get_node(4).index(), 4);
    EXPECT_NEAR(net->get_node(4).x(), 0.5, 1e-6);
    EXPECT_NEAR(net->get_node(4).y(), 0.5, 1e-6);
    EXPECT_EQ(net->get_node(4).neigh(0), 3);
    EXPECT_EQ(net->get_node(4).neigh(1), 5);
    EXPECT_EQ(net->get_node(4).neigh(2), 1);
    EXPECT_EQ(net->get_node(4).neigh(3), 7);

    // node 5
    EXPECT_EQ(net->get_node(5).index(), 5);
    EXPECT_NEAR(net->get_node(5).x(), 0.5, 1e-6);
    EXPECT_NEAR(net->get_node(5).y(), 1., 1e-6);
    EXPECT_EQ(net->get_node(5).neigh(0), 4);
    EXPECT_EQ(net->get_node(5).neigh(1), 3);
    EXPECT_EQ(net->get_node(5).neigh(2), 2);
    EXPECT_EQ(net->get_node(5).neigh(3), 8);

    // node 6
    EXPECT_EQ(net->get_node(6).index(), 6);
    EXPECT_NEAR(net->get_node(6).x(), 1., 1e-6);
    EXPECT_NEAR(net->get_node(6).y(), 0., 1e-6);
    EXPECT_EQ(net->get_node(6).neigh(0), 8);
    EXPECT_EQ(net->get_node(6).neigh(1), 7);
    EXPECT_EQ(net->get_node(6).neigh(2), 3);
    EXPECT_EQ(net->get_node(6).neigh(3), 0);

    // node 7
    EXPECT_EQ(net->get_node(7).index(), 7);
    EXPECT_NEAR(net->get_node(7).x(), 1., 1e-6);
    EXPECT_NEAR(net->get_node(7).y(), 0.5, 1e-6);
    EXPECT_EQ(net->get_node(7).neigh(0), 6);
    EXPECT_EQ(net->get_node(7).neigh(1), 8);
    EXPECT_EQ(net->get_node(7).neigh(2), 4);
    EXPECT_EQ(net->get_node(7).neigh(3), 1);

    // node 8
    EXPECT_EQ(net->get_node(8).index(), 8);
    EXPECT_NEAR(net->get_node(8).x(), 1., 1e-6);
    EXPECT_NEAR(net->get_node(8).y(), 1., 1e-6);
    EXPECT_EQ(net->get_node(8).neigh(0), 7);
    EXPECT_EQ(net->get_node(8).neigh(1), 6);
    EXPECT_EQ(net->get_node(8).neigh(2), 5);
    EXPECT_EQ(net->get_node(8).neigh(3), 2);
}



} // namespace coopPE


int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
