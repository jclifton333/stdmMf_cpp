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

    CHECK_EQ(nrf.num_features(), 1 + 3);
}



} // namespace stdmMf
