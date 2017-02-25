#include <gtest/gtest.h>
#include <glog/logging.h>

#include "objFns.hpp"
#include "polValIteration.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "infStateNoSoModel.hpp"

namespace stdmMf {


TEST(TestPolValIteration, ValueIterationNoTrtAgent) {
    NetworkInit init;
    init.set_dim_x(2);
    init.set_dim_y(2);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> network = Network::gen_network(init);

    std::shared_ptr<Agent<InfState> > agent(
            new NoTrtAgent<InfState>(network));
    std::shared_ptr<Model<InfState> > model(new InfStateNoSoModel(network));


    // latent infections
    const double prob_inf_latent = 0.01;
    const double intcp_inf_latent =
        std::log(1. / (1. - prob_inf_latent) - 1);

    // neighbor infections
    const double prob_inf = 0.5;
    const uint32_t prob_num_neigh = 3;
    const double intcp_inf =
        std::log(std::pow(1. - prob_inf, -1. / prob_num_neigh) - 1.);

    const double trt_act_inf =
        std::log(std::pow(1. - prob_inf * 0.25, -1. / prob_num_neigh) - 1.)
        - intcp_inf;

    const double trt_pre_inf =
        std::log(std::pow(1. - prob_inf * 0.75, -1. / prob_num_neigh) - 1.)
        - intcp_inf;

    // recovery
    const double prob_rec = 0.25;
    const double intcp_rec = std::log(1. / (1. - prob_rec) - 1.);
    const double trt_act_rec =
        std::log(1. / ((1. - prob_rec) * 0.5) - 1.) - intcp_rec;


    std::vector<double> par =
        {intcp_inf_latent,
         intcp_inf,
         intcp_rec,
         trt_act_inf,
         trt_act_rec,
         trt_pre_inf};

    model->par(par);


    CHECK_LT(network->size(), 32);
    const uint32_t max_inf_bits = (1 << network->size());

    System<InfState> s(network, model);

    const uint32_t num_reps = 200;
    const uint32_t time_points = 200;
    const double gamma = 0.3;

    StateLookup<InfState, boost::dynamic_bitset<> > agent_lookup;

    std::vector<double> est_value(max_inf_bits, 0.0);
    for (uint32_t i = 0; i < max_inf_bits; ++i) {
        const boost::dynamic_bitset<> start_inf_bits(network->size(), i);

        agent_lookup.put(start_inf_bits, agent->apply_trt(start_inf_bits));

        double sum_val = 0.0;
        for (uint32_t r = 0; r < num_reps; ++r) {
            s.reset();
            s.state(start_inf_bits);

            sum_val += runner(&s, agent.get(), time_points, gamma);
        }
        est_value.at(i) = sum_val / num_reps;
    }

    const std::vector<double> value = value_iteration(agent_lookup,
            gamma, network, model);

    ASSERT_EQ(value.size(), est_value.size());

    for (uint32_t i = 0; i < max_inf_bits; ++i) {
        EXPECT_LT(std::abs(value.at(i) - est_value.at(i))
                / std::abs(value.at(i)), 0.2);
    }
}



} // namespace stdmMf

int main(int argc, char *argv[]) {
    ::google::InitGoogleLogging(argv[0]);
    ::testing::InitGoogleTest(&argc,argv);
    return RUN_ALL_TESTS();
}
