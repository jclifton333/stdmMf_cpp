#include "system.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "infShieldStatePosImNoSoModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"
#include "vfnBrStartSimPerturbAgent.hpp"

#include "brMinWtdSimPerturbAgent.hpp"


#include "networkRunSymFeatures.hpp"
#include "finiteQfnFeatures.hpp"

#include "objFns.hpp"

#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/info/project.hpp>
#include <njm_cpp/tools/stats.hpp>

#include <njm_cpp/tools/progress.hpp>

#include <thread>

#include <fstream>

#include <glog/logging.h>

#include <chrono>

using namespace stdmMf;

int main(int argc, char *argv[]) {
    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);
    auto network(Network::gen_network(init));

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

    // shield
    const double shield_coef = 0.9;


    std::vector<double> par =
        {intcp_inf_latent,
         intcp_inf,
         intcp_rec,
         trt_act_inf,
         trt_act_rec,
         trt_pre_inf,
         shield_coef};

    const uint32_t time_points(25);

    auto mod_system(std::shared_ptr<Model<InfShieldState> >(
                    new InfShieldStateNoImNoSoModel(
                            network)));
    auto mod_agents(std::shared_ptr<Model<InfShieldState> >(
                    new InfShieldStateNoImNoSoModel(
                            network)));
    mod_system->par(par);
    mod_agents->par(par);


    System<InfShieldState> s(network, mod_system->clone());
    s.seed(0);

    BrMinWtdSimPerturbAgent<InfShieldState> a(network,
            std::shared_ptr<Features<InfShieldState> >(
                    new FiniteQfnFeatures<InfShieldState>(
                            network, {mod_agents->clone()},
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<
                                    InfShieldState>(
                                            network, 2)), 1)),
            mod_agents->clone(),
            0.1, 0.2, 1.41, 1, 0.85, 7.15e-3,
            true, true, false, 100, 10, 0);
    a.seed(0);

    s.start();

    runner(&s, &a, time_points, 1.0);

    return 0;
}
