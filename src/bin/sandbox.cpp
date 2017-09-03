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

#include "ebolaStateGravityModel.hpp"

#include "ebolaFeatures.hpp"
#include "ebolaModelFeatures.hpp"
#include "ebolaBinnedFeatures.hpp"
#include "ebolaTransProbFeatures.hpp"

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

#include "ebolaData.hpp"

using namespace stdmMf;

int main(int argc, char *argv[]) {
    EbolaData::init();

    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    std::shared_ptr<Network> net(Network::gen_network(init));

    // init model
    const std::shared_ptr<InfShieldStatePosImNoSoModel> mod(
            new InfShieldStatePosImNoSoModel(net));

    // latent infections
    const double prob_inf_latent = 0.01;
    const double intcp_inf_latent =
        std::log(1. / (1. - prob_inf_latent) - 1);

    // neighbor infections
    const double prob_inf = 0.5;
    const uint32_t prob_num_neigh = 3;
    const double intcp_inf =
        std::log(std::pow((1. - prob_inf) / (1. - prob_inf_latent),
                        -1. / prob_num_neigh)
                - 1.);

    const double trt_act_inf =
        std::log(std::pow((1. - prob_inf * 0.25) / (1. - prob_inf_latent),
                        -1. / prob_num_neigh)
                - 1.)
        - intcp_inf;

    const double trt_pre_inf =
        std::log(std::pow((1. - prob_inf * 0.75) / (1. - prob_inf_latent),
                        -1. / prob_num_neigh)
                - 1.)
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

    std::vector<double> par_sep =
        {intcp_inf_latent,
         intcp_inf,
         intcp_rec,
         trt_act_inf,
         -trt_act_inf,
         trt_act_rec,
         -trt_act_rec,
         trt_pre_inf,
         -trt_pre_inf,
         shield_coef};


    mod->par(par);

    const uint32_t time_points(25);

    std::shared_ptr<Model<InfShieldState> > modNoIm(
            new InfShieldStateNoImNoSoModel(net));
    std::shared_ptr<Model<InfShieldState> > modPosIm(
            new InfShieldStatePosImNoSoModel(net));

    System<InfShieldState> s(net, mod->clone());
    s.seed(1);
    BrMinSimPerturbAgent<InfShieldState> a(net,
            std::shared_ptr<Features<InfShieldState> >(
                    new FiniteQfnFeatures<InfShieldState>(
                            net, {modNoIm, modPosIm},
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<
                                    InfShieldState>(
                                            net, 2)), 1,
                            true, false)),
            mod->clone(),
            0.1, 0.2, 1.41, 1, 0.85, 7.15e-3,
            true, true, false, 0, 0, 0, 0, 0, false, true);
    // VfnMaxSimPerturbAgent<InfShieldState> a(net,
    //         std::shared_ptr<Features<InfShieldState> >(
    //                 new EbolaTransProbFeatures(
    //                         net, mod->clone())),
    //         mod->clone(),
    //         2, time_points, 1, 10.0, 0.1, 10, 1, 0.4, 1.2);
    a.seed(1);

    s.start();

    runner(&s, &a, time_points, 1.0);

    return 0;
}
