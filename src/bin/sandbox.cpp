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

#include "networkRunSymFeatures.hpp"

#include "objFns.hpp"

#include <njm_cpp/data/result.hpp>
#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/info/project.hpp>
#include <njm_cpp/tools/stats.hpp>

#include <njm_cpp/tools/progress.hpp>

#include <thread>

#include <fstream>

using namespace stdmMf;

using njm::data::Result;
using njm::tools::mean_and_var;

void run(const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model<InfShieldState> > & mod_system,
        const std::shared_ptr<Model<InfShieldState> > & mod_agents,
        const uint32_t & time_points) {


    // vfn max length 2
    System<InfShieldState> s(net->clone(), mod_system->clone());
    s.seed(0);
    VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
            std::shared_ptr<Features<InfShieldState> >(
                    new NetworkRunSymFeatures<InfShieldState>(
                            net->clone(), 1)),
            mod_agents->clone(),
            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
    a.seed(0);

    s.start();

    runner(&s, &a, time_points, 1.0);
}


int main(int argc, char *argv[]) {
    // setup networks

    // network 1
    NetworkInit init;
    init.set_dim_x(2);
    init.set_dim_y(2);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);
    std::shared_ptr<Network> network( Network::gen_network(init));

    // models
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


    // Correct: NoIm NoSo,  Postulated: NoIm NoSo
    std::shared_ptr<Model<InfShieldState> > systemModel(
                    new InfShieldStateNoImNoSoModel(
                            network));
    std::shared_ptr<Model<InfShieldState> > agentModel(
                    new InfShieldStateNoImNoSoModel(
                            network));

    systemModel->par(par);
    agentModel->par(par);

    const uint32_t time_points = 100;
    run(network, systemModel, agentModel, time_points);

    return 0;
}
