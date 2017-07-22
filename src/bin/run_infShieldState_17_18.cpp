#include "system.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "infShieldStatePosImNoSoModel.hpp"
#include "mixtureModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"
#include "vfnBrStartSimPerturbAgent.hpp"
#include "brMinIterSimPerturbAgent.hpp"

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

#include <future>

#include <thread>

#include <fstream>

#include <chrono>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

#include "run_infShieldState_helper.hpp"

using namespace stdmMf;


int main(int argc, char *argv[]) {
    // gflags::ParseCommandLineFlags(&argc, &argv, true);
    // google::SetCommandLineOption("GLOG_minloglevel", "2");
    google::InitGoogleLogging(argv[0]);

    // setup networks
    std::vector<std::shared_ptr<const Network> > networks;
    { // random 100
        NetworkInit init;
        init.set_size(100);
        init.set_type(NetworkInit_NetType_RANDOM);
        networks.push_back(Network::gen_network(init));
    }

    { // random 500
        NetworkInit init;
        init.set_size(500);
        init.set_type(NetworkInit_NetType_RANDOM);
        networks.push_back(Network::gen_network(init));
    }

    { // random 1000
        NetworkInit init;
        init.set_size(1000);
        init.set_type(NetworkInit_NetType_RANDOM);
        networks.push_back(Network::gen_network(init));
    }

    // double vector since model depends on network
    std::vector<std::pair<std::string,
                          std::vector<ModelPair> > > models;
    { // models
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


        { // Correct: 0.25 NoIm NoSo + 0.75 PosIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                // set par in advance because mixture model doesn't
                // implement par functionality
                auto mod_one(std::make_shared<InfShieldStateNoImNoSoModel>(
                                networks.at(i)));
                mod_one->par(par);
                auto mod_two(std::make_shared<InfShieldStatePosImNoSoModel>(
                                networks.at(i)));
                mod_two->par(par);
                ModelPair mp (std::shared_ptr<MixtureModel<InfShieldState,
                        InfShieldStateModel> >(
                                new MixtureModel<InfShieldState,
                                InfShieldStateModel>(
                                {mod_one, mod_two}, {0.25, 0.75},
                                networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >(
                            "Mixture-NoImNoSo-25-PosImNoSo-75-NoImNoSo",
                            models_add));
        }
    }

    run_infShieldState_sim(argv[0], networks, models);

    return 0;
}
