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
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::SetCommandLineOption("GLOG_minloglevel", "2");
    google::InitGoogleLogging(argv[0]);

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

    std::cout << std::right << std::setw(16)
              << "network size"
              << std::right << std::setw(16)
              << "nn fitting"
              << std::right << std::setw(16)
              << "nn argmax"
              << std::right << std::setw(16)
              << "runs argmax"
              << std::endl;

    const std::vector<uint32_t> net_sizes({2, 3, 4, 5, 6, 7, 8, 9, 10});
    for (uint32_t i = 0; i < net_sizes.size(); ++i) {
        NetworkInit init;
        init.set_dim_x(net_sizes.at(i));
        init.set_dim_y(net_sizes.at(i));
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        const auto net(Network::gen_network(init));

        const auto mod_system(std::shared_ptr<Model<InfShieldState> >(
                        new InfShieldStateNoImNoSoModel(net)));
        mod_system->par(par);

        const auto mod_agents(std::shared_ptr<Model<InfShieldState> >(
                        new InfShieldStateNoImNoSoModel(net)));

        System<InfShieldState> s(net->clone(), mod_system->clone());
        RandomAgent<InfShieldState> ra(net->clone());
        s.start();
        for (uint32_t j = 0; j < 20; ++j) {
            const auto trt_bits(ra.apply_trt(s.state(), s.history()));

            s.trt_bits(trt_bits);

            s.turn_clock();
        }


        // const auto finiteQfnFeat(
        //         std::make_shared<FiniteQfnNnFeatures<InfShieldState> >(
        //                 net->clone(), mod_agents->clone(), 3));

        const auto networkRunFeat(
                std::make_shared<NetworkRunSymFeatures<InfShieldState> >(
                        net->clone(), 3));

        const auto finiteQfnFeat(
                std::shared_ptr<FiniteQfnFeatures<InfShieldState> >(
                        new FiniteQfnFeatures<InfShieldState>(
                                net->clone(), {mod_agents->clone()},
                                networkRunFeat->clone(), 3)));

        SweepAgent<InfShieldState> saFiniteQfn(net, finiteQfnFeat,
                std::vector<double>(finiteQfnFeat->num_features(), 1.0),
                njm::linalg::dot_a_and_b, 2, true);

        SweepAgent<InfShieldState> saNetworkRun(net, networkRunFeat,
                std::vector<double>(networkRunFeat->num_features(), 1.0),
                njm::linalg::dot_a_and_b, 2, true);

        std::cout << std::setw(16) << net_sizes.at(i) * net_sizes.at(i);

        { // time the update for the neural network
            const auto tick(std::chrono::high_resolution_clock::now());
            finiteQfnFeat->update(s.state(), s.history(),
                    saFiniteQfn.num_trt());

            const auto tock(std::chrono::high_resolution_clock::now());

            const auto elapsed(std::chrono::duration_cast<
                    std::chrono::duration<double> >(tock - tick));

            std::cout << std::fixed << std::setw(16) << std::setprecision(6)
                      << elapsed.count();

        }

        { // time the arg max of the neural network features using sweep
            const auto tick(std::chrono::high_resolution_clock::now());
            for (uint32_t r = 0; r < 100; r++) {
                saFiniteQfn.apply_trt(s.state());
            }
            const auto tock(std::chrono::high_resolution_clock::now());

            const auto elapsed(std::chrono::duration_cast<
                    std::chrono::duration<double> >(tock - tick));

            std::cout << std::fixed << std::setw(16) << std::setprecision(6)
                      << elapsed.count() / 100.0;
        }

        { // time the arg max of the network run features using sweep
            networkRunFeat->update(s.state(), s.history(),
                    saNetworkRun.num_trt());
            const auto tick(std::chrono::high_resolution_clock::now());
            for (uint32_t r = 0; r < 100; ++r) {
                saNetworkRun.apply_trt(s.state());
            }
            const auto tock(std::chrono::high_resolution_clock::now());

            const auto elapsed(std::chrono::duration_cast<
                    std::chrono::duration<double> >(tock - tick));

            std::cout << std::fixed << std::setw(16) << std::setprecision(6)
                      << elapsed.count() / 100.0 << std::endl;
        }


    }

    return 0;
}
