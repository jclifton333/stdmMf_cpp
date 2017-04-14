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

using namespace stdmMf;

using njm::tools::mean_and_var;

void run(const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model<InfShieldState> > & mod_system,
        const std::shared_ptr<Model<InfShieldState> > & mod_agents,
        const uint32_t & num_reps,
        const uint32_t & time_points) {
    // njm::tools::Rng rng;
    // rng.seed(0);
    // for (uint32_t i = 0; i < 25; ++i) {
    //     rng.runif_01();
    // }
    // for (uint32_t i = 0; i < 10; ++i) {
    //     rng.rnorm_01();
    // }

    const uint32_t i(0);
    System<InfShieldState> s(net->clone(), mod_system->clone());
    s.seed(i);
    VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
            std::shared_ptr<Features<InfShieldState> >(
                    new NetworkRunSymFeatures<InfShieldState>(
                            net->clone(), 1)),
            mod_agents->clone(),
            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
    a.seed(i);

    runner(&s, &a, time_points, 1.0);

    // njm::tools::Rng rng;
    // for (uint32_t i = 0; i < 10; ++i) {
    //     rng.seed(i);
    //     std::cout << "seed: " << i << std::endl;
    //     for (uint32_t j = 0; j < 3; ++j) {
    //         std::cout << j << ": " << rng.rnorm_01() << std::endl;
    //     }
    // }

    // njm::tools::Rng rng;
    // for (uint32_t i = 0; i < 10; ++i) {
    //     std::cout << i << ": " << rng.rnorm_01() << std::endl;
    // }
    // std::cout << std::endl;
    // rng.seed(0);
    // for (uint32_t i = 0; i < 10; ++i) {
    //     std::cout << i << ": " << rng.rnorm_01() << std::endl;
    // }
    // System<InfShieldState> s(net->clone(), mod_system->clone());
    // RandomAgent<InfShieldState> ra(net->clone());
    // s.start();
    // for (uint32_t i = 0; i < 1; ++i) {
    //     const auto trt_bits(ra.apply_trt(s.state(), s.history()));

    //     s.trt_bits(trt_bits);

    //     s.turn_clock();
    // }

    // const std::vector<Transition<InfShieldState> > transitions(
    //         Transition<InfShieldState>::from_sequence(s.history(),
    //                 s.state()));

    // CHECK_EQ(transitions.size(), 1);
    // std::string bits_str;
    // boost::to_string(transitions.at(0).curr_state.inf_bits, bits_str);
    // std::cout << "inf_bits: " << bits_str << std::endl
    //           << "shield:";
    // std::for_each(transitions.at(0).curr_state.shield.begin(),
    //         transitions.at(0).curr_state.shield.end(),
    //         [] (const double & x_) {
    //             std::cout << " " << x_;
    //         });
    // std::cout << std::endl;

    // boost::to_string(transitions.at(0).curr_trt_bits, bits_str);
    // std::cout << "trt_bits: " << bits_str << std::endl;

    // boost::to_string(transitions.at(0).next_state.inf_bits, bits_str);
    // std::cout << "inf_bits: " << bits_str << std::endl
    //           << "shield:";
    // std::for_each(transitions.at(0).next_state.shield.begin(),
    //         transitions.at(0).next_state.shield.end(),
    //         [] (const double & x_) {
    //             std::cout << " " << x_;
    //         });
    // std::cout << std::endl;


    // std::vector<double> par(mod_agents->par_size(), 0.0);
    // mod_agents->par(par);

    // std::cout << "ll: " << mod_agents->ll(transitions) << std::endl;
    // for (uint32_t i = 0; i < mod_agents->par_size(); ++i) {
    //     std::fill(par.begin(), par.end(), 0.0);
    //     par.at(i) = 1.0;
    //     mod_agents->par(par);
    //     std::cout << "ll(" << i << "):" << mod_agents->ll(transitions)
    //               << std::endl;
    // }

    // std::fill(par.begin(), par.end(), 0.0);
    // std::cout << "ll: " << mod_agents->ll(transitions) << std::endl;
    // for (uint32_t i = 0; i < mod_agents->par_size(); ++i) {
    //     std::fill(par.begin(), par.end(), 0.0);
    //     par.at(i) = 1.0;
    //     mod_agents->par(par);
    //     std::cout << "ll(" << i << "):" << mod_agents->ll(transitions)
    //               << std::endl;
    // }
}


int main(int argc, char *argv[]) {
    gflags::ParseCommandLineFlags(&argc, &argv, true);
    google::SetCommandLineOption("GLOG_minloglevel", "2");
    google::InitGoogleLogging(argv[0]);


    // setup networks
    std::vector<std::shared_ptr<Network> > networks;
    { // network 1
        NetworkInit init;
        init.set_dim_x(5);
        init.set_dim_y(5);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    // { // network 2
    //     NetworkInit init;
    //     init.set_size(100);
    //     init.set_type(NetworkInit_NetType_BARABASI);
    //     networks.push_back(Network::gen_network(init));
    // }

    // { // network 3
    //     NetworkInit init;
    //     init.set_dim_x(25);
    //     init.set_dim_y(20);
    //     init.set_wrap(false);
    //     init.set_type(NetworkInit_NetType_GRID);
    //     networks.push_back(Network::gen_network(init));
    // }

    // { // network 4
    //     NetworkInit init;
    //     init.set_size(500);
    //     init.set_type(NetworkInit_NetType_BARABASI);
    //     networks.push_back(Network::gen_network(init));
    // }

    // { // network 5
    //     NetworkInit init;
    //     init.set_dim_x(25);
    //     init.set_dim_y(40);
    //     init.set_wrap(false);
    //     init.set_type(NetworkInit_NetType_GRID);
    //     networks.push_back(Network::gen_network(init));
    // }

    // { // network 6
    //     NetworkInit init;
    //     init.set_size(1000);
    //     init.set_type(NetworkInit_NetType_BARABASI);
    //     networks.push_back(Network::gen_network(init));
    // }

    // double vector since model depends on network
    typedef std::pair<std::shared_ptr<Model<InfShieldState> >,
                      std::shared_ptr<Model<InfShieldState> > > ModelPair;
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


        { // Correct: NoIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Model_NoImNoSo_NoImNoSo",
                            models_add));
        }

        // { // Correct: PosIm NoSo,  Postulated: PosIm NoSo
        //     std::vector<ModelPair> models_add;
        //     for (uint32_t i = 0; i < networks.size(); ++i) {
        //         ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
        //                         new InfShieldStatePosImNoSoModel(
        //                                 networks.at(i))),
        //                 std::shared_ptr<Model<InfShieldState> >(
        //                         new InfShieldStatePosImNoSoModel(
        //                                 networks.at(i))));
        //         mp.first->par(par);
        //         mp.second->par(par);

        //         models_add.push_back(mp);
        //     }
        //     models.push_back(std::pair<std::string,
        //             std::vector<ModelPair> >("Model_PosImNoSo_PosImNoSo",
        //                     models_add));
        // }

        // { // Correct: PosIm NoSo,  Postulated: NoIm NoSo
        //     std::vector<ModelPair> models_add;
        //     for (uint32_t i = 0; i < networks.size(); ++i) {
        //         ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
        //                         new InfShieldStatePosImNoSoModel(
        //                                 networks.at(i))),
        //                 std::shared_ptr<Model<InfShieldState> >(
        //                         new InfShieldStateNoImNoSoModel(
        //                                 networks.at(i))));
        //         mp.first->par(par);
        //         mp.second->par(par);

        //         models_add.push_back(mp);
        //     }
        //     models.push_back(std::pair<std::string,
        //             std::vector<ModelPair> >("Model_PosImNoSo_NoImNoSo",
        //                     models_add));
        // }

        // { // Correct: NoIm NoSo,  Postulated: PosIm NoSo
        //     std::vector<ModelPair> models_add;
        //     for (uint32_t i = 0; i < networks.size(); ++i) {
        //         ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
        //                         new InfShieldStateNoImNoSoModel(
        //                                 networks.at(i))),
        //                 std::shared_ptr<Model<InfShieldState> >(
        //                         new InfShieldStatePosImNoSoModel(
        //                                 networks.at(i))));
        //         mp.first->par(par);
        //         mp.second->par(par);

        //         models_add.push_back(mp);
        //     }
        //     models.push_back(std::pair<std::string,
        //             std::vector<ModelPair> >("Model_NoImNoSo_PosImNoSo",
        //                     models_add));
        // }
    }

    const uint32_t num_reps = 50;
    const uint32_t time_points = 100;


    for (uint32_t i = 0; i < networks.size(); ++i) {
        const std::shared_ptr<Network> & net = networks.at(i);

        for (uint32_t j = 0; j < models.size(); ++j) {
            ModelPair & mp(models.at(j).second.at(i));

            run(net, mp.first, mp.second, num_reps, time_points);

        }
    }

    return 0;
}
