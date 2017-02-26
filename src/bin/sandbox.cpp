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
        const uint32_t & num_reps,
        const uint32_t & time_points) {

    // Pool pool(std::min(num_reps, std::thread::hardware_concurrency()));
    njm::thread::Pool pool(std::thread::hardware_concurrency());

    std::shared_ptr<njm::tools::Progress<std::ostream> > progress(
            new njm::tools::Progress<std::ostream>(&std::cout));

    uint32_t total_sims = 0;

    // vfn max length 1
    std::vector<std::shared_ptr<Result<double> > > vfn_len_1_val;
    std::vector<std::shared_ptr<Result<double> > > vfn_len_1_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        vfn_len_1_val.push_back(r_val);
        vfn_len_1_time.push_back(r_time);

        pool.service().post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 1)),
                            mod_agents->clone(),
                            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
                    a.seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::steady_clock> tick =
                        std::chrono::steady_clock::now();

                    r_val->set(runner(&s, &a, time_points, 1.0));

                    std::chrono::time_point<
                        std::chrono::steady_clock> tock =
                        std::chrono::steady_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());

                    progress->update();
                });
    }


    // vfn max length 2
    std::vector<std::shared_ptr<Result<double> > > vfn_len_2_val;
    std::vector<std::shared_ptr<Result<double> > > vfn_len_2_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        vfn_len_2_val.push_back(r_val);
        vfn_len_2_time.push_back(r_time);

        pool.service().post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 2)),
                            mod_agents->clone(),
                            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
                    a.seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::steady_clock> tick =
                        std::chrono::steady_clock::now();

                    r_val->set(runner(&s, &a, time_points, 1.0));

                    std::chrono::time_point<
                        std::chrono::steady_clock> tock =
                        std::chrono::steady_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());

                    progress->update();
                });
    }


    // vfn max length 3
    std::vector<std::shared_ptr<Result<double> > > vfn_len_3_val;
    std::vector<std::shared_ptr<Result<double> > > vfn_len_3_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        vfn_len_3_val.push_back(r_val);
        vfn_len_3_time.push_back(r_time);

        pool.service().post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 3)),
                            mod_agents->clone(),
                            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
                    a.seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::steady_clock> tick =
                        std::chrono::steady_clock::now();

                    r_val->set(runner(&s, &a, time_points, 1.0));

                    std::chrono::time_point<
                        std::chrono::steady_clock> tock =
                        std::chrono::steady_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());

                    progress->update();
                });
    }


    // br min length 1
    std::vector<std::shared_ptr<Result<double> > > br_len_1_val;
    std::vector<std::shared_ptr<Result<double> > > br_len_1_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        br_len_1_val.push_back(r_val);
        br_len_1_time.push_back(r_time);

        pool.service().post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    BrMinSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 1)),
                            2e-1, 0.75, 1.41e-3, 1, 0.85, 9.130e-6);
                    a.seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::steady_clock> tick =
                        std::chrono::steady_clock::now();

                    r_val->set(runner(&s, &a, time_points, 1.0));

                    std::chrono::time_point<
                        std::chrono::steady_clock> tock =
                        std::chrono::steady_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());

                    progress->update();
                });
    }


    // br min length 2
    std::vector<std::shared_ptr<Result<double> > > br_len_2_val;
    std::vector<std::shared_ptr<Result<double> > > br_len_2_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        br_len_2_val.push_back(r_val);
        br_len_2_time.push_back(r_time);

        pool.service().post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    BrMinSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 2)),
                            2e-1, 0.75, 1.41e-3, 1, 0.85, 9.130e-6);
                    a.seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::steady_clock> tick =
                        std::chrono::steady_clock::now();

                    r_val->set(runner(&s, &a, time_points, 1.0));

                    std::chrono::time_point<
                        std::chrono::steady_clock> tock =
                        std::chrono::steady_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());

                    progress->update();
                });
    }


    // br min length 3
    std::vector<std::shared_ptr<Result<double> > > br_len_3_val;
    std::vector<std::shared_ptr<Result<double> > > br_len_3_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        ++total_sims;
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        br_len_3_val.push_back(r_val);
        br_len_3_time.push_back(r_time);

        pool.service().post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    BrMinSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 3)),
                            2e-1, 0.75, 1.41e-3, 1, 0.85, 9.130e-6);
                    a.seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::steady_clock> tick =
                        std::chrono::steady_clock::now();

                    r_val->set(runner(&s, &a, time_points, 1.0));

                    std::chrono::time_point<
                        std::chrono::steady_clock> tock =
                        std::chrono::steady_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());

                    progress->update();
                });
    }

    progress->total(total_sims);

    pool.join();

    progress->done();

}


int main(int argc, char *argv[]) {
    // setup networks
    std::vector<std::shared_ptr<Network> > networks;
    { // network 1
        NetworkInit init;
        init.set_dim_x(2);
        init.set_dim_y(2);
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
