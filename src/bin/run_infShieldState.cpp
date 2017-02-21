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

#include <thread>

#include <fstream>

using namespace stdmMf;

using njm::data::Result;
using njm::tools::mean_and_var;

std::vector<std::pair<std::string, std::vector<double> > >
run(const std::shared_ptr<Network> & net,
        const std::shared_ptr<Model<InfShieldState> > & mod_system,
        const std::shared_ptr<Model<InfShieldState> > & mod_agents,
        const uint32_t & num_reps,
        const uint32_t & time_points) {

    // Pool pool(std::min(num_reps, std::thread::hardware_concurrency()));
    njm::thread::Pool pool(std::thread::hardware_concurrency());

    // none
    std::vector<std::shared_ptr<Result<double> > > none_val;
    std::vector<std::shared_ptr<Result<double> > > none_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        none_val.push_back(r_val);
        none_time.push_back(r_time);

        pool.service()->post([=](){
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    NoTrtAgent<InfShieldState> a(net->clone());
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
                });
    }

    // random
    std::vector<std::shared_ptr<Result<double> > > random_val;
    std::vector<std::shared_ptr<Result<double> > > random_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        random_val.push_back(r_val);
        random_time.push_back(r_time);

        pool.service()->post([=](){
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    RandomAgent<InfShieldState> a(net->clone());
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
                });
    }


    // proximal
    std::vector<std::shared_ptr<Result<double> > > proximal_val;
    std::vector<std::shared_ptr<Result<double> > > proximal_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        proximal_val.push_back(r_val);
        proximal_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    ProximalAgent<InfShieldState> a(net->clone());
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
                });
    }


    // myopic
    std::vector<std::shared_ptr<Result<double> > > myopic_val;
    std::vector<std::shared_ptr<Result<double> > > myopic_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        myopic_val.push_back(r_val);
        myopic_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    MyopicAgent<InfShieldState> a(net->clone(),
                            mod_agents->clone());
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
                });
    }


    // vfn max length 2
    std::vector<std::shared_ptr<Result<double> > > vfn_len_2_val;
    std::vector<std::shared_ptr<Result<double> > > vfn_len_2_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        vfn_len_2_val.push_back(r_val);
        vfn_len_2_time.push_back(r_time);

        pool.service()->post([=]() {
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
                });
    }


    // vfn max length 3
    std::vector<std::shared_ptr<Result<double> > > vfn_len_3_val;
    std::vector<std::shared_ptr<Result<double> > > vfn_len_3_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        vfn_len_3_val.push_back(r_val);
        vfn_len_3_time.push_back(r_time);

        pool.service()->post([=]() {
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
                });
    }


    // vfn max length 4
    std::vector<std::shared_ptr<Result<double> > > vfn_len_4_val;
    std::vector<std::shared_ptr<Result<double> > > vfn_len_4_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        vfn_len_4_val.push_back(r_val);
        vfn_len_4_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnMaxSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 4)),
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
                });
    }


    // br min length 2
    std::vector<std::shared_ptr<Result<double> > > br_len_2_val;
    std::vector<std::shared_ptr<Result<double> > > br_len_2_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        br_len_2_val.push_back(r_val);
        br_len_2_time.push_back(r_time);

        pool.service()->post([=]() {
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
                });
    }


    // br min length 3
    std::vector<std::shared_ptr<Result<double> > > br_len_3_val;
    std::vector<std::shared_ptr<Result<double> > > br_len_3_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        br_len_3_val.push_back(r_val);
        br_len_3_time.push_back(r_time);

        pool.service()->post([=]() {
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
                });
    }


    // br min length 4
    std::vector<std::shared_ptr<Result<double> > > br_len_4_val;
    std::vector<std::shared_ptr<Result<double> > > br_len_4_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        br_len_4_val.push_back(r_val);
        br_len_4_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    BrMinSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 4)),
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
                });
    }


    // // vr max br min adapt step mult 1
    // std::vector<std::shared_ptr<Result<double> > > adapt_1_val;
    // std::vector<std::shared_ptr<Result<double> > > adapt_1_time;
    // for (uint32_t i = 0; i < num_reps; ++i) {
    //     std::shared_ptr<Result<double> > r_val(new Result<double>);
    //     std::shared_ptr<Result<double> > r_time(new Result<double>);
    //     adapt_1_val.push_back(r_val);
    //     adapt_1_time.push_back(r_time);

    //     pool.service()->post([=]() {
    //                 System s(net->clone(), mod_system->clone());
    //                 s.seed(i);
    //                 VfnBrAdaptSimPerturbAgent a(net->clone(),
    //                         std::shared_ptr<Features>(
    //                                 new NetworkRunSymFeatures(net->clone(), run_length)),
    //                         mod_agents->clone(),
    //                         2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7,
    //                         1e-1, 1.0, 1e-3, 1, 0.85, 1e-5,
    //                         1);
    //                 a.seed(i);

    //                 s.start();

    //                 std::chrono::time_point<
    //                     std::chrono::steady_clock> tick =
    //                     std::chrono::steady_clock::now();

    //                 r_val->set(runner(&s, &a, time_points, 1.0));

    //                 std::chrono::time_point<
    //                     std::chrono::steady_clock> tock =
    //                     std::chrono::steady_clock::now();

    //                 r_time->set(std::chrono::duration_cast<
    //                         std::chrono::seconds>(tock - tick).count());
    //             });
    // }

    // // vr max br min adapt step mult 2
    // std::vector<std::shared_ptr<Result<double> > > adapt_2_val;
    // std::vector<std::shared_ptr<Result<double> > > adapt_2_time;
    // for (uint32_t i = 0; i < num_reps; ++i) {
    //     std::shared_ptr<Result<double> > r_val(new Result<double>);
    //     std::shared_ptr<Result<double> > r_time(new Result<double>);
    //     adapt_2_val.push_back(r_val);
    //     adapt_2_time.push_back(r_time);

    //     pool.service()->post([=]() {
    //                 System s(net->clone(), mod_system->clone());
    //                 s.seed(i);
    //                 VfnBrAdaptSimPerturbAgent a(net->clone(),
    //                         std::shared_ptr<Features>(
    //                                 new NetworkRunSymFeatures(net->clone(), run_length)),
    //                         mod_agents->clone(),
    //                         2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7,
    //                         1e-1, 1.0, 1e-3, 1, 0.85, 1e-5,
    //                         2);
    //                 a.seed(i);

    //                 s.start();

    //                 std::chrono::time_point<
    //                     std::chrono::steady_clock> tick =
    //                     std::chrono::steady_clock::now();

    //                 r_val->set(runner(&s, &a, time_points, 1.0));

    //                 std::chrono::time_point<
    //                     std::chrono::steady_clock> tock =
    //                     std::chrono::steady_clock::now();

    //                 r_time->set(std::chrono::duration_cast<
    //                         std::chrono::seconds>(tock - tick).count());
    //             });
    // }

    // // vr max br min adapt step mult 5
    // std::vector<std::shared_ptr<Result<double> > > adapt_5_val;
    // std::vector<std::shared_ptr<Result<double> > > adapt_5_time;
    // for (uint32_t i = 0; i < num_reps; ++i) {
    //     std::shared_ptr<Result<double> > r_val(new Result<double>);
    //     std::shared_ptr<Result<double> > r_time(new Result<double>);
    //     adapt_5_val.push_back(r_val);
    //     adapt_5_time.push_back(r_time);

    //     pool.service()->post([=]() {
    //                 System s(net->clone(), mod_system->clone());
    //                 s.seed(i);
    //                 VfnBrAdaptSimPerturbAgent a(net->clone(),
    //                         std::shared_ptr<Features>(
    //                                 new NetworkRunSymFeatures(net->clone(), run_length)),
    //                         mod_agents->clone(),
    //                         2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7,
    //                         1e-1, 1.0, 1e-3, 1, 0.85, 1e-5,
    //                         5);
    //                 a.seed(i);

    //                 s.start();

    //                 std::chrono::time_point<
    //                     std::chrono::steady_clock> tick =
    //                     std::chrono::steady_clock::now();

    //                 r_val->set(runner(&s, &a, time_points, 1.0));

    //                 std::chrono::time_point<
    //                     std::chrono::steady_clock> tock =
    //                     std::chrono::steady_clock::now();

    //                 r_time->set(std::chrono::duration_cast<
    //                         std::chrono::seconds>(tock - tick).count());
    //             });
    // }


    // vr max br min adapt step mult 10 length 2
    std::vector<std::shared_ptr<Result<double> > > adapt_10_len_2_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_10_len_2_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_10_len_2_val.push_back(r_val);
        adapt_10_len_2_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnBrAdaptSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 2)),
                            mod_agents->clone(),
                            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            2e-1, 1.0, 1e-3, 1, 1, 9.13e-6,
                            10);
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
                });
    }


    // vr max br min adapt step mult 10 length 3
    std::vector<std::shared_ptr<Result<double> > > adapt_10_len_3_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_10_len_3_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_10_len_3_val.push_back(r_val);
        adapt_10_len_3_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnBrAdaptSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 3)),
                            mod_agents->clone(),
                            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            2e-1, 1.0, 1e-3, 1, 1, 9.13e-6,
                            10);
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
                });
    }


    // vr max br min adapt step mult 10 length 4
    std::vector<std::shared_ptr<Result<double> > > adapt_10_len_4_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_10_len_4_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_10_len_4_val.push_back(r_val);
        adapt_10_len_4_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnBrAdaptSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 4)),
                            mod_agents->clone(),
                            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            2e-1, 1.0, 1e-3, 1, 1, 9.13e-6,
                            10);
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
                });
    }


    // vr max br min adapt step mult 100 length 2
    std::vector<std::shared_ptr<Result<double> > > adapt_100_len_2_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_100_len_2_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_100_len_2_val.push_back(r_val);
        adapt_100_len_2_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnBrAdaptSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 2)),
                            mod_agents->clone(),
                            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            2e-1, 1.0, 1e-3, 1, 1, 9.13e-6,
                            100);
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
                });
    }


    // vr max br min adapt step mult 100 length 3
    std::vector<std::shared_ptr<Result<double> > > adapt_100_len_3_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_100_len_3_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_100_len_3_val.push_back(r_val);
        adapt_100_len_3_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnBrAdaptSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 3)),
                            mod_agents->clone(),
                            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            2e-1, 1.0, 1e-3, 1, 1, 9.13e-6,
                            100);
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
                });
    }


    // vr max br min adapt step mult 100 length 4
    std::vector<std::shared_ptr<Result<double> > > adapt_100_len_4_val;
    std::vector<std::shared_ptr<Result<double> > > adapt_100_len_4_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        adapt_100_len_4_val.push_back(r_val);
        adapt_100_len_4_time.push_back(r_time);

        pool.service()->post([=]() {
                    System<InfShieldState> s(net->clone(), mod_system->clone());
                    s.seed(i);
                    VfnBrAdaptSimPerturbAgent<InfShieldState> a(net->clone(),
                            std::shared_ptr<Features<InfShieldState> >(
                                    new NetworkRunSymFeatures<InfShieldState>(
                                            net->clone(), 4)),
                            mod_agents->clone(),
                            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7,
                            2e-1, 1.0, 1e-3, 1, 1, 9.13e-6,
                            100);
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
                });
    }


    pool.join();

    std::vector<std::pair<std::string, std::vector<double> > > all_results;


    {
        const std::string agent_name = "none";
        const std::pair<double, double> none_stats = mean_and_var(
                result_to_vec(none_val));
        const std::vector<double> agent_res =
            {none_stats.first,
             std::sqrt(none_stats.second / num_reps),
             mean_and_var(result_to_vec(none_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "random";
        const std::pair<double, double> random_stats = mean_and_var(
                result_to_vec(random_val));
        const std::vector<double> agent_res =
            {random_stats.first,
             std::sqrt(random_stats.second / num_reps),
             mean_and_var(result_to_vec(random_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "proximal";
        const std::pair<double, double> proximal_stats = mean_and_var(
                result_to_vec(proximal_val));
        const std::vector<double> agent_res =
            {proximal_stats.first,
             std::sqrt(proximal_stats.second / num_reps),
             mean_and_var(result_to_vec(proximal_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "myopic";
        const std::pair<double, double> myopic_stats = mean_and_var(
                result_to_vec(myopic_val));
        const std::vector<double> agent_res =
            {myopic_stats.first,
             std::sqrt(myopic_stats.second / num_reps),
             mean_and_var(result_to_vec(myopic_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "vfn_len_2";
        const std::pair<double, double> vfn_len_2_stats = mean_and_var(
                result_to_vec(vfn_len_2_val));
        const std::vector<double> agent_res =
            {vfn_len_2_stats.first,
             std::sqrt(vfn_len_2_stats.second / num_reps),
             mean_and_var(result_to_vec(vfn_len_2_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "vfn_len_3";
        const std::pair<double, double> vfn_len_3_stats = mean_and_var(
                result_to_vec(vfn_len_3_val));
        const std::vector<double> agent_res =
            {vfn_len_3_stats.first,
             std::sqrt(vfn_len_3_stats.second / num_reps),
             mean_and_var(result_to_vec(vfn_len_3_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "vfn_len_4";
        const std::pair<double, double> vfn_len_4_stats = mean_and_var(
                result_to_vec(vfn_len_4_val));
        const std::vector<double> agent_res =
            {vfn_len_4_stats.first,
             std::sqrt(vfn_len_4_stats.second / num_reps),
             mean_and_var(result_to_vec(vfn_len_4_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    // {
    //     const std::string agent_name = "vfnBrStart";
    //     const std::pair<double, double> vfnBrStart_stats = mean_and_var(
    //             result_to_vec(vfnBrStart_val));
    //     const std::vector<double> agent_res =
    //         {vfnBrStart_stats.first,
    //          std::sqrt(vfnBrStart_stats.second / num_reps),
    //          mean_and_var(result_to_vec(vfnBrStart_time)).first};
    //     all_results.push_back(std::pair<std::string, std::vector<double> >
    //             (agent_name, agent_res));
    // }

    {
        const std::string agent_name = "br_len_2";
        const std::pair<double, double> br_len_2_stats = mean_and_var(
                result_to_vec(br_len_2_val));
        const std::vector<double> agent_res =
            {br_len_2_stats.first,
             std::sqrt(br_len_2_stats.second / num_reps),
             mean_and_var(result_to_vec(br_len_2_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "br_len_3";
        const std::pair<double, double> br_len_3_stats = mean_and_var(
                result_to_vec(br_len_3_val));
        const std::vector<double> agent_res =
            {br_len_3_stats.first,
             std::sqrt(br_len_3_stats.second / num_reps),
             mean_and_var(result_to_vec(br_len_3_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "br_len_4";
        const std::pair<double, double> br_len_4_stats = mean_and_var(
                result_to_vec(br_len_4_val));
        const std::vector<double> agent_res =
            {br_len_4_stats.first,
             std::sqrt(br_len_4_stats.second / num_reps),
             mean_and_var(result_to_vec(br_len_4_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    // {
    //     const std::string agent_name = "adapt_1";
    //     const std::pair<double, double> adapt_1_stats = mean_and_var(
    //             result_to_vec(adapt_1_val));
    //     const std::vector<double> agent_res =
    //         {adapt_1_stats.first,
    //          std::sqrt(adapt_1_stats.second / num_reps),
    //          mean_and_var(result_to_vec(adapt_1_time)).first};
    //     all_results.push_back(std::pair<std::string, std::vector<double> >
    //             (agent_name, agent_res));
    // }

    // {
    //     const std::string agent_name = "adapt_2";
    //     const std::pair<double, double> adapt_2_stats = mean_and_var(
    //             result_to_vec(adapt_2_val));
    //     const std::vector<double> agent_res =
    //         {adapt_2_stats.first,
    //          std::sqrt(adapt_2_stats.second / num_reps),
    //          mean_and_var(result_to_vec(adapt_2_time)).first};
    //     all_results.push_back(std::pair<std::string, std::vector<double> >
    //             (agent_name, agent_res));
    // }

    // {
    //     const std::string agent_name = "adapt_5";
    //     const std::pair<double, double> adapt_5_stats = mean_and_var(
    //             result_to_vec(adapt_5_val));
    //     const std::vector<double> agent_res =
    //         {adapt_5_stats.first,
    //          std::sqrt(adapt_5_stats.second / num_reps),
    //          mean_and_var(result_to_vec(adapt_5_time)).first};
    //     all_results.push_back(std::pair<std::string, std::vector<double> >
    //             (agent_name, agent_res));
    // }

    {
        const std::string agent_name = "adapt_10_len_2";
        const std::pair<double, double> adapt_10_len_2_stats = mean_and_var(
                result_to_vec(adapt_10_len_2_val));
        const std::vector<double> agent_res =
            {adapt_10_len_2_stats.first,
             std::sqrt(adapt_10_len_2_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_10_len_2_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "adapt_10_len_3";
        const std::pair<double, double> adapt_10_len_3_stats = mean_and_var(
                result_to_vec(adapt_10_len_3_val));
        const std::vector<double> agent_res =
            {adapt_10_len_3_stats.first,
             std::sqrt(adapt_10_len_3_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_10_len_3_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "adapt_10_len_4";
        const std::pair<double, double> adapt_10_len_4_stats = mean_and_var(
                result_to_vec(adapt_10_len_4_val));
        const std::vector<double> agent_res =
            {adapt_10_len_4_stats.first,
             std::sqrt(adapt_10_len_4_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_10_len_4_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "adapt_100_len_2";
        const std::pair<double, double> adapt_100_len_2_stats = mean_and_var(
                result_to_vec(adapt_100_len_2_val));
        const std::vector<double> agent_res =
            {adapt_100_len_2_stats.first,
             std::sqrt(adapt_100_len_2_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_100_len_2_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "adapt_100_len_3";
        const std::pair<double, double> adapt_100_len_3_stats = mean_and_var(
                result_to_vec(adapt_100_len_3_val));
        const std::vector<double> agent_res =
            {adapt_100_len_3_stats.first,
             std::sqrt(adapt_100_len_3_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_100_len_3_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    {
        const std::string agent_name = "adapt_100_len_4";
        const std::pair<double, double> adapt_100_len_4_stats = mean_and_var(
                result_to_vec(adapt_100_len_4_val));
        const std::vector<double> agent_res =
            {adapt_100_len_4_stats.first,
             std::sqrt(adapt_100_len_4_stats.second / num_reps),
             mean_and_var(result_to_vec(adapt_100_len_4_time)).first};
        all_results.push_back(std::pair<std::string, std::vector<double> >
                (agent_name, agent_res));
    }

    return all_results;
}


int main(int argc, char *argv[]) {
    // setup networks
    std::vector<std::shared_ptr<Network> > networks;
    { // network 1
        NetworkInit init;
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        networks.push_back(Network::gen_network(init));
    }

    { // network 2
        NetworkInit init;
        init.set_size(100);
        init.set_type(NetworkInit_NetType_BARABASI);
        networks.push_back(Network::gen_network(init));
    }

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

        { // Correct: PosIm NoSo,  Postulated: PosIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Model_PosImNoSo_PosImNoSo",
                            models_add));
        }

        { // Correct: PosIm NoSo,  Postulated: NoIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Model_PosImNoSo_NoImNoSo",
                            models_add));
        }

        { // Correct: NoIm NoSo,  Postulated: PosIm NoSo
            std::vector<ModelPair> models_add;
            for (uint32_t i = 0; i < networks.size(); ++i) {
                ModelPair mp (std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStateNoImNoSoModel(
                                        networks.at(i))),
                        std::shared_ptr<Model<InfShieldState> >(
                                new InfShieldStatePosImNoSoModel(
                                        networks.at(i))));
                mp.first->par(par);
                mp.second->par(par);

                models_add.push_back(mp);
            }
            models.push_back(std::pair<std::string,
                    std::vector<ModelPair> >("Model_NoImNoSo_PosImNoSo",
                            models_add));
        }
    }

    const uint32_t num_reps = 50;
    const uint32_t time_points = 100;

    std::ofstream ofs_raw;
    ofs_raw.open("run_results_raw.txt", std::ios_base::out);
    CHECK(ofs_raw.good()) << "could not open file";
    ofs_raw << "network,model,agent,mean,se,time" << std::endl;
    ofs_raw.close();

    std::ofstream ofs_read;
    ofs_read.open("run_results_read.txt", std::ios_base::out);
    CHECK(ofs_read.good()) << "could not open file";
    ofs_read.close();

    njm::data::TrapperKeeper tk(argv[0],
            njm::info::project::PROJECT_ROOT_DIR + "/data");

    njm::data::Entry & e_read_all = tk.entry("all_read.txt");

    for (uint32_t i = 0; i < networks.size(); ++i) {
        const std::shared_ptr<Network> & net = networks.at(i);

        njm::data::Entry & e_read_net = tk.entry(net->kind() + "_read.txt");

        for (uint32_t j = 0; j < models.size(); ++j) {
            ModelPair & mp(models.at(j).second.at(i));

            njm::data::Entry & e_raw = tk.entry(
                    net->kind() + "_" + models.at(j).first + "_raw.txt");
            njm::data::Entry & e_read = tk.entry(
                    net->kind() + "_" + models.at(j).first + "_read.txt");

            std::vector<std::pair<std::string, std::vector<double> > >
                results = run(net, mp.first, mp.second, num_reps, time_points);

            std::cout << "=====================================" << std::endl
                      << "results for network " << net->kind()
                      << " and model pair " << j << std::endl;

            e_read << "=====================================" << "\n"
                   << "results for network " << net->kind()
                   << " and model pair " << models.at(j).first << "\n";

            e_read_net << "=====================================" << "\n"
                       << "results for network " << net->kind()
                       << " and model pair " << models.at(j).first << "\n";

            e_read_all << "=====================================" << "\n"
                       << "results for network " << net->kind()
                       << " and model pair " << models.at(j).first << "\n";

            for (uint32_t k = 0; k < results.size(); ++k) {
                e_raw << net->kind() << ","
                      << models.at(j).first << ","
                      << results.at(k).first << ","
                      << results.at(k).second.at(0) << ","
                      << results.at(k).second.at(1) << ","
                      << results.at(k).second.at(2) << "\n";

                std::cout << results.at(k).first << ": "
                          << results.at(k).second.at(0) << " ("
                          << results.at(k).second.at(1) << ")  ["
                          << results.at(k).second.at(2) << "]"
                          << std::endl;

                e_read << results.at(k).first << ": "
                       << results.at(k).second.at(0) << " ("
                       << results.at(k).second.at(1) << ")  ["
                       << results.at(k).second.at(2) << "]"
                       << "\n";

                e_read_net << results.at(k).first << ": "
                           << results.at(k).second.at(0) << " ("
                           << results.at(k).second.at(1) << ")  ["
                           << results.at(k).second.at(2) << "]"
                           << "\n";
                e_read_all << results.at(k).first << ": "
                           << results.at(k).second.at(0) << " ("
                           << results.at(k).second.at(1) << ")  ["
                           << results.at(k).second.at(2) << "]"
                           << "\n";
            }
        }
    }

    tk.finished();

    return 0;
}
