#include "system.hpp"
#include "noCovEdgeModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"

#include "networkRunFeatures.hpp"

#include "objFns.hpp"

#include "result.hpp"

#include "pool.hpp"

#include <thread>

using namespace stdmMf;

int main(int argc, char *argv[]) {

    // Pool pool(std::thread::hardware_concurrency());
    Pool pool(50);

    const uint32_t num_reps = 50;

    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> net(Network::gen_network(init));

    const std::shared_ptr<Model> mod(new NoCovEdgeModel(net));
    mod->par({-4.0, -2.0, -1.5, -0.25, 0.25, -4.0});


    // none
    std::vector<std::shared_ptr<Result<double> > > none_val;
    std::vector<std::shared_ptr<Result<double> > > none_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        none_val.push_back(r_val);
        none_time.push_back(r_time);

        pool.service()->post([=](){
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    NoTrtAgent a(net->clone());

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

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
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    RandomAgent a(net->clone());
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

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
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    ProximalAgent a(net->clone());
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

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
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    MyopicAgent a(net->clone(), std::shared_ptr<Model>(
                                    new NoCovEdgeModel(net->clone())));
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }


    // vfn max
    std::vector<std::shared_ptr<Result<double> > > vfn_val;
    std::vector<std::shared_ptr<Result<double> > > vfn_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        vfn_val.push_back(r_val);
        vfn_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    VfnMaxSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunFeatures(net->clone(), 1)),
                            std::shared_ptr<Model>(
                                    new NoCovEdgeModel(net->clone())),
                            2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7);
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                        std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }


    // br min
    std::vector<std::shared_ptr<Result<double> > > br_val;
    std::vector<std::shared_ptr<Result<double> > > br_time;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r_val(new Result<double>);
        std::shared_ptr<Result<double> > r_time(new Result<double>);
        br_val.push_back(r_val);
        br_time.push_back(r_time);

        pool.service()->post([=]() {
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    BrMinSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunFeatures(net->clone(), 1)),
                            1e-1, 1.0, 1e-3, 1, 0.85, 1e-5);
                    a.set_seed(i);

                    s.start();

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tick =
                        std::chrono::high_resolution_clock::now();

                    r_val->set(runner(&s, &a, 20, 1.0));

                    std::chrono::time_point<
                        std::chrono::high_resolution_clock> tock =
                        std::chrono::high_resolution_clock::now();

                    r_time->set(std::chrono::duration_cast<
                            std::chrono::seconds>(tock - tick).count());
                });
    }

    // // vr max br min adapt
    // std::vector<std::shared_ptr<Result<double> > > adapt_val;
    // std::vector<std::shared_ptr<Result<double> > > adapt_time;
    // for (uint32_t i = 0; i < num_reps; ++i) {
    //     std::shared_ptr<Result<double> > r_val(new Result<double>);
    //     std::shared_ptr<Result<double> > r_time(new Result<double>);
    //     adapt_val.push_back(r_val);
    //     adapt_time.push_back(r_time);

    //     pool.service()->post([=]() {
    //                 System s(net->clone(), mod->clone());
    //                 s.set_seed(i);
    //                 VfnBrAdaptSimPerturbAgent a(net->clone(),
    //                         std::shared_ptr<Features>(
    //                                 new NetworkRunFeatures(net->clone(), 1)),
    //                         std::shared_ptr<Model>(
    //                                 new NoCovEdgeModel(net->clone())),
    //                         2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7,
    //                         1e-1, 1.0, 1e-3, 1, 0.85, 1e-5);
    //                 a.set_seed(i);

    //                 s.start();

    //                 std::chrono::time_point<
    //                     std::chrono::high_resolution_clock> tick =
    //                     std::chrono::high_resolution_clock::now();

    //                 r_val->set(runner(&s, &a, 20, 1.0));

    //                 std::chrono::time_point<
    //                     std::chrono::high_resolution_clock> tock =
    //                     std::chrono::high_resolution_clock::now();

    //                 r_time->set(std::chrono::duration_cast<
    //                         std::chrono::seconds>(tock - tick).count());
    //             });
    // }


    pool.join();

    const std::pair<double, double> none_stats = mean_and_var(
            result_to_vec(none_val));
    std::cout << "none: "
              << none_stats.first
              << " (" << std::sqrt(none_stats.second / num_reps) << ")"
              << " in " << mean_and_var(result_to_vec(none_time)).first
              << " seconds"
              << std::endl;

    const std::pair<double, double> random_stats = mean_and_var(
            result_to_vec(random_val));
    std::cout << "random: "
              << random_stats.first
              << " (" << std::sqrt(random_stats.second / num_reps) << ")"
              << " in " << mean_and_var(result_to_vec(random_time)).first
              << " seconds"
              << std::endl;

    const std::pair<double, double> proximal_stats = mean_and_var(
            result_to_vec(proximal_val));
    std::cout << "proximal: "
              << proximal_stats.first
              << " (" << std::sqrt(proximal_stats.second / num_reps) << ")"
              << " in " << mean_and_var(result_to_vec(proximal_time)).first
              << " seconds"
              << std::endl;

    const std::pair<double, double> myopic_stats = mean_and_var(
            result_to_vec(myopic_val));
    std::cout << "myopic: "
              << myopic_stats.first
              << " (" << std::sqrt(myopic_stats.second / num_reps) << ")"
              << " in " << mean_and_var(result_to_vec(myopic_time)).first
              << " seconds"
              << std::endl;

    const std::pair<double, double> vfn_stats = mean_and_var(
            result_to_vec(vfn_val));
    std::cout << "vfn: "
              << vfn_stats.first
              << " (" << std::sqrt(vfn_stats.second / num_reps) << ")"
              << " in " << mean_and_var(result_to_vec(vfn_time)).first
              << " seconds"
              << std::endl;

    const std::pair<double, double> br_stats = mean_and_var(
            result_to_vec(br_val));
    std::cout << "br: "
              << br_stats.first
              << " (" << std::sqrt(br_stats.second / num_reps) << ")"
              << " in " << mean_and_var(result_to_vec(br_time)).first
              << " seconds"
              << std::endl;

    // const std::pair<double, double> adapt_stats = mean_and_var(
    //         result_to_vec(adapt_val));
    // std::cout << "adapt: "
    //           << adapt_stats.first
    //           << " (" << std::sqrt(adapt_stats.second / num_reps) << ")"
    //           << " in " << mean_and_var(result_to_vec(adapt_time)).first
    //           << " seconds"
    //           << std::endl;


    return 0;
}
