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

    Pool pool(std::thread::hardware_concurrency());

    const uint32_t num_reps = 50;

    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> net(Network::gen_network(init));

    const std::shared_ptr<Model> mod(new NoCovEdgeModel(net));
    mod->par({-4.0, -4.0, -1.5, -8.0, 2.0, -8.0});


    // none
    std::vector<std::shared_ptr<Result<double> > > none;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r(new Result<double>);
        none.push_back(r);

        pool.service()->post([=](){
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    NoTrtAgent a(net->clone());

                    s.start();

                    r->set(runner(&s, &a, 20, 1.0));
                });
    }

    // random
    std::vector<std::shared_ptr<Result<double> > > random;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r(new Result<double>);
        random.push_back(r);

        pool.service()->post([=](){
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    RandomAgent a(net->clone());
                    a.set_seed(i);

                    s.start();

                    r->set(runner(&s, &a, 20, 1.0));
                });
    }


    // proximal
    std::vector<std::shared_ptr<Result<double> > > proximal;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r(new Result<double>);
        proximal.push_back(r);

        pool.service()->post([=]() {
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    ProximalAgent a(net->clone());
                    a.set_seed(i);

                    s.start();

                    r->set(runner(&s, &a, 20, 1.0));
                });
    }


    // myopic
    std::vector<std::shared_ptr<Result<double> > > myopic;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r(new Result<double>);
        myopic.push_back(r);

        pool.service()->post([=]() {
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    MyopicAgent a(net->clone(), std::shared_ptr<Model>(
                                    new NoCovEdgeModel(net->clone())));
                    a.set_seed(i);

                    s.start();

                    r->set(runner(&s, &a, 20, 1.0));
                });
    }


    // // vfn max
    // std::vector<std::shared_ptr<Result<double> > > vfn;
    // for (uint32_t i = 0; i < num_reps; ++i) {
    //     std::shared_ptr<Result<double> > r(new Result<double>);
    //     vfn.push_back(r);

    //     pool.service()->post([=]() {
    //                 System s(net->clone(), mod->clone());
    //                 s.set_seed(i);
    //                 VfnMaxSimPerturbAgent a(net->clone(),
    //                         std::shared_ptr<Features>(
    //                                 new NetworkRunFeatures(net->clone(), 4)),
    //                         std::shared_ptr<Model>(
    //                                 new NoCovEdgeModel(net->clone())),
    //                         2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7);
    //                 a.set_seed(i);

    //                 s.start();

    //                 r->set(runner(&s, &a, 20, 1.0));
    //             });
    // }


    // br min
    std::vector<std::shared_ptr<Result<double> > > br;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r(new Result<double>);
        br.push_back(r);

        pool.service()->post([=]() {
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    BrMinSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunFeatures(net->clone(), 4)),
                            1e-06, 0.2, 5e-06, 1, 0.5, 3e-7);
                    a.set_seed(i);

                    s.start();

                    r->set(runner(&s, &a, 20, 1.0));
                });
    }

    // // vr max br min adapt
    // std::vector<std::shared_ptr<Result<double> > > adapt;
    // for (uint32_t i = 0; i < num_reps; ++i) {
    //     std::shared_ptr<Result<double> > r(new Result<double>);
    //     adapt.push_back(r);

    //     pool.service()->post([=]() {
    //                 System s(net->clone(), mod->clone());
    //                 s.set_seed(i);
    //                 VfnBrAdaptSimPerturbAgent a(net->clone(),
    //                         std::shared_ptr<Features>(
    //                                 new NetworkRunFeatures(net->clone(), 4)),
    //                         std::shared_ptr<Model>(
    //                                 new NoCovEdgeModel(net->clone())),
    //                         2, 20, 10.0, 1.0, 1, 1, 0.4, 0.3,
    //                         1e-06, 0.2, 5e-06, 1, 0.5, 3e-7);
    //                 a.set_seed(i);

    //                 s.start();

    //                 r->set(runner(&s, &a, 20, 1.0));
    //             });
    // }


    pool.join();

    const std::pair<double, double> none_stats = mean_and_var(
            result_to_vec(none));
    std::cout << "none: "
              << none_stats.first
              << " (" << std::sqrt(none_stats.second / num_reps) << ")"
              << std::endl;

    const std::pair<double, double> random_stats = mean_and_var(
            result_to_vec(random));
    std::cout << "random: "
              << random_stats.first
              << " (" << std::sqrt(random_stats.second / num_reps) << ")"
              << std::endl;

    const std::pair<double, double> proximal_stats = mean_and_var(
            result_to_vec(proximal));
    std::cout << "proximal: "
              << proximal_stats.first
              << " (" << std::sqrt(proximal_stats.second / num_reps) << ")"
              << std::endl;

    const std::pair<double, double> myopic_stats = mean_and_var(
            result_to_vec(myopic));
    std::cout << "myopic: "
              << myopic_stats.first
              << " (" << std::sqrt(myopic_stats.second / num_reps) << ")"
              << std::endl;

    const std::pair<double, double> vfn_stats = mean_and_var(
            result_to_vec(vfn));
    std::cout << "vfn: "
              << vfn_stats.first
              << " (" << std::sqrt(vfn_stats.second / num_reps) << ")"
              << std::endl;

    const std::pair<double, double> br_stats = mean_and_var(
            result_to_vec(br));
    std::cout << "br: "
              << br_stats.first
              << " (" << std::sqrt(br_stats.second / num_reps) << ")"
              << std::endl;

    const std::pair<double, double> adapt_stats = mean_and_var(
            result_to_vec(adapt));
    std::cout << "adapt: "
              << adapt_stats.first
              << " (" << std::sqrt(adapt_stats.second / num_reps) << ")"
              << std::endl;


    return 0;
}
