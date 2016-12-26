#include "system.hpp"
#include "noCovEdgeModel.hpp"
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

                    r->set(runner(&s, &a, 20, 0.9));
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

                    r->set(runner(&s, &a, 20, 0.9));
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

                    r->set(runner(&s, &a, 20, 0.9));
                });
    }


    // vfn max
    std::vector<std::shared_ptr<Result<double> > > vfn;
    for (uint32_t i = 0; i < num_reps; ++i) {
        std::shared_ptr<Result<double> > r(new Result<double>);
        vfn.push_back(r);

        pool.service()->post([=]() {
                    System s(net->clone(), mod->clone());
                    s.set_seed(i);
                    VfnMaxSimPerturbAgent a(net->clone(),
                            std::shared_ptr<Features>(
                                    new NetworkRunFeatures(net->clone(), 4)),
                            std::shared_ptr<Model>(
                                    new NoCovEdgeModel(net->clone())),
                            2, 20, 10.0, 1.0, 1, 1, 0.4, 0.3);
                    a.set_seed(i);

                    s.start();

                    r->set(runner(&s, &a, 20, 0.9));
                });
    }


    // // br min
    // std::vector<std::shared_ptr<Result<double> > > br;
    // for (uint32_t i = 0; i < num_reps; ++i) {
    //     std::shared_ptr<Result<double> > r(new Result<double>);
    //     br.push_back(r);

    //     pool.service()->post([=]() {
    //                 System s(net->clone(), mod->clone());
    //                 s.set_seed(i);
    //                 BrMinSimPerturbAgent a(net->clone(),
    //                         std::shared_ptr<Features>(
    //                                 new NetworkRunFeatures(net->clone(), 4)),
    //                         std::shared_ptr<Model>(
    //                                 new NoCovEdgeModel(net->clone())),
    //                         2, 20, 1e-06, 0.2, 5e-06, 1, 0.5, 3e-7);
    //                 a.set_seed(i);

    //                 s.start();

    //                 r->set(runner(&s, &a, 20, 0.9));
    //             });
    // }

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
    //                         1e-07, 0.2, 1e-07, 1, 0.5, 3e-7);
    //                 a.set_seed(i);

    //                 s.start();

    //                 r->set(runner(&s, &a, 20, 0.9));
    //             });
    // }


    pool.join();

    std::cout << "random: "
              << std::accumulate(random.begin(), random.end(), 0.,
                      [](const double & x,
                              const std::shared_ptr<Result<double> > & r) {
                return x + r->get()/static_cast<double>(num_reps);
            })
              << std::endl;

    std::cout << "proximal: "
              << std::accumulate(proximal.begin(), proximal.end(), 0.,
                      [](const double & x,
                              const std::shared_ptr<Result<double> > & r) {
                          return x + r->get()/static_cast<double>(num_reps);
            })
              << std::endl;

    std::cout << "myopic: "
              << std::accumulate(myopic.begin(), myopic.end(), 0.,
                      [](const double & x,
                              const std::shared_ptr<Result<double> > & r) {
                          return x + r->get()/static_cast<double>(num_reps);
                      })
              << std::endl;

    std::cout << "vfn: "
              << std::accumulate(vfn.begin(), vfn.end(), 0.,
                      [](const double & x,
                              const std::shared_ptr<Result<double> > & r) {
                          return x + r->get()/static_cast<double>(num_reps);
                      })
              << std::endl;

    // std::cout << "br: "
    //           << std::accumulate(br.begin(), br.end(), 0.,
    //                   [](const double & x,
    //                           const std::shared_ptr<Result<double> > & r) {
    //                       return x + r->get()/static_cast<double>(num_reps);
    //                   })
    //           << std::endl;

    // std::cout << "adapt: "
    //           << std::accumulate(adapt.begin(), adapt.end(), 0.,
    //                   [](const double & x,
    //                           const std::shared_ptr<Result<double> > & r) {
    //                       return x + r->get()/static_cast<double>(num_reps);
    //                   })
    //           << std::endl;



    return 0;
}
