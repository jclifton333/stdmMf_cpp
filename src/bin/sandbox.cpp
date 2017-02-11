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

    NetworkInit init;
    init.set_dim_x(10);
    init.set_dim_y(10);
    init.set_wrap(false);
    init.set_type(NetworkInit_NetType_GRID);

    const std::shared_ptr<Network> net(Network::gen_network(init));

    const std::shared_ptr<Model> mod(new NoCovEdgeModel(net));
    mod->par({-4.0, -4.0, -1.5, -8.0, 2.0, -8.0});


    // vfn max
    std::vector<std::shared_ptr<Result<double> > > vfn_val;
    std::vector<std::shared_ptr<Result<double> > > vfn_time;

    std::shared_ptr<Result<double> > r_val(new Result<double>);
    std::shared_ptr<Result<double> > r_time(new Result<double>);
    vfn_val.push_back(r_val);
    vfn_time.push_back(r_time);

    System s(net->clone(), mod->clone());
    s.seed(0);
    VfnMaxSimPerturbAgent a(net->clone(),
            std::shared_ptr<Features>(
                    new NetworkRunFeatures(net->clone(), 4)),
            std::shared_ptr<Model>(
                    new NoCovEdgeModel(net->clone())),
            2, 20, 10.0, 0.1, 5, 1, 0.4, 0.7);
    a.seed(0);

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

    return 0;
}
