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

#include "brMinWtdSimPerturbAgent.hpp"


#include "networkRunSymFeatures.hpp"
#include "finiteQfnFeatures.hpp"

#include "objFns.hpp"

#include "ebolaStateGravityModel.hpp"

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

#include "ebolaData.hpp"

using namespace stdmMf;

int main(int argc, char *argv[]) {
    EbolaData::init();

    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);

    std::shared_ptr<Network> n = Network::gen_network(init);

    // init model
    const std::shared_ptr<EbolaStateGravityModel> m(
            new EbolaStateGravityModel(n));

    // set par
    njm::tools::Rng rng;
    std::vector<double> par(m->par());
    par.at(0) = -5.246;
    par.at(1) = -155.8;
    par.at(2) = 0.186;
    par.at(3) = -2.0;
    par.at(4) = -1.0;
    m->par(par);

    System<EbolaState> s(n,m);
    s.start();

    RandomAgent<EbolaState> ra(n);

    runner(&s, &ra, 1, 1.0);

    const auto history(Transition<EbolaState>::from_sequence(
                    s.history(), s.state()));

    const std::vector<double> grad(m->ll_grad(history));

    for (uint32_t i = 0; i < m->par_size(); ++i) {
        std::cout << grad.at(i) << std::endl;
    }


    return 0;
}
