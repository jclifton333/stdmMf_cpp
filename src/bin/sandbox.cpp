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

#include "ebolaFeatures.hpp"

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

    std::shared_ptr<Network> net(Network::gen_network(init));

    const uint32_t time_points(50);

    // init model
    const std::shared_ptr<EbolaStateGravityModel> mod(
            new EbolaStateGravityModel(net));

    System<EbolaState> s(net, mod->clone());
    s.seed(0);
    VfnMaxSimPerturbAgent<EbolaState> a(net,
            std::shared_ptr<Features<EbolaState> >(
                    new FiniteQfnFeatures<EbolaState>(
                            net, {mod->clone()},
                            std::shared_ptr<Features<EbolaState> >(
                                    new EbolaFeatures(
                                            net, 20, 10)), 1)),
            mod->clone(),
            2, time_points, 10.0, 0.1, 5, 1, 0.4, 0.7);
    a.seed(0);

    s.start();

    runner(&s, &a, time_points, 1.0);

    // EbolaFeatures ef(net, 2, 1);

    // s.start();
    // const std::vector<double> feat(ef.get_features(s.state(),
    //                 boost::dynamic_bitset<>(net->size())));

    return 0;
}
