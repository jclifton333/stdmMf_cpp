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
#include "ebolaModelFeatures.hpp"
#include "ebolaBinnedFeatures.hpp"
#include "ebolaTransProbFeatures.hpp"

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

    // init model
    const std::shared_ptr<EbolaStateGravityModel> mod(
            new EbolaStateGravityModel(net));

    std::vector<double> par{-7.443e+00, -2.836e-01, -1.491e-06,
                            -1.015e+00, -1.015e+00};


    mod->par(par);

    const uint32_t time_points(25);

    System<EbolaState> s(net, mod->clone());
    s.seed(1);
    VfnMaxSimPerturbAgent<EbolaState> a(net,
            std::shared_ptr<Features<EbolaState> >(
                    new EbolaTransProbFeatures(
                            net, mod->clone())),
            mod->clone(),
            2, time_points, 1, 10.0, 0.1, 10, 1, 0.4, 1.2);
    a.seed(1);

    const double starting_prop(0.0);
    std::vector<uint32_t> outbreak_dates;
    for (uint32_t i = 0; i < EbolaData::outbreaks().size(); ++i) {
        if (EbolaData::outbreaks().at(i) >= 0) {
            outbreak_dates.push_back(EbolaData::outbreaks().at(i));
        }
    }
    std::sort(outbreak_dates.begin(), outbreak_dates.end());
    const uint32_t last_index(
            std::min(std::max(1u,
                            static_cast<uint32_t>(
                                    outbreak_dates.size()
                                    * starting_prop)),
                    static_cast<uint32_t>(outbreak_dates.size() - 1u)));
    const uint32_t outbreaks_cutoff(outbreak_dates.at(last_index));

    EbolaState start_state(EbolaState(EbolaData::outbreaks().size()));
    for (uint32_t i = 0; i < EbolaData::outbreaks().size(); ++i) {
        if (EbolaData::outbreaks().at(i) >= 0
                && EbolaData::outbreaks().at(i) <= outbreaks_cutoff) {
            start_state.inf_bits.set(i);
        }
    }

    s.reset();
    s.state(start_state);

    runner(&s, &a, time_points, 1.0);

    return 0;
}
