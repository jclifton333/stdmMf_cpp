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

    std::vector<double> par{-3.105, 1.434, 0.051, -1.117, -1.117};

    mod->par(par);

    SweepAgent<EbolaState> a(net,
            std::shared_ptr<Features<EbolaState> >(
                    new EbolaModelFeatures(
                            net, mod->clone())),
            {0.0,
                    -1000.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                    -1.0, 0.0, 0.0, 0.0, 0.0, 0.0},
            njm::linalg::dot_a_and_b, 2, false);

    njm::tools::Rng rng;

    EbolaState state(EbolaState::random(net->size(), rng));

    const boost::dynamic_bitset<> trt_bits(a.apply_trt(state));

    std::cout << "inf: " << (trt_bits & state.inf_bits).count()
              << std::endl;
    state.inf_bits.flip();
    std::cout << "not: " << (trt_bits & state.inf_bits).count()
              << std::endl;

    // const std::vector<double> probs(mod->probs(state,
    //                 boost::dynamic_bitset<>(net->size())));

    // std::vector<std::pair<double, uint32_t> > prob_match;
    // for (uint32_t i = 0; i < net->size(); ++i) {
    //     if (!state.inf_bits.test(i)) {
    //         prob_match.emplace_back(-probs.at(i), i);
    //     }
    // }

    // std::sort(prob_match.begin(), prob_match.end());

    // for (uint32_t i = 0; i < prob_match.size(); ++i) {
    //     std::cout << prob_match.at(i).second
    //               << ": "
    //               << prob_match.at(i).first
    //               << " -> "
    //               << trt_bits.test(prob_match.at(i).second)
    //               << std::endl;
    // }

    return 0;
}
