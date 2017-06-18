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

    std::vector<StateAndTrt<EbolaState> > obs_history;

    // init model
    const std::shared_ptr<EbolaStateGravityModel> mod(
            new EbolaStateGravityModel(net));

    const std::vector<int> & outbreaks(EbolaData::outbreaks());
    CHECK_EQ(outbreaks.size(), net->size());
    // preallocate history
    const int & max_outbreak(*std::max_element(
                    outbreaks.begin(), outbreaks.end()));
    for (uint32_t i = 0; i < (max_outbreak + 1); ++i) {
        obs_history.emplace_back(EbolaState(net->size()),
                boost::dynamic_bitset<>(net->size()));
    }

    // fill in history
    for (uint32_t i = 0; i < net->size(); ++i) {
        const int outbreak(outbreaks.at(i));
        for (uint32_t j = 0; j < obs_history.size(); ++j) {
            EbolaState & state(obs_history.at(j).state);
            CHECK(!state.inf_bits.test(i));
            if (j >= outbreak && outbreak >= 0 && outbreak <= 157) {
                state.inf_bits.flip(i);
            }
        }
    }

    std::vector<double> par(mod->par_size(), 0.0);
    // par.at(0) = -5.25;
    // par.at(1) = -std::log(156);
    // par.at(2) = 0.186;
    mod->par(par);

    mod->est_par(Transition<EbolaState>::from_sequence(obs_history));

    par = mod->par();

    std::cout << "par:";
    std::for_each(par.begin(), par.end(),
            [] (const double & x_) {std::cout << " " << x_;});
    std::cout << std::endl;

    return 0;
}
