#include "system.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "infShieldStatePosImNoSoModel.hpp"
#include "noTrtAgent.hpp"
#include "allTrtAgent.hpp"
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
    mod->par(par);

    // mod->est_par(Transition<EbolaState>::from_sequence(obs_history));

    // par = mod->par();

    par = {-4.50446, 1.80575, 0.051293, 0.0, 0.0};

    std::cout << "estimated par:";
    std::for_each(par.begin(), par.end(),
            [] (const double & x_) {std::cout << " " << x_;});
    std::cout << std::endl;

    // sort outbreaks
    std::vector<uint32_t> outbreak_dates;
    for (uint32_t i = 0; i < net->size(); ++i) {
        if (EbolaData::outbreaks().at(i) >= 0) {
            outbreak_dates.push_back(EbolaData::outbreaks().at(i));
        }
    }
    std::sort(outbreak_dates.begin(), outbreak_dates.end());
    const uint32_t outbreaks_cutoff(
            outbreak_dates.at(static_cast<uint32_t>(
                            outbreak_dates.size() * 0.25)));

    EbolaState start_state(EbolaState(net->size()));
    for (uint32_t i = 0; i < net->size(); ++i) {
        if (EbolaData::outbreaks().at(i) >= 0
                && EbolaData::outbreaks().at(i) <= outbreaks_cutoff) {
            start_state.inf_bits.set(i);
        }
    }

    std::cout << "starting infection: "
              << start_state.inf_bits.count()
              << " / " << net->size()
              << " -> "
              << (start_state.inf_bits.count()
                      / static_cast<double>(net->size()))
              << std::endl;

    NoTrtAgent<EbolaState> agent_tune_inf(net);
    AllTrtAgent<EbolaState> agent_tune_trt(net);
    // RandomAgent<EbolaState> agent_tune_trt(net);
    // ProximalAgent<EbolaState> agent_tune_trt(net);
    // MyopicAgent<EbolaState> agent_tune_trt(net, mod);

    const uint32_t time_points(25);
    const uint32_t num_reps(100);

    // tune infection rate
    const double target_tune_inf(0.5);
    bool calibrated(false);
    bool was_above(false);
    double scale = 0.1;
    std::cout << "tuning infection rate" << std::endl;
    std::cout << "target infection: "
              << std::setw(8)
              << std::setfill('0')
              << std::setprecision(6)
              << std::fixed
              << target_tune_inf << std::endl;
    uint32_t iter(0);
    while(!calibrated) {
        mod->par(par);
        System<EbolaState> s(net, mod);
        double avg_inf(0.0);
        for (uint32_t rep = 0; rep < num_reps; ++rep) {
            // set seeds
            s.seed(rep);
            agent_tune_inf.seed(rep);

            // set up starting state
            s.reset();
            s.state(start_state);

            // run
            runner(&s, &agent_tune_inf, time_points, 1.0);

            // record final infections
            avg_inf += s.n_inf();
        }
        avg_inf /= net->size();
        avg_inf /= num_reps;

        std::cout << "\rcurrent: "
                  << std::setw(8)
                  << std::setfill('0')
                  << std::setprecision(6)
                  << std::fixed
                  << avg_inf << std::flush;

        if (std::abs(avg_inf - target_tune_inf) < 0.01) {
            // done
            calibrated = true;
        } else if (avg_inf > target_tune_inf) {
            // reduce rate of spread
            par.at(0) *= 1.0 + scale;
            par.at(1) = std::log(std::exp(par.at(1)) * (1.0 + scale));
            if (!was_above) {
                // jumping across target, decrease scale size
                scale *= 0.9;
            }
            was_above = true;
        } else {
            // increase rate of spread
            par.at(0) /= 1.0 + scale;
            par.at(1) = std::log(std::exp(par.at(1)) / (1.0 + scale));
            if (was_above) {
                // jumping across target, decrease scale size
                scale *= 0.9;
            }
            was_above = false;
        }
    }
    std::cout << std::endl;

    // print current par
    std::cout << "par:";
    std::for_each(par.begin(), par.end(),
            [] (const double & x_) {std::cout << " " << x_;});
    std::cout << std::endl;


    // tune treatment effect size
    par.at(3) = par.at(4) = -10.0;
    const double start_inf = start_state.inf_bits.count()
        / static_cast<double>(net->size());
    const double target_tune_trt(start_inf
            + 0.05 * (target_tune_inf - start_inf));
    calibrated = false;
    was_above = false;
    scale = 0.1;
    std::cout << "tuning treatment effect size" << std::endl;
    std::cout << "target infection: "
              << std::setw(8)
              << std::setfill('0')
              << std::setprecision(6)
              << std::fixed
              << target_tune_trt << std::endl;
    iter = 0;
    while(!calibrated) {
        mod->par(par);
        System<EbolaState> s(net, mod);
        double avg_inf(0.0);
        for (uint32_t rep = 0; rep < num_reps; ++rep) {
            // set seeds
            s.seed(rep);
            agent_tune_trt.seed(rep);

            // set up starting state
            s.reset();
            s.state(start_state);

            // run
            runner(&s, &agent_tune_trt, time_points, 1.0);

            // record final infections
            avg_inf += s.n_inf();
        }
        avg_inf /= net->size();
        avg_inf /= num_reps;

        std::cout << "\rcurrent: "
                  << std::setw(8)
                  << std::setfill('0')
                  << std::setprecision(6)
                  << std::fixed
                  << avg_inf
                  << " ("
                  << std::setw(3)
                  << iter
                  << " -> "
                  << std::setw(6)
                  << std::setfill('0')
                  << std::setprecision(3)
                  << std::fixed
                  << par.at(3)
                  << ")" << std::flush;

        if (std::abs(avg_inf - target_tune_trt) < 0.01) {
            // done
            calibrated = true;
        } else if (avg_inf > target_tune_trt) {
            // increase treatment effect size
            par.at(3) *= 1.0 + scale;
            par.at(4) *= 1.0 + scale;
            if (!was_above) {
                // jumping across target, decrease scale size
                scale *= 0.9;
            }
            was_above = true;
        } else {
            // decrease treatment effect size
            par.at(3) /= 1.0 + scale;
            par.at(4) /= 1.0 + scale;
            if (was_above) {
                // jumping across target, decrease scale size
                scale *= 0.9;
            }
            was_above = false;
        }
        ++iter;
    }
    std::cout << std::endl;

    // print final par
    std::cout << "final par:";
    std::for_each(par.begin(), par.end(),
            [] (const double & x_) {std::cout << " " << x_;});
    std::cout << std::endl;


    // evalute policies
    {
        NoTrtAgent<EbolaState> agent_notrt(net);
        System<EbolaState> s(net, mod);
        double avg_inf(0.0);
        for (uint32_t rep = 0; rep < num_reps; ++rep) {
            // set seeds
            s.seed(rep);
            agent_tune_trt.seed(rep);

            // set up starting state
            s.reset();
            s.state(start_state);

            // run
            runner(&s, &agent_notrt, time_points, 1.0);

            // record final infections
            avg_inf += s.n_inf();
        }
        avg_inf /= net->size();
        avg_inf /= num_reps;
        std::cout << "no trt: "
                  << std::setw(8)
                  << std::setfill('0')
                  << std::setprecision(6)
                  << std::fixed
                  << avg_inf << std::endl;
    }

    {
        RandomAgent<EbolaState> agent_random(net);
        System<EbolaState> s(net, mod);
        double avg_inf(0.0);
        for (uint32_t rep = 0; rep < num_reps; ++rep) {
            // set seeds
            s.seed(rep);
            agent_tune_trt.seed(rep);

            // set up starting state
            s.reset();
            s.state(start_state);

            // run
            runner(&s, &agent_random, time_points, 1.0);

            // record final infections
            avg_inf += s.n_inf();
        }
        avg_inf /= net->size();
        avg_inf /= num_reps;
        std::cout << "random: "
                  << std::setw(8)
                  << std::setfill('0')
                  << std::setprecision(6)
                  << std::fixed
                  << avg_inf << std::endl;
    }

    {
        ProximalAgent<EbolaState> agent_proximal(net);
        System<EbolaState> s(net, mod);
        double avg_inf(0.0);
        for (uint32_t rep = 0; rep < num_reps; ++rep) {
            // set seeds
            s.seed(rep);
            agent_tune_trt.seed(rep);

            // set up starting state
            s.reset();
            s.state(start_state);

            // run
            runner(&s, &agent_proximal, time_points, 1.0);

            // record final infections
            avg_inf += s.n_inf();
        }
        avg_inf /= net->size();
        avg_inf /= num_reps;
        std::cout << "proximal: "
                  << std::setw(8)
                  << std::setfill('0')
                  << std::setprecision(6)
                  << std::fixed
                  << avg_inf << std::endl;

    }

    return 0;
}
