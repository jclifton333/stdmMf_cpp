#include "system.hpp"
#include "ebolaStateGravityModel.hpp"
#include "mixtureModel.hpp"
#include "noTrtAgent.hpp"
#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "myopicAgent.hpp"
#include "vfnMaxSimPerturbAgent.hpp"
#include "brMinSimPerturbAgent.hpp"
#include "vfnBrAdaptSimPerturbAgent.hpp"
#include "vfnBrStartSimPerturbAgent.hpp"
#include "brMinIterSimPerturbAgent.hpp"

#include "brMinWtdSimPerturbAgent.hpp"

#include "ebolaData.hpp"
#include "ebolaFeatures.hpp"
#include "ebolaBinnedFeatures.hpp"
#include "ebolaModelFeatures.hpp"
#include "ebolaTransProbFeatures.hpp"
#include "networkRunSymFeatures.hpp"

#include "finiteQfnFeatures.hpp"

#include "objFns.hpp"

#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/thread/pool.hpp>
#include <njm_cpp/info/project.hpp>
#include <njm_cpp/tools/stats.hpp>

#include <njm_cpp/tools/progress.hpp>

#include <future>

#include <thread>

#include <fstream>

#include <chrono>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics/stats.hpp>
#include <boost/accumulators/statistics/mean.hpp>
#include <boost/accumulators/statistics/variance.hpp>

using namespace stdmMf;


int main(int argc, char *argv[]) {
    google::InitGoogleLogging(argv[0]);

    EbolaData::init();

    NetworkInit init;
    init.set_type(NetworkInit_NetType_EBOLA);
    std::shared_ptr<const Network> net(Network::gen_network(init));

    std::vector<double> grav_par{-7.4429e+00, -2.8362e-01, -1.4912e-06,
                                 -1.0153e+00, -1.0153e+00};

    std::shared_ptr<Model<EbolaState> > mod_system(
            new EbolaStateGravityModel(net));
    mod_system->par(grav_par);

    std::shared_ptr<Model<EbolaState> > mod_agents(
            new EbolaStateGravityModel(net));


    // sort outbreaks
    const double starting_prop(0.25);
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


    const uint32_t num_reps = 50;
    const uint32_t time_points = 25;

    njm::thread::Pool pool(std::thread::hardware_concurrency());

    njm::data::TrapperKeeper tk(argv[0],
            njm::info::project::PROJECT_ROOT_DIR + "/data");
    tk.print_data_dir();

    njm::tools::Progress<std::ostream> progress(num_reps * 2, &std::cout);

    std::vector<std::vector<std::vector<double> > > optim_par_history_emf(
            num_reps);
    std::vector<double> values_emf(num_reps);

    std::vector<std::vector<std::vector<double> > > optim_par_history_etpf(
            num_reps);
    std::vector<double> values_etpf(num_reps);

    std::mutex mtx;



    for (uint32_t i = 0; i < num_reps; ++i) {
        pool.service().post([=, &progress, &mtx,
                        &optim_par_history_emf, &values_emf]() {
            System<EbolaState> s(net, mod_system->clone());
            s.seed(i);
            VfnMaxSimPerturbAgent<EbolaState> a(net,
                    std::shared_ptr<Features<EbolaState> >(
                            new EbolaModelFeatures(
                                    net, mod_agents->clone())),
                    mod_agents->clone(),
                    2, time_points, 1, 10.0, 0.1, 10, 1, 0.4, 1.20);
            a.seed(i);

            s.reset();
            s.state(start_state);

            const double value(runner(&s, &a, time_points, 1.0));

            std::lock_guard<std::mutex> lock(mtx);
            optim_par_history_emf.at(i) = a.history();
            values_emf.at(i) = value;

            progress.update();
        });
    }

    for (uint32_t i = 0; i < num_reps; ++i) {
        pool.service().post([=, &progress, &mtx,
                        &optim_par_history_etpf, &values_etpf]() {
            System<EbolaState> s(net, mod_system->clone());
            s.seed(i);
            VfnMaxSimPerturbAgent<EbolaState> a(net,
                    std::shared_ptr<Features<EbolaState> >(
                            new EbolaTransProbFeatures(
                                    net, mod_agents->clone())),
                    mod_agents->clone(),
                    2, time_points, 1, 10.0, 0.1, 10, 1, 0.4, 1.20);
            a.seed(i);

            s.reset();
            s.state(start_state);

            const double value(runner(&s, &a, time_points, 1.0));

            std::lock_guard<std::mutex> lock(mtx);
            optim_par_history_etpf.at(i) = a.history();
            values_etpf.at(i) = value;

            progress.update();
        });
    }

    pool.join();
    progress.done();

    // write data
    njm::data::Entry * values_emf_entry(tk.entry("values_emf.txt"));
    njm::data::Entry * coefs_emf_entry(tk.entry("coefs_emf.txt"));

    *values_emf_entry << "rep" << ","
                  << "value" << "\n";

    *coefs_emf_entry << "rep" << ","
                 << "time" << ","
                 << "index" << ","
                 << "coef" << "\n";
    for (uint32_t i = 0; i < num_reps; ++i) {
        *values_emf_entry << i << ","
                      << values_emf.at(i) << "\n";

        const uint32_t history_len(optim_par_history_emf.at(i).size());
        for (uint32_t j = 0; j < history_len; ++j) {
            const uint32_t num_coef(optim_par_history_emf.at(i).at(j).size());
            for (uint32_t k = 0; k < num_coef; ++k) {
                *coefs_emf_entry << i << ","
                             << j << ","
                             << k << ","
                             << optim_par_history_emf.at(i).at(j).at(k)
                             << "\n";
            }
        }
    }

    njm::data::Entry * values_etpf_entry(tk.entry("values_etpf.txt"));
    njm::data::Entry * coefs_etpf_entry(tk.entry("coefs_etpf.txt"));

    *values_etpf_entry << "rep" << ","
                  << "value" << "\n";

    *coefs_etpf_entry << "rep" << ","
                 << "time" << ","
                 << "index" << ","
                 << "coef" << "\n";
    for (uint32_t i = 0; i < num_reps; ++i) {
        *values_etpf_entry << i << ","
                      << values_etpf.at(i) << "\n";

        const uint32_t history_len(optim_par_history_etpf.at(i).size());
        for (uint32_t j = 0; j < history_len; ++j) {
            const uint32_t num_coef(optim_par_history_etpf.at(i).at(j).size());
            for (uint32_t k = 0; k < num_coef; ++k) {
                *coefs_etpf_entry << i << ","
                             << j << ","
                             << k << ","
                             << optim_par_history_etpf.at(i).at(j).at(k)
                             << "\n";
            }
        }
    }

    tk.finished();
    tk.print_data_dir();
}
