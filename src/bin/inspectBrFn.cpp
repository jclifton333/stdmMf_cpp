#include "network.hpp"
#include "model.hpp"
#include "infShieldStateNoImNoSoModel.hpp"
#include "system.hpp"
#include "features.hpp"
#include "networkRunSymFeatures.hpp"
#include "sweepAgent.hpp"
#include "brMinSimPerturbAgent.hpp"

#include "proximalAgent.hpp"
#include "randomAgent.hpp"
#include "epsAgent.hpp"

#include "objFns.hpp"

#include <njm_cpp/tools/random.hpp>
#include <njm_cpp/linalg/stdVectorAlgebra.hpp>
#include <njm_cpp/data/trapperKeeper.hpp>
#include <njm_cpp/tools/experiment.hpp>
#include <njm_cpp/tools/progress.hpp>
#include <njm_cpp/info/project.hpp>

#include <algorithm>

#include <thread>

#include <glog/logging.h>

#include <boost/accumulators/accumulators.hpp>
#include <boost/accumulators/statistics.hpp>

using namespace stdmMf;


void run(const uint32_t & rep, const std::shared_ptr<const Network> & network,
        const std::shared_ptr<Model<InfShieldState> > & model,
        const uint32_t & num_obs,
        const uint32_t & run_length,
        const bool & gs_step,
        const bool & sq_total_br,
        njm::data::Entry * const entry) {
    std::shared_ptr<njm::tools::Rng> rng(new njm::tools::Rng);
    rng->seed(rep);

    // system
    System<InfShieldState> s(network, model);
    s.rng(rng);

    // features
    std::shared_ptr<Features<InfShieldState> > features(
            new NetworkRunSymFeatures<InfShieldState>(network, run_length));

    // eps agent
    std::shared_ptr<ProximalAgent<InfShieldState> > pa(
            new ProximalAgent<InfShieldState>(network));
    pa->rng(rng);
    std::shared_ptr<RandomAgent<InfShieldState> > ra(
            new RandomAgent<InfShieldState>(network));
    ra->rng(rng);
    EpsAgent<InfShieldState> ea(network, pa, ra, 0.2);
    ea.rng(rng);

    // set initial infections
    s.start();
    // simulate history
    for (uint32_t i = 0; i < num_obs; ++i) {
        const boost::dynamic_bitset<> trt_bits = ea.apply_trt(s.state(),
                s.history());

        s.trt_bits(trt_bits);

        s.turn_clock();
    }

    const std::vector<Transition<InfShieldState> > all_history(
            Transition<InfShieldState>::from_sequence(s.history(), s.state()));

    BrMinSimPerturbAgent<InfShieldState> brAgent(network->clone(),
            features->clone(), 0.10, 0.25, 1.41, 1.0, 0.85, 0.00397, false,
            gs_step, sq_total_br);
    brAgent.rng(rng);
    brAgent.record(true);

    brAgent.train(all_history);

    const std::vector<std::pair<double, std::vector<double> > > train_history(
            brAgent.train_history());
    const uint32_t train_size(train_history.size());
    const uint32_t value_mc_reps(100);

    typedef boost::accumulators::features<
        boost::accumulators::tag::mean,
        boost::accumulators::tag::variance> MeanVarFeatures;
    typedef boost::accumulators::accumulator_set<double, MeanVarFeatures>
        MeanVarAccumulator;

    const std::vector<double> gamma({1.0, 0.95, 0.9});

    for (uint32_t i = 0; i < train_size; ++i) {
        if (i % 10 == 0 || (i + 1 == train_size)) {
            const std::vector<double> & par(train_history.at(i).second);
            const double & obj_fn(train_history.at(i).first);

            // bellman residual
            *entry << rep << ", "
                   << num_obs << ", "
                   << run_length << ", "
                   << gs_step << ", "
                   << sq_total_br << ", "
                   << i << ", "
                   << "br" << ", "
                   << "NA" << ", "
                   << obj_fn << ", "
                   << "NA" << "\n";

            SweepAgent<InfShieldState> agent(network->clone(),
                    features->clone(), par, 2, false);
            agent.rng(rng);
            for (uint32_t g = 0; g < gamma.size(); ++g) {
                MeanVarAccumulator acc;
                for (uint32_t rep = 0; rep < value_mc_reps; ++rep) {
                    rng->seed(rep);
                    s.start();
                    acc(runner(&s, &agent, 100, gamma.at(g)));
                }

                // value function
                *entry << rep << ", "
                       << num_obs << ", "
                       << run_length << ", "
                       << gs_step << ", "
                       << sq_total_br << ", "
                       << i << ", "
                       << "value" << ", "
                       << gamma.at(g) << ", "
                       << boost::accumulators::mean(acc) << ", "
                       << boost::accumulators::variance(acc) << "\n";
            }
        }
    }
}




int main(int argc, char *argv[]) {
    // setup networks
    std::shared_ptr<Network> network;
    { // network 1
        NetworkInit init;
        init.set_dim_x(10);
        init.set_dim_y(10);
        init.set_wrap(false);
        init.set_type(NetworkInit_NetType_GRID);
        network = Network::gen_network(init);
    }

    std::shared_ptr<Model<InfShieldState> > model(
            new InfShieldStateNoImNoSoModel(
                    network));

    {
        // latent infections
        const double prob_inf_latent = 0.01;
        const double intcp_inf_latent =
            std::log(1. / (1. - prob_inf_latent) - 1);

        // neighbor infections
        const double prob_inf = 0.5;
        const uint32_t prob_num_neigh = 3;
        const double intcp_inf =
            std::log(std::pow(1. - prob_inf, -1. / prob_num_neigh) - 1.);

        const double trt_act_inf =
            std::log(std::pow(1. - prob_inf * 0.25, -1. / prob_num_neigh) - 1.)
            - intcp_inf;

        const double trt_pre_inf =
            std::log(std::pow(1. - prob_inf * 0.75, -1. / prob_num_neigh) - 1.)
            - intcp_inf;

        // recovery
        const double prob_rec = 0.25;
        const double intcp_rec = std::log(1. / (1. - prob_rec) - 1.);
        const double trt_act_rec =
            std::log(1. / ((1. - prob_rec) * 0.5) - 1.) - intcp_rec;

        // shield
        const double shield_coef = 0.9;


        std::vector<double> par =
            {intcp_inf_latent,
             intcp_inf,
             intcp_rec,
             trt_act_inf,
             trt_act_rec,
             trt_pre_inf,
             shield_coef};

        std::vector<double> par_sep =
            {intcp_inf_latent,
             intcp_inf,
             intcp_rec,
             trt_act_inf,
             -trt_act_inf,
             trt_act_rec,
             -trt_act_rec,
             trt_pre_inf,
             -trt_pre_inf,
             shield_coef};

        // set parameters
        model->par(par);
    }

    njm::tools::Experiment e;

    {
        njm::tools::Experiment::FactorGroup * g = e.add_group();

        g->add_factor(std::vector<int>({5, 10, 50, 100, 500, 1000})); // num_obs
        g->add_factor(std::vector<int>({1, 2, 3})); // run_length
        g->add_factor(std::vector<bool>({false, true})); // gs_step
        g->add_factor(std::vector<bool>({false, true})); // sq_total_br
    }

    njm::thread::Pool p(std::thread::hardware_concurrency());

    std::shared_ptr<njm::tools::Progress<std::ostream> > progress(
            new njm::tools::Progress<std::ostream>(&std::cout));

    njm::data::TrapperKeeper tk(argv[0],
            njm::info::project::PROJECT_ROOT_DIR + "/data");

    *tk.entry("inspectBrFn_results.csv")
        << "rep, "
        << "num_obs, "
        << "iter, "
        << "fn_type, "
        << "gamma, "
        << "fn_value, "
        << "fn_error\n";


    e.start();

    uint32_t num_jobs = 0;
    do {
        const njm::tools::Experiment::Factor f = e.get();


        for (uint32_t rep = 0; rep < 50; ++rep) {
            uint32_t i = 0;
            CHECK_EQ(f.at(i).type,
                    njm::tools::Experiment::FactorLevel::Type::is_int);
            const uint32_t num_obs = static_cast<uint32_t>(
                    f.at(i++).val.int_val);
            CHECK_EQ(f.at(i).type,
                    njm::tools::Experiment::FactorLevel::Type::is_int);
            const uint32_t run_length = static_cast<uint32_t>(
                    f.at(i++).val.int_val);
            CHECK_EQ(f.at(i).type,
                    njm::tools::Experiment::FactorLevel::Type::is_bool);
            const bool gs_step = f.at(i++).val.bool_val;
            CHECK_EQ(f.at(i).type,
                    njm::tools::Experiment::FactorLevel::Type::is_bool);
            const bool sq_total_br = f.at(i++).val.bool_val;

            CHECK_EQ(i, f.size());


            njm::data::Entry * new_entry = tk.entry(
                    "inspectBrFn_results.csv");

            p.service().post([=]() {
                run(rep, network->clone(), model->clone(),
                        num_obs, run_length, gs_step, sq_total_br, new_entry);
                progress->update();
            });

            ++num_jobs;
        }

    } while (e.next());

    progress->total(num_jobs);

    p.join();
    progress->done();

    tk.finished();

    return 0;
}
